"""
Router Pipeline  –  Converted from router_pipeline.ipynb
========================================================
Trains a 3-class MLP probe (BENIGN / GCG / PAIR) on hidden-state
features extracted from **LLaMA-7B** (huggyllama/llama-7b).

Improvements over v1:
  - Multi-layer concatenation (combines features from all extracted layers)
  - Mean pooling over all non-padding tokens (not just last token)
  - Deeper MLP with residual connection
  - Class-weighted loss, LR scheduler, early stopping
  - More training epochs (30) with patience-based stopping

Usage:
    python router_pipeline.py --data_path data.csv --save_path router_model.pt
    python router_pipeline.py --data_path data.csv --mode single_layer   # original per-layer sweep
    python router_pipeline.py --data_path data.csv --mode multi_layer    # concat all layers (default)
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import IntEnum
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


# ════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ════════════════════════════════════════════════════════════════════════

MODEL_NAME = "huggyllama/llama-7b"
USE_CHAT_TEMPLATE = False
# More layers for richer multi-layer features
# Use fewer layers to prevent extreme overfitting (65k features is too many for 2k rows)
LAYERS_TO_EXTRACT = [24, 31]  
FEATURE_BATCH_SIZE = 4

TRAIN_BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Early stopping
PATIENCE = 7


# ════════════════════════════════════════════════════════════════════════
# 2. Router Model Definition
# ════════════════════════════════════════════════════════════════════════

class RouterLabel(IntEnum):
    BENIGN = 0
    GCG = 1
    PAIR = 2


@dataclass(frozen=True)
class RouterOutput:
    logits: torch.Tensor       # [B, 3]
    probs: torch.Tensor        # [B, 3] softmax(logits)
    pred_label: torch.Tensor   # [B]   argmax over probs
    action: torch.Tensor       # [B]   routed action label using thresholds


def route_actions(
    probs: torch.Tensor,
    tau_gcg: float = 0.60,
    tau_pair: float = 0.60,
) -> torch.Tensor:
    """
    Convert class probabilities into an action using thresholds.
    """
    if probs.ndim != 2 or probs.size(-1) != 3:
        raise ValueError(f"Expected probs of shape [B,3], got {tuple(probs.shape)}")

    p_gcg = probs[:, RouterLabel.GCG]
    p_pair = probs[:, RouterLabel.PAIR]

    actions = torch.full(
        (probs.size(0),), int(RouterLabel.BENIGN),
        device=probs.device, dtype=torch.long,
    )
    actions = torch.where(
        p_gcg > tau_gcg,
        torch.tensor(int(RouterLabel.GCG), device=probs.device),
        actions,
    )
    actions = torch.where(
        (p_gcg <= tau_gcg) & (p_pair > tau_pair),
        torch.tensor(int(RouterLabel.PAIR), device=probs.device),
        actions,
    )
    return actions


class PromptRouterMLP(nn.Module):
    """
    3-class router with residual connections and deeper architecture.
    x_features shape: [B, D]
    Output classes: BENIGN=0, GCG=1, PAIR=2
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        # A simpler architecture is more stable for massive input dimensions
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x_features: torch.Tensor) -> torch.Tensor:
        if x_features.ndim != 2:
            raise ValueError(f"Expected x_features [B,D], got {tuple(x_features.shape)}")
        return self.net(x_features)

    @torch.no_grad()
    def predict(
        self,
        x_features: torch.Tensor,
        tau_gcg: float = 0.60,
        tau_pair: float = 0.60,
    ) -> RouterOutput:
        logits = self.forward(x_features)
        probs = F.softmax(logits, dim=-1)
        pred_label = probs.argmax(dim=-1)
        action = route_actions(probs, tau_gcg=tau_gcg, tau_pair=tau_pair)
        return RouterOutput(logits=logits, probs=probs, pred_label=pred_label, action=action)


# ════════════════════════════════════════════════════════════════════════
# 3. Data Loading & Preprocessing
# ════════════════════════════════════════════════════════════════════════

def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    print("Sample:")
    print(df.head())

    def get_label(row):
        if row['benign'] == 1:
            return 0
        if row['gcg'] == 1:
            return 1
        if row['pair'] == 1:
            return 2
        return 0

    if {'benign', 'gcg', 'pair'}.issubset(df.columns):
        df['label'] = df.apply(get_label, axis=1)
        print("\nClass counts:")
        print(df['label'].value_counts())
    else:
        raise ValueError("Expected columns 'benign', 'gcg', 'pair' not found in data.")

    return df


# ════════════════════════════════════════════════════════════════════════
# 4. Feature Extraction (LLaMA-7B Hidden States)
# ════════════════════════════════════════════════════════════════════════

def load_llama_model(model_name: str):
    """Load LLaMA-7B model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer


def extract_hidden_states(
    prompts: list,
    model,
    tokenizer,
    layers: list,
    batch_size: int = 4,
    pooling: str = "mean_last",
) -> dict:
    """
    Extract hidden states from specified layers.

    Pooling strategies:
      - "last":      last non-padding token only (original)
      - "mean":      mean pool over all non-padding tokens
      - "mean_last": concatenate mean-pool and last-token (default, most informative)

    Returns dict mapping layer_idx -> tensor of shape [N, feature_dim].
    """
    all_features = {l: [] for l in layers}

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting features"):
        batch_prompts = prompts[i : i + batch_size]

        if USE_CHAT_TEMPLATE and hasattr(tokenizer, 'apply_chat_template'):
            formatted = []
            for p in batch_prompts:
                msgs = [{"role": "user", "content": p}]
                txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                formatted.append(txt)
            inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        else:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        attention_mask = inputs.attention_mask  # [B, seq_len]
        last_token_idxs = attention_mask.sum(dim=1) - 1

        for l in layers:
            if l < len(outputs.hidden_states):
                # Convert to float32 BEFORE summing to prevent float16 overflow to `inf`!
                layer_hidden_fp32 = outputs.hidden_states[l].float()
                B = layer_hidden_fp32.size(0)
                batch_indices = torch.arange(B, device=layer_hidden_fp32.device)

                if pooling == "last":
                    embeds = layer_hidden_fp32[batch_indices, last_token_idxs, :]
                elif pooling == "mean":
                    mask_expanded = attention_mask.unsqueeze(-1).to(torch.float32)
                    sum_hidden = (layer_hidden_fp32 * mask_expanded).sum(dim=1)
                    counts = attention_mask.sum(dim=1, keepdim=True).to(torch.float32)
                    embeds = sum_hidden / counts.clamp(min=1)
                elif pooling == "mean_last":
                    last_embeds = layer_hidden_fp32[batch_indices, last_token_idxs, :]
                    mask_expanded = attention_mask.unsqueeze(-1).to(torch.float32)
                    sum_hidden = (layer_hidden_fp32 * mask_expanded).sum(dim=1)
                    counts = attention_mask.sum(dim=1, keepdim=True).to(torch.float32)
                    mean_embeds = sum_hidden / counts.clamp(min=1)
                    embeds = torch.cat([mean_embeds, last_embeds], dim=-1)
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                all_features[l].append(embeds.cpu())

    results = {}
    for l in layers:
        if all_features[l]:
            layer_feats = torch.cat(all_features[l], dim=0)
            results[l] = layer_feats
    return results


# ════════════════════════════════════════════════════════════════════════
# 5. Dataset
# ════════════════════════════════════════════════════════════════════════

class RouterDataset(Dataset):
    def __init__(self, features, labels, prompts):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.prompts = prompts

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.prompts[idx]


# ════════════════════════════════════════════════════════════════════════
# 6. Compute class weights
# ════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels):
    """Compute inverse-frequency class weights for balanced loss."""
    counts = np.bincount(labels, minlength=3)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════════
# 7. Training (Single-Layer Sweep – original approach)
# ════════════════════════════════════════════════════════════════════════

def train_single_layer(
    cached_features: dict,
    df: pd.DataFrame,
    layers: list,
    embedding_dim: int,
    save_path: str = "router_model.pt",
):
    """Train separate probes per layer, pick the best one."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device} (single-layer sweep)")

    labels = df['label'].values
    prompts = df['prompt'].tolist()
    class_weights = compute_class_weights(labels).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    total_size = len(df)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    best_val_acc = 0.0
    best_layer = -1
    best_model_state = None
    results_log = []

    for layer_idx in layers:
        print(f"\n=== Training Probe on LLaMA Layer {layer_idx} ===")

        features_tensor = cached_features[layer_idx]
        full_dataset = RouterDataset(features_tensor, labels, prompts)
        train_subset, val_subset, _ = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(SPLIT_SEED),
        )

        train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=TRAIN_BATCH_SIZE)

        router = PromptRouterMLP(in_dim=embedding_dim).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        layer_best_val = 0.0
        patience_counter = 0

        for epoch in range(EPOCHS):
            router.train()
            for feats, labs, _ in train_loader:
                feats, labs = feats.to(device), labs.to(device)
                optimizer.zero_grad()
                loss = criterion(router(feats), labs)
                loss.backward()
                optimizer.step()
            scheduler.step()

            router.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for feats, labs, _ in val_loader:
                    feats, labs = feats.to(device), labs.to(device)
                    preds = router(feats).argmax(dim=1)
                    correct += (preds == labs).sum().item()
                    total += labs.size(0)

            current_val_acc = 100.0 * correct / total
            if current_val_acc > layer_best_val:
                layer_best_val = current_val_acc
                patience_counter = 0
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_layer = layer_idx
                    best_model_state = {k: v.clone() for k, v in router.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        print(f"Layer {layer_idx} Best Val Acc: {layer_best_val:.2f}%")
        results_log.append((layer_idx, layer_best_val))

    print(f"\n>>> Best Layer: {best_layer} with Val Acc: {best_val_acc:.2f}% <<<")

    router = PromptRouterMLP(in_dim=embedding_dim).to(device)
    router.load_state_dict(best_model_state)
    router.eval()

    # Test
    features_tensor = cached_features[best_layer]
    full_dataset = RouterDataset(features_tensor, labels, prompts)
    _, _, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    test_loader = DataLoader(test_subset, batch_size=TRAIN_BATCH_SIZE)
    correct = total = 0
    with torch.no_grad():
        for feats, labs, _ in test_loader:
            feats, labs = feats.to(device), labs.to(device)
            preds = router(feats).argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    test_acc = 100.0 * correct / total
    print(f"Test Acc (layer {best_layer}): {test_acc:.2f}%")

    torch.save({
        'model_state_dict': best_model_state,
        'best_layer': best_layer,
        'embedding_dim': embedding_dim,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'results_log': results_log,
        'mode': 'single_layer',
    }, save_path)
    print(f"Saved router model to {save_path}")

    return router, best_layer


# ════════════════════════════════════════════════════════════════════════
# 8. Training (Multi-Layer Concatenation – improved approach)
# ════════════════════════════════════════════════════════════════════════

def train_multi_layer(
    cached_features: dict,
    df: pd.DataFrame,
    layers: list,
    per_layer_dim: int,
    save_path: str = "router_model.pt",
):
    """Concatenate features from ALL layers into one big vector, train one probe."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device} (multi-layer concatenation)")

    labels = df['label'].values
    prompts = df['prompt'].tolist()
    class_weights = compute_class_weights(labels).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Concatenate features from all layers: [N, num_layers * per_layer_dim]
    layer_features = [cached_features[l] for l in layers if l in cached_features]
    available_layers = [l for l in layers if l in cached_features]
    concat_features = torch.cat(layer_features, dim=-1)
    total_dim = concat_features.shape[1]
    print(f"Concatenated {len(available_layers)} layers → feature dim = {total_dim}")

    total_size = len(df)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    full_dataset = RouterDataset(concat_features, labels, prompts)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=TRAIN_BATCH_SIZE)

    router = PromptRouterMLP(in_dim=total_dim).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        router.train()
        epoch_loss = 0.0
        n_batches = 0
        for feats, labs, _ in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = criterion(router(feats), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # --- Validate ---
        router.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, labs, _ in val_loader:
                feats, labs = feats.to(device), labs.to(device)
                preds = router(feats).argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += labs.size(0)

        val_acc = 100.0 * correct / total
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in router.state_dict().items()}
            patience_counter = 0
            print(" ★")
        else:
            patience_counter += 1
            print()
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n>>> Best Val Acc: {best_val_acc:.2f}% <<<")

    # Restore best and evaluate on test
    router = PromptRouterMLP(in_dim=total_dim).to(device)
    router.load_state_dict(best_model_state)
    router.eval()

    correct = total = 0
    with torch.no_grad():
        for feats, labs, _ in test_loader:
            feats, labs = feats.to(device), labs.to(device)
            preds = router(feats).argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    test_acc = 100.0 * correct / total
    print(f"Test Acc (multi-layer): {test_acc:.2f}%")

    # Per-class accuracy
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for feats, labs, _ in test_loader:
            feats = feats.to(device)
            preds = router(feats).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labs.tolist())

    name_map = {0: 'BENIGN', 1: 'GCG', 2: 'PAIR'}
    print("\nPer-class accuracy:")
    for cls_id in range(3):
        cls_mask = [i for i, l in enumerate(all_labels) if l == cls_id]
        if cls_mask:
            cls_correct = sum(1 for i in cls_mask if all_preds[i] == cls_id)
            print(f"  {name_map[cls_id]}: {100.0 * cls_correct / len(cls_mask):.2f}% ({cls_correct}/{len(cls_mask)})")

    torch.save({
        'model_state_dict': best_model_state,
        'layers_used': available_layers,
        'total_dim': total_dim,
        'per_layer_dim': per_layer_dim,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'mode': 'multi_layer',
    }, save_path)
    print(f"\nSaved router model to {save_path}")

    return router, available_layers


# ════════════════════════════════════════════════════════════════════════
# 9. Inference Demo
# ════════════════════════════════════════════════════════════════════════

def inference_demo(router, features, df):
    """Run inference on a few test samples."""
    device = next(router.parameters()).device
    labels = df['label'].values
    prompts = df['prompt'].tolist()

    total_size = len(df)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    full_dataset = RouterDataset(features, labels, prompts)
    _, _, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    demo_loader = DataLoader(test_subset, batch_size=5, shuffle=True)
    feats_batch, labels_batch, prompts_batch = next(iter(demo_loader))
    feats_batch = feats_batch.to(device)

    output = router.predict(feats_batch, tau_gcg=0.6, tau_pair=0.6)
    name_map = {0: 'BENIGN', 1: 'GCG', 2: 'PAIR'}

    print("\n--- Inference Demo (from Test Set) ---\n")
    for i in range(len(feats_batch)):
        lbl_true = labels_batch[i].item()
        lbl_pred = output.pred_label[i].item()
        act = output.action[i].item()
        probs = output.probs[i].detach().cpu().numpy()

        print(f"True: {name_map[lbl_true]} | Pred: {name_map[lbl_pred]} | Action: {name_map[act]}")
        print(f"  Probs: BENIGN={probs[0]:.2f}, GCG={probs[1]:.2f}, PAIR={probs[2]:.2f}")

        short_prompt = " ".join(prompts_batch[i].split()[:10])
        print(f"  Prompt: {short_prompt}...")
        print("-" * 30)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Router Pipeline – LLaMA-7B embeddings (improved)")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Path to labelled CSV")
    parser.add_argument("--save_path", type=str, default="router_model.pt", help="Where to save the trained router")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="HF model for embedding extraction")
    parser.add_argument("--batch_size", type=int, default=FEATURE_BATCH_SIZE, help="Batch size for feature extraction")
    parser.add_argument("--pooling", type=str, default="mean_last", choices=["last", "mean", "mean_last"],
                        help="Pooling strategy: last (original), mean, or mean_last (default)")
    parser.add_argument("--mode", type=str, default="multi_layer", choices=["single_layer", "multi_layer"],
                        help="Training mode: single_layer (original sweep) or multi_layer (concat all)")
    parser.add_argument("--skip_demo", action="store_true", help="Skip the inference demo")
    args = parser.parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────
    df = load_data(args.data_path)

    # ── 2. Load LLaMA-7B & extract hidden states ─────────────────────
    model, tokenizer = load_llama_model(args.model_name)

    prompts = df['prompt'].tolist()
    cached_features = extract_hidden_states(
        prompts, model, tokenizer,
        layers=LAYERS_TO_EXTRACT,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )

    if not cached_features:
        raise RuntimeError("Feature extraction failed – no features produced.")

    first_layer = [l for l in LAYERS_TO_EXTRACT if l in cached_features][0]
    per_layer_dim = cached_features[first_layer].shape[1]
    print(f"Per-layer feature dim: {per_layer_dim} (pooling={args.pooling})")

    # Free the large LM
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ── 3. Train router ───────────────────────────────────────────────
    if args.mode == "single_layer":
        router, best_layer = train_single_layer(
            cached_features, df,
            layers=LAYERS_TO_EXTRACT,
            embedding_dim=per_layer_dim,
            save_path=args.save_path,
        )
        demo_features = cached_features[best_layer]
    else:
        router, used_layers = train_multi_layer(
            cached_features, df,
            layers=LAYERS_TO_EXTRACT,
            per_layer_dim=per_layer_dim,
            save_path=args.save_path,
        )
        demo_features = torch.cat(
            [cached_features[l] for l in used_layers], dim=-1,
        )

    # ── 4. Demo ───────────────────────────────────────────────────────
    if not args.skip_demo:
        inference_demo(router, demo_features, df)


if __name__ == "__main__":
    main()
