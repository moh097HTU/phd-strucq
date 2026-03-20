"""
Router Pipeline  –  Converted from router_pipeline.ipynb
========================================================
Trains a 3-class MLP probe (BENIGN / GCG / PAIR) on hidden-state
features extracted from **LLaMA-7B** (huggyllama/llama-7b).

Usage:
    python router_pipeline.py                       # default data.csv
    python router_pipeline.py --data_path data.csv  # explicit path
    python router_pipeline.py --save_path router_model.pt
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

# --- LLaMA-7B (32 transformer layers → hidden_states has 33 entries: embed + 32 layers) ---
MODEL_NAME = "huggyllama/llama-7b"
USE_CHAT_TEMPLATE = False  # Base LLaMA-7B has no chat template
LAYERS_TO_EXTRACT = [8, 16, 24, 30]  # Spread across 32 layers
FEATURE_BATCH_SIZE = 4  # LLaMA-7B is large; keep batch small to avoid OOM

TRAIN_BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


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

    Routing logic:
      if P(GCG)  > tau_gcg  -> GCG action
      elif P(PAIR) > tau_pair -> PAIR action
      else -> BENIGN action

    probs: [B, 3] with columns [BENIGN, GCG, PAIR] in RouterLabel order.
    Returns:
      actions: [B] int tensor in RouterLabel id space.
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
    3-class router that takes *vector features* per prompt.
    x_features shape: [B, D]
    Output classes: BENIGN=0, GCG=1, PAIR=2
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
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
        return 0  # default to benign

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
) -> dict:
    """
    Extract last-token hidden states from specified layers.

    Returns dict mapping layer_idx -> tensor of shape [N, hidden_dim].
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
            inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)
        else:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # Index of the last real (non-padding) token per sequence
        last_token_idxs = inputs.attention_mask.sum(dim=1) - 1

        for l in layers:
            if l < len(outputs.hidden_states):
                layer_hidden = outputs.hidden_states[l]
                batch_indices = torch.arange(layer_hidden.size(0), device=layer_hidden.device)
                final_embeds = layer_hidden[batch_indices, last_token_idxs, :]
                all_features[l].append(final_embeds.cpu())

    results = {}
    for l in layers:
        if all_features[l]:
            results[l] = torch.cat(all_features[l], dim=0).float()
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
# 6. Training Loop (Layer Sweep)
# ════════════════════════════════════════════════════════════════════════

def train_router(
    cached_features: dict,
    df: pd.DataFrame,
    layers: list,
    embedding_dim: int,
    save_path: str = "router_model.pt",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device}")

    labels = df['label'].values
    prompts = df['prompt'].tolist()

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

        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(SPLIT_SEED),
        )

        train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=TRAIN_BATCH_SIZE)

        # Init model
        router = PromptRouterMLP(in_dim=embedding_dim).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        layer_best_val = 0.0

        for epoch in range(EPOCHS):
            # --- Train ---
            router.train()
            for feats, labs, _ in train_loader:
                feats, labs = feats.to(device), labs.to(device)
                optimizer.zero_grad()
                loss = criterion(router(feats), labs)
                loss.backward()
                optimizer.step()

            # --- Validate ---
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

        print(f"Layer {layer_idx} Best Val Acc: {layer_best_val:.2f}%")
        results_log.append((layer_idx, layer_best_val))

        if layer_best_val > best_val_acc:
            best_val_acc = layer_best_val
            best_layer = layer_idx
            best_model_state = router.state_dict()

    print(f"\n>>> Best Layer: {best_layer} with Val Acc: {best_val_acc:.2f}% <<<")

    # ── Restore best model & evaluate on test set ─────────────────────
    router = PromptRouterMLP(in_dim=embedding_dim).to(device)
    router.load_state_dict(best_model_state)
    router.eval()

    features_tensor = cached_features[best_layer]
    full_dataset = RouterDataset(features_tensor, labels, prompts)
    _, _, test_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    test_loader = DataLoader(test_subset, batch_size=TRAIN_BATCH_SIZE)

    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labs, _ in test_loader:
            feats, labs = feats.to(device), labs.to(device)
            preds = router(feats).argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    test_acc = 100.0 * correct / total
    print(f"Test Acc (layer {best_layer}): {test_acc:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────
    torch.save({
        'model_state_dict': best_model_state,
        'best_layer': best_layer,
        'embedding_dim': embedding_dim,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'results_log': results_log,
    }, save_path)
    print(f"Saved router model to {save_path}")

    return router, best_layer


# ════════════════════════════════════════════════════════════════════════
# 7. Inference Demo
# ════════════════════════════════════════════════════════════════════════

def inference_demo(router, cached_features, best_layer, df):
    device = next(router.parameters()).device
    labels = df['label'].values
    prompts = df['prompt'].tolist()

    total_size = len(df)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    features_tensor = cached_features[best_layer]
    full_dataset = RouterDataset(features_tensor, labels, prompts)
    _, _, test_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    demo_loader = DataLoader(test_subset, batch_size=5, shuffle=True)
    features, labels_batch, prompts_batch = next(iter(demo_loader))
    features = features.to(device)

    output = router.predict(features, tau_gcg=0.6, tau_pair=0.6)
    name_map = {0: 'BENIGN', 1: 'GCG', 2: 'PAIR'}

    print("\n--- Inference Demo (from Test Set) ---\n")
    for i in range(len(features)):
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
    parser = argparse.ArgumentParser(description="Router Pipeline – LLaMA-7B embeddings")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Path to labelled CSV")
    parser.add_argument("--save_path", type=str, default="router_model.pt", help="Where to save the trained router")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="HF model for embedding extraction")
    parser.add_argument("--batch_size", type=int, default=FEATURE_BATCH_SIZE, help="Batch size for feature extraction")
    parser.add_argument("--skip_demo", action="store_true", help="Skip the inference demo at the end")
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
    )

    if not cached_features or LAYERS_TO_EXTRACT[0] not in cached_features:
        raise RuntimeError("Feature extraction failed – no features produced.")

    embedding_dim = cached_features[LAYERS_TO_EXTRACT[0]].shape[1]
    print(f"Hidden Dim: {embedding_dim}")

    # Free the large LM from GPU
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ── 3. Train router (sweeps across layers) ────────────────────────
    router, best_layer = train_router(
        cached_features, df,
        layers=LAYERS_TO_EXTRACT,
        embedding_dim=embedding_dim,
        save_path=args.save_path,
    )

    # ── 4. Demo ───────────────────────────────────────────────────────
    if not args.skip_demo:
        inference_demo(router, cached_features, best_layer, df)


if __name__ == "__main__":
    main()
