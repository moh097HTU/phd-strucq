"""
Embedding-level mitigations applied **only** to DATA tokens.

Three strategies keyed to the router's classification:
  BENIGN → small Gaussian noise on DATA embeddings
  PAIR   → large Gaussian noise on DATA embeddings
  GCG    → saliency-based shrink on the most salient DATA suffix tokens

PROMPT and DELIMITER tokens are **never** modified.
"""

import torch
import torch.nn.functional as F

from span_utils import SPAN_DATA


# ══════════════════════════════════════════════════════════════════════
# Noise-based mitigations
# ══════════════════════════════════════════════════════════════════════

def apply_benign_noise(inputs_embeds, span_labels, std=0.01):
    """Add small Gaussian noise to DATA embeddings only.

    Parameters
    ----------
    inputs_embeds : torch.FloatTensor  (1, seq_len, hidden_dim)
    span_labels : torch.LongTensor  (seq_len,)
    std : float

    Returns
    -------
    inputs_embeds : torch.FloatTensor  (1, seq_len, hidden_dim)  — modified in-place
    """
    data_mask = (span_labels == SPAN_DATA).to(inputs_embeds.device)  # (seq_len,)
    if data_mask.sum() == 0 or std <= 0:
        return inputs_embeds

    noise = torch.randn_like(inputs_embeds[0]) * std  # (seq_len, hidden_dim)
    noise[~data_mask] = 0.0                            # zero-out non-DATA
    inputs_embeds = inputs_embeds + noise.unsqueeze(0)
    return inputs_embeds


def apply_pair_noise(inputs_embeds, span_labels, std=0.30):
    """Add large Gaussian noise to DATA embeddings only.

    Same logic as benign_noise but with a larger σ.
    """
    return apply_benign_noise(inputs_embeds, span_labels, std=std)


# ══════════════════════════════════════════════════════════════════════
# Saliency-based shrink (for GCG-classified inputs)
# ══════════════════════════════════════════════════════════════════════

def apply_gcg_shrink(
    inputs_embeds,
    span_labels,
    model,
    suffix_window=128,
    topk=10,
    shrink_factor=0.01,
    target_text="Sure, here is",
    tokenizer=None,
    input_ids=None,
):
    """Shrink the most salient DATA-suffix embeddings to neutralise GCG suffixes.

    1. Forward pass with grad tracking on inputs_embeds
    2. Compute affirmation loss (cross-entropy toward target_text)
    3. Backprop → per-token saliency = ||grad||
    4. Restrict to the last `suffix_window` DATA tokens
    5. Shrink top-k most salient embeddings by `shrink_factor`

    Parameters
    ----------
    inputs_embeds : torch.FloatTensor  (1, seq_len, hidden_dim)
    span_labels : torch.LongTensor  (seq_len,)
    model : transformers.PreTrainedModel
    suffix_window : int
    topk : int
    shrink_factor : float
    target_text : str
    tokenizer : transformers.PreTrainedTokenizer

    Returns
    -------
    inputs_embeds : torch.FloatTensor  (1, seq_len, hidden_dim)  — modified
    """
    device = inputs_embeds.device
    data_mask = (span_labels == SPAN_DATA).to(device)  # (seq_len,)
    data_indices = data_mask.nonzero(as_tuple=True)[0]

    if len(data_indices) == 0:
        return inputs_embeds

    # Restrict to the suffix window of DATA tokens
    if len(data_indices) > suffix_window:
        candidate_indices = data_indices[-suffix_window:]
    else:
        candidate_indices = data_indices

    # Make a copy that requires grad for the backward pass
    embeds = inputs_embeds.clone().detach().requires_grad_(True)

    # ── Forward pass ──────────────────────────────────────────────────
    outputs = model(inputs_embeds=embeds)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # ── Compute affirmation loss ──────────────────────────────────────
    if tokenizer is not None and target_text:
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
        # Use the last `len(target_ids)` positions of the logits
        n_target = len(target_ids)
        pred_logits = logits[0, -n_target:]   # (n_target, vocab_size)
        loss = F.cross_entropy(pred_logits, target_ids)
    else:
        # Fallback: maximise probability of the single most-likely next token
        loss = -logits[0, -1].max()

    # ── Backward → saliency ───────────────────────────────────────────
    loss.backward()
    grad = embeds.grad[0]  # (seq_len, hidden_dim)

    # Saliency score = L2 norm of gradient per token
    saliency = grad.norm(dim=-1)  # (seq_len,)

    # Restrict to candidate DATA suffix tokens
    candidate_saliency = saliency[candidate_indices]

    # Top-k most salient
    k = min(topk, len(candidate_saliency))
    _, topk_local = candidate_saliency.topk(k)
    topk_global = candidate_indices[topk_local]

    # ── Shrink those embeddings ───────────────────────────────────────
    import os
    hard_remove = os.environ.get('HARD_REMOVE', 'false').lower() == 'true'

    if hard_remove:
        keep_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=device)
        keep_mask[topk_global] = False
        modified = inputs_embeds[:, keep_mask, :]

        if input_ids is not None and tokenizer is not None:
            topk_global_cpu = topk_global.cpu()
            removed_ids = input_ids[topk_global_cpu].tolist()
            removed_tokens = tokenizer.convert_ids_to_tokens(removed_ids)
            print(f"\\n[Router Defense] Hard removal applied! Removed highly salient tokens: {removed_tokens}")
    else:
        modified = inputs_embeds.clone()
        modified[0, topk_global] *= shrink_factor

        if input_ids is not None and tokenizer is not None:
            topk_global_cpu = topk_global.cpu()
            shrunk_ids = input_ids[topk_global_cpu].tolist()
            shrunk_tokens = tokenizer.convert_ids_to_tokens(shrunk_ids)
            print(f"\\n[Router Defense] Soft removal applied! Shrunk highly salient tokens: {shrunk_tokens}")

    return modified
