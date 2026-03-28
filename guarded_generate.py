"""
Guarded generation wrapper for StruQ + 3-way router.

Ties together:
  1. span-aware tokenization
  2. router classification (DATA-only features)
  3. embedding-level mitigations (DATA-only)
  4. model.generate() via inputs_embeds

Usage
-----
    from guarded_generate import guarded_generate, init_router
    router = init_router()   # loads .pkl
    output, route = guarded_generate(text, model, tokenizer, router, 'SpclSpclSpcl')
"""

import logging

import torch

from config import ROUTER_CONFIG
from span_utils import tokenize_with_spans
from router import DataRouter
from mitigations import apply_benign_noise, apply_pair_noise, apply_gcg_shrink

logger = logging.getLogger(__name__)


def init_router(config=None):
    """Convenience helper — returns a DataRouter (or None if no .pkl found)."""
    config = config or ROUTER_CONFIG
    try:
        return DataRouter(config)
    except FileNotFoundError as e:
        logger.warning("Router not loaded: %s", e)
        return None


@torch.no_grad()
def guarded_generate(
    text,
    model,
    tokenizer,
    router,
    frontend_delimiters,
    config=None,
):
    """Generate a response with router-driven, DATA-only mitigations.

    Parameters
    ----------
    text : str
        Fully-formatted StruQ prompt string.
    model : transformers.PreTrainedModel
    tokenizer : transformers.PreTrainedTokenizer
    router : DataRouter | None
        If None, falls back to vanilla generation (no mitigations).
    frontend_delimiters : str
        Key into DELIMITERS, e.g. 'SpclSpclSpcl'.
    config : dict | None

    Returns
    -------
    output_text : str
    route_label : str   'BENIGN' | 'PAIR' | 'GCG' | 'NONE'
    """
    config = config or ROUTER_CONFIG
    device = next(model.parameters()).device

    # ── 1. Span-aware tokenization ────────────────────────────────────
    input_ids, span_labels = tokenize_with_spans(text, tokenizer, frontend_delimiters)

    # ── 2. Router classification ──────────────────────────────────────
    if router is not None:
        route_label, _ = router.classify(model, input_ids, span_labels)
    else:
        route_label = 'NONE'

    # HARDCODED FOR TESTING: Force Soft Removal execution
    route_label = 'GCG'

    # ── 3. Build input embeddings ─────────────────────────────────────
    ids_on_device = input_ids.unsqueeze(0).to(device)
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(ids_on_device)  # (1, seq_len, hidden_dim)

    # ── 4. Apply mitigation based on route ────────────────────────────
    if route_label == 'BENIGN':
        inputs_embeds = apply_benign_noise(
            inputs_embeds, span_labels,
            std=config['benign_noise_std'],
        )
    elif route_label == 'PAIR':
        inputs_embeds = apply_pair_noise(
            inputs_embeds, span_labels,
            std=config['pair_noise_std'],
        )
    elif route_label == 'GCG':
        # GCG path needs grad → run outside no_grad
        with torch.enable_grad():
            inputs_embeds = apply_gcg_shrink(
                inputs_embeds, span_labels, model,
                suffix_window=config['gcg_suffix_window'],
                topk=config['gcg_topk_shrink'],
                shrink_factor=config['gcg_shrink_factor'],
                tokenizer=tokenizer,
                input_ids=input_ids,
            )
    # else: route_label == 'NONE' → no mitigation

    # ── 5. Generate via inputs_embeds ─────────────────────────────────
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    gen_config = model.generation_config
    gen_config.max_new_tokens = tokenizer.model_max_length
    gen_config.do_sample = False
    gen_config.temperature = 0.0

    output_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        generation_config=gen_config,
        pad_token_id=tokenizer.pad_token_id,
    )

    # When passing `inputs_embeds` without `input_ids`, HuggingFace's model.generate 
    # returns ONLY the newly generated tokens (since it doesn't have the original input_ids to prepend).
    # Therefore, we do not need to slice off the prompt length!
    generated_ids = output_ids[0]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Strip leading spaces and EOS
    output_text = output_text.lstrip()
    eos = tokenizer.eos_token
    if eos and eos in output_text:
        output_text = output_text[:output_text.find(eos)]

    return output_text, route_label
