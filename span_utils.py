"""
Span-aware tokenization for StruQ structured queries.

Splits a formatted StruQ prompt into PROMPT / DATA / DELIMITER regions
and returns per-token span labels alongside input_ids.
"""

import torch
from config import DELIMITERS

# Span label constants
SPAN_PROMPT    = 0
SPAN_DATA      = 1
SPAN_DELIMITER = 2


def tokenize_with_spans(text, tokenizer, frontend_delimiters):
    """Tokenize a StruQ-formatted string and label each token by region.

    Parameters
    ----------
    text : str
        A fully-formatted StruQ prompt (output of PROMPT_FORMAT[...].format_map).
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer that matches the model being defended.
    frontend_delimiters : str
        Key into DELIMITERS, e.g. 'SpclSpclSpcl'.

    Returns
    -------
    input_ids : torch.LongTensor  [seq_len]
    span_labels : torch.LongTensor  [seq_len]
        Each element is SPAN_PROMPT (0), SPAN_DATA (1), or SPAN_DELIMITER (2).
    """
    delims = DELIMITERS[frontend_delimiters]
    inst_delim = delims[0]   # e.g.  [MARK] [INST][COLN]
    data_delim = delims[1]   # e.g.  [MARK] [INPT][COLN]
    resp_delim = delims[2]   # e.g.  [MARK] [RESP][COLN]

    # ── Locate delimiter positions in the raw string ──────────────────
    idx_inst = text.find(inst_delim)
    idx_data = text.find(data_delim)
    idx_resp = text.find(resp_delim)

    # Build ordered (start, end, label) segments
    segments = []

    if idx_inst == -1 or idx_resp == -1:
        # Fallback: cannot parse → treat everything as PROMPT
        segments.append((text, SPAN_PROMPT))
    elif idx_data == -1:
        # prompt_no_input  (no DATA block)
        # [pre-system PROMPT] [inst_delim] [instruction PROMPT] [resp_delim] [trailing]
        segments.append((text[:idx_inst],                          SPAN_PROMPT))    # system preamble
        segments.append((inst_delim,                               SPAN_DELIMITER))
        segments.append((text[idx_inst+len(inst_delim):idx_resp],  SPAN_PROMPT))    # instruction
        segments.append((resp_delim,                               SPAN_DELIMITER))
        segments.append((text[idx_resp+len(resp_delim):],          SPAN_PROMPT))    # trailing
    else:
        # prompt_input  (has DATA block)
        # [pre-system PROMPT] [inst_delim] [instruction PROMPT] [data_delim] [DATA] [resp_delim] [trailing]
        segments.append((text[:idx_inst],                          SPAN_PROMPT))    # system preamble
        segments.append((inst_delim,                               SPAN_DELIMITER))
        segments.append((text[idx_inst+len(inst_delim):idx_data],  SPAN_PROMPT))    # instruction
        segments.append((data_delim,                               SPAN_DELIMITER))
        segments.append((text[idx_data+len(data_delim):idx_resp],  SPAN_DATA))     # input / DATA
        segments.append((resp_delim,                               SPAN_DELIMITER))
        segments.append((text[idx_resp+len(resp_delim):],          SPAN_PROMPT))    # trailing

    # ── Tokenize each segment and record labels ───────────────────────
    all_ids = []
    all_labels = []

    for seg_text, label in segments:
        if not seg_text:
            continue
        toks = tokenizer.encode(seg_text, add_special_tokens=False)
        all_ids.extend(toks)
        all_labels.extend([label] * len(toks))

    input_ids   = torch.tensor(all_ids,   dtype=torch.long)
    span_labels = torch.tensor(all_labels, dtype=torch.long)
    return input_ids, span_labels


def get_data_mask(span_labels):
    """Boolean mask selecting DATA tokens."""
    return span_labels == SPAN_DATA


def get_prompt_mask(span_labels):
    """Boolean mask selecting PROMPT tokens."""
    return span_labels == SPAN_PROMPT


def get_delimiter_mask(span_labels):
    """Boolean mask selecting DELIMITER tokens."""
    return span_labels == SPAN_DELIMITER
