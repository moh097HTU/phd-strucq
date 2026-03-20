"""
3-Way Router: classifies DATA portion of a StruQ query as BENIGN / PAIR / GCG.

Loads a pre-trained sklearn-compatible classifier from a .pkl file and
extracts features from the model's hidden states over DATA tokens only.
"""

import os
import pickle
import logging

import numpy as np
import torch

from config import ROUTER_CONFIG
from span_utils import SPAN_DATA

logger = logging.getLogger(__name__)


class DataRouter:
    """Classifies DATA content as BENIGN / PAIR / GCG using hidden-state features."""

    def __init__(self, config=None):
        config = config or ROUTER_CONFIG
        model_path = config['model_path']
        if not os.path.isabs(model_path):
            # resolve relative to the StruQ project root
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Router model not found at {model_path}. "
                "Train your router and place the .pkl file there."
            )

        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        self.labels = config['labels']
        self.feature_layer = config['feature_layer']
        logger.info("DataRouter loaded from %s", model_path)

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_data_features(self, model, input_ids, span_labels):
        """Forward-pass and mean-pool hidden states over DATA tokens.

        Parameters
        ----------
        model : transformers.PreTrainedModel
        input_ids : torch.LongTensor  [seq_len]
        span_labels : torch.LongTensor  [seq_len]

        Returns
        -------
        features : np.ndarray  (hidden_dim,)
        """
        device = next(model.parameters()).device
        ids = input_ids.unsqueeze(0).to(device)

        outputs = model(
            input_ids=ids,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self.feature_layer]  # (1, seq_len, hidden_dim)

        data_mask = (span_labels == SPAN_DATA).to(device)    # (seq_len,)
        if data_mask.sum() == 0:
            # no DATA tokens (prompt_no_input) → return zeros
            hidden_dim = hidden.shape[-1]
            return np.zeros(hidden_dim, dtype=np.float32)

        pooled = hidden[0, data_mask].mean(dim=0)            # (hidden_dim,)
        return pooled.float().cpu().numpy()

    # ──────────────────────────────────────────────────────────────────
    def classify(self, model, input_ids, span_labels):
        """Classify the DATA region of a structured query.

        Returns
        -------
        label : str   One of 'BENIGN', 'PAIR', 'GCG'.
        pred_id : int  Raw integer prediction from the router.
        """
        feats = self.extract_data_features(model, input_ids, span_labels)
        pred = int(self.clf.predict(feats.reshape(1, -1))[0])
        label = self.labels.get(pred, 'BENIGN')  # default to BENIGN
        return label, pred
