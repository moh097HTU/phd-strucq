"""
3-Way Router: classifies DATA portion of a StruQ query as BENIGN / PAIR / GCG.

Loads a pre-trained PyTorch MLP classifier from a .pt file and
extracts features from the model's hidden states over DATA tokens only,
using the same multi-layer mean+last pooling strategy used in training.
"""

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ROUTER_CONFIG
from span_utils import SPAN_DATA
from router_pipeline import PromptRouterMLP, route_actions

logger = logging.getLogger(__name__)


class DataRouter:
    """Classifies DATA content as BENIGN / PAIR / GCG using hidden-state features."""

    def __init__(self, config=None):
        config = config or ROUTER_CONFIG
        model_path = config.get('model_path', 'router_model.pt')
        
        # Change default .pkl to .pt if using the new pipeline
        if model_path.endswith('.pkl'):
            model_path = model_path.replace('.pkl', '.pt')
            
        if not os.path.isabs(model_path):
            # resolve relative to the StruQ project root
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Router model not found at {model_path}. "
                "Train your router and place the .pt file there."
            )

        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Setup configs based on what was trained
        self.layers_used = checkpoint.get('layers_used', [24, 31])
        self.total_dim = float(checkpoint.get('total_dim', 16384)) # cast to float just for printing, keep int for model
        self.mode = checkpoint.get('mode', 'multi_layer')
        
        self.clf = PromptRouterMLP(in_dim=int(self.total_dim))
        self.clf.load_state_dict(checkpoint['model_state_dict'])
        self.clf.eval()
        
        self.labels = config['labels']
        logger.info("DataRouter loaded from %s (Layers: %s)", model_path, self.layers_used)

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_data_features(self, model, input_ids, span_labels):
        """Forward-pass and apply mean+last pooling over DATA tokens for specified layers.

        Parameters
        ----------
        model : transformers.PreTrainedModel
        input_ids : torch.LongTensor  [seq_len]
        span_labels : torch.LongTensor  [seq_len]

        Returns
        -------
        features : torch.Tensor  [1, total_dim]
        """
        device = next(model.parameters()).device
        self.clf.to(device)
        
        ids = input_ids.unsqueeze(0).to(device)

        outputs = model(
            input_ids=ids,
            output_hidden_states=True,
            use_cache=False,
        )
        
        data_mask = (span_labels == SPAN_DATA).to(device)    # (seq_len,)
        if data_mask.sum() == 0:
            # no DATA tokens (prompt_no_input) → return zeros
            return torch.zeros((1, int(self.total_dim)), device=device, dtype=torch.float32)

        # Get the index of the last DATA token
        # Using nonzero to find all indices where mask is True, then taking the max
        data_indices = data_mask.nonzero(as_tuple=True)[0]
        last_data_token_idx = data_indices.max()

        all_layer_features = []
        
        for l in self.layers_used:
            if l < len(outputs.hidden_states):
                # Convert to float32 to prevent float16 overflow during sum
                layer_hidden_fp32 = outputs.hidden_states[l].float() # (1, seq_len, hidden_dim)
                
                # 1. Last token embedding (of the DATA region)
                last_embeds = layer_hidden_fp32[:, last_data_token_idx, :] # (1, hidden_dim)
                
                # 2. Mean pool embedding (over the DATA region)
                mask_expanded = data_mask.unsqueeze(0).unsqueeze(-1).to(torch.float32) # (1, seq_len, 1)
                sum_hidden = (layer_hidden_fp32 * mask_expanded).sum(dim=1) # (1, hidden_dim)
                counts = data_mask.sum().to(torch.float32)
                mean_embeds = sum_hidden / counts.clamp(min=1)
                
                # Concatenate [mean, last] -> matches 'mean_last' pooling in training
                embeds = torch.cat([mean_embeds, last_embeds], dim=-1) # (1, 2*hidden_dim)
                all_layer_features.append(embeds)
                
        # Concatenate all featured layers
        concat_features = torch.cat(all_layer_features, dim=-1) # (1, num_layers * 2 * hidden_dim)
        return concat_features

    # ──────────────────────────────────────────────────────────────────
    def classify(self, model, input_ids, span_labels):
        """Classify the DATA region of a structured query.

        Returns
        -------
        label : str   One of 'BENIGN', 'PAIR', 'GCG'.
        pred_id : int  Raw integer prediction from the router.
        """
        feats = self.extract_data_features(model, input_ids, span_labels)
        
        # feats shape [1, total_dim]
        # Run MLP forward pass
        output = self.clf.predict(feats, tau_gcg=0.60, tau_pair=0.60)
        
        pred_id = output.pred_label[0].item()
        action_id = output.action[0].item()
        
        label = self.labels.get(action_id, 'BENIGN')
        return label, pred_id
