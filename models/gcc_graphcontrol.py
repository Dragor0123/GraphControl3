from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from utils.register import register
import copy
from .gcc import GCC, AntiSmoothingGINConv

@register.model_register
class GCC_GraphControl(nn.Module):

    def __init__(
        self,
        **kwargs
    ):
        super(GCC_GraphControl, self).__init__()
        operator_mode = kwargs.pop('operator_mode', 'standard')
        input_dim = kwargs['positional_embedding_size']
        hidden_size = kwargs['node_hidden_dim']
        output_dim = kwargs['num_classes']

        self.encoder = GCC(**kwargs)
        self.trainable_copy = copy.deepcopy(self.encoder)

        if operator_mode == 'anti_smoothing':
            self._apply_anti_smoothing(self.trainable_copy)
        
        self.zero_conv1 = torch.nn.Linear(input_dim, input_dim)     
        self.zero_conv2 = torch.nn.Linear(hidden_size, hidden_size)

        self.linear_classifier = torch.nn.Linear(hidden_size, output_dim)

        with torch.no_grad():
            self.zero_conv1.weight = torch.nn.Parameter(torch.zeros(input_dim, input_dim))
            self.zero_conv1.bias = torch.nn.Parameter(torch.zeros(input_dim))
            self.zero_conv2.weight = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.zero_conv2.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        
        self.prompt = torch.nn.Parameter(torch.normal(mean=0, std=0.01, size=(1, input_dim)))
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        raise NotImplementedError('Please use --subsampling')
    
    @staticmethod
    def _apply_anti_smoothing(gcc_model):
        """Replace all GINConv layers in UnsupervisedGIN with AntiSmoothingGINConv.
        The MLP weights are preserved; only the aggregation sign is flipped."""
        for i in range(len(gcc_model.gnn.ginlayers)):
            gcc_model.gnn.ginlayers[i].__class__ = AntiSmoothingGINConv

    def reset_classifier(self):
        self.linear_classifier.reset_parameters()

    def forward_with_diagnostics(
        self,
        x,
        x_sim,
        edge_index,
        batch,
        root_n_id,
        edge_weight=None,
        frozen=False,
        **kwargs
    ):
        if not frozen:
            raise NotImplementedError('Please freeze pre-trained models')

        with torch.no_grad():
            self.encoder.eval()
            h_f = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

        x_down = self.zero_conv1(x_sim)
        x_down = x_down + x

        h_tc_raw = self.trainable_copy.forward_subgraph(
            x_down, edge_index, batch, root_n_id)
        h_c = self.zero_conv2(h_tc_raw)
        h_fc = h_f + h_c

        z_f = self.linear_classifier(h_f).detach()
        z_c = self.linear_classifier(h_c)
        z_fc = self.linear_classifier(h_fc)

        return {
            'h_f': h_f,
            'h_tc_raw': h_tc_raw,
            'h_c': h_c,
            'h_fc': h_fc,
            'z_f': z_f,
            'z_c': z_c,
            'z_fc': z_fc,
        }
    
    def forward_subgraph(self, x, x_sim, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        diagnostics = self.forward_with_diagnostics(
            x,
            x_sim,
            edge_index,
            batch,
            root_n_id,
            edge_weight=edge_weight,
            frozen=frozen,
            **kwargs,
        )
        return diagnostics['z_fc']
