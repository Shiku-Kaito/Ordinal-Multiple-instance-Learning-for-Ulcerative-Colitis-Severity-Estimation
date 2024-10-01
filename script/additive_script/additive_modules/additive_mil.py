import torch
import torch.nn as nn
import numpy as np

from typeguard import typechecked
from typing import Tuple, Optional, Dict, Union, List, Sequence
from .common import Sum, StableSoftmax


class AdditiveClassifier(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
    ):

        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.additive_function = Sum()
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = torch.nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dims):
                layers.append(self.hidden_activation)
        model = torch.nn.Sequential(*layers)
        return model

    def forward(self, features, attention):
        attended_features = attention * features
        patch_logits = self.model(attended_features)
        logits = self.additive_function.pool(patch_logits, dim=1, keepdim=False)
        # patch_logits = patch_logits.reshape(-1,10).shape
        classifier_out_dict = {}
        classifier_out_dict['logits'] = logits
        classifier_out_dict['patch_logits'] = patch_logits
        return classifier_out_dict
