import torch
import torch.nn as nn

from typeguard import typechecked
from typing import Tuple, Optional, Sequence
from .common import Sum, StableSoftmax


class DefaultAttentionModule(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        input_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = nn.ReLU(),
        output_activation: torch.nn.Module = StableSoftmax(dim=1),
        use_batch_norm: bool = True,
        track_bn_stats: bool = True,
    ):

        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats

        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [1]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dims):
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(nodes_out, track_running_stats=self.track_bn_stats))
                layers.append(self.hidden_activation)
        model = nn.Sequential(*layers)
        return model

    def forward(self, features, bag_size):
        out = self.model(features)
        out = out.view([-1, bag_size])
        attention = self.output_activation(out)
        return attention.unsqueeze(-1)


class DefaultClassifier(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        pooling_mode = Sum(),
    ):

        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.pooling_mode = pooling_mode
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
        features = self.pooling_mode.pool(attention * features, dim=1, keepdim=False)
        logits = self.model(features)
        classifier_out_dict = {}
        classifier_out_dict['logits'] = logits
        return classifier_out_dict


class DefaultMILGraph(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        featurizer: torch.nn.Module,
        classifier: torch.nn.Module,
        pointer: torch.nn.Module,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.pointer = pointer

    def forward(self, images: torch.Tensor):
        batch_size, bag_size = images.shape[:2]
        shape = [-1] + list(images.shape[2:])  # merge batch and bag dim
        images = images.view(shape)
        features = self.featurizer(images)
        attention = self.pointer(features, bag_size)
        if not torch.all(attention >= 0):
            raise ValueError("{}: Attention weights cannot be negative".format(attention))

        features = features.view([batch_size, bag_size] + list(features.shape[1:]))  # separate batch and bag dim
        classifier_out_dict = self.classifier(features, attention)
        bag_logits = classifier_out_dict['logits']

        patch_logits = classifier_out_dict['patch_logits'] if 'patch_logits' in classifier_out_dict else None
        # out = {}
        # out['value'] = bag_logits
        # if patch_logits is not None:
        #     out['patch_logits'] = patch_logits
        # out['attention'] = attention

        # return out
        patch_logits = patch_logits.reshape(-1, patch_logits.shape[-1])
        return {"bag": bag_logits, "ins": patch_logits}
