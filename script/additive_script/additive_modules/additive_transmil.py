import torch
import torch.nn as nn
import numpy as np
from typeguard import typechecked
from typing import Tuple, Optional, Dict, Union, List, Sequence

from .transmil import PPEG, TransLayer
from .common import Sum
from torch.nn.utils.rnn import pad_sequence


class AdditiveTransMIL(torch.nn.Module):
    def __init__(self, n_classes, additive_hidden_dims):
        super().__init__()
        self.pos_layer = PPEG(dim=512, has_cls=False)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)

        self.hidden_dims = additive_hidden_dims
        self.hidden_activation = torch.nn.ReLU()
        self.additive_function = Sum()
        self.model = self.build_model(512)

    def build_model(self, input_dims):
        nodes_by_layer = [input_dims] + list(self.hidden_dims) + [self.n_classes]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = torch.nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dims):
                layers.append(self.hidden_activation)
        model = torch.nn.Sequential(*layers)
        return model

    def forward(self, features, len_list):
        h = features  # [B, n, 1024]

        ini_idx = 0
        ins_feats = []
        for length in len_list:
            ins_feats.append(features[ini_idx : ini_idx + length])
            ini_idx += length
        
        padded_ins_feats = pad_sequence(ins_feats, batch_first=True, padding_value=0)

        padded_ins_feats = padded_ins_feats.float()

        padded_ins_feats = padded_ins_feats.float()
        h = self._fc1(padded_ins_feats)  # [B, n, 512]

        # ---->pad
        # H = h.shape[1]
        # _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # add_length = _H * _W - H
        # h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        # h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)

        # ---->predict
        patch_logits = self.model(h)
        logits = self.additive_function.pool(patch_logits, dim=1, keepdim=False)
        results_dict = {'logits': logits}
        results_dict['patch_logits'] = patch_logits
        return results_dict

    
class TransformerMILGraph(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        featurizer: torch.nn.Module,
        classifier: torch.nn.Module,
        fixed_bag_size: Optional[int] = None,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.fixed_bag_size = fixed_bag_size
        self.output_dims = self.classifier.n_classes

    def forward(self, images: torch.Tensor, len_list):
        # batch_size, bag_size = images.shape[:2]
        # shape = [-1] + list(images.shape[2:])  # merge batch and bag dim
        # if self.fixed_bag_size and bag_size != self.fixed_bag_size:
        #     raise ValueError(
        #         f"Provided bag-size {bag_size} is inconsistent with expected bag-size {self.fixed_bag_size}"
        #     )
        # images = images.view(shape)
        features = self.featurizer(images)

        # features = features.view([batch_size, bag_size] + list(features.shape[1:]))  # separate batch and bag dim
        classifier_out_dict = self.classifier(features, len_list)
        bag_logits = classifier_out_dict['logits']

        # patch_logits = classifier_out_dict['patch_logits'] if 'patch_logits' in classifier_out_dict else None
        # # out = {}
        # # out['value'] = bag_logits
        # # if patch_logits is not None:
        # #     out['patch_logits'] = patch_logits
        # #  out['attention'] = attention
        # patch_logits = patch_logits.reshape(-1, patch_logits.shape[-1])
        return {"bag": bag_logits, "ins": []}


