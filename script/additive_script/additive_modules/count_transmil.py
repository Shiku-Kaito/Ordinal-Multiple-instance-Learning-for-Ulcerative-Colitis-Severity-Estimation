import torch
import torch.nn as nn
import numpy as np
from typeguard import typechecked
from typing import Tuple, Optional, Dict, Union, List, Sequence
import torch.nn.functional as F
from .transmil import PPEG, TransLayer
from .common import Sum


class CountTransMIL(torch.nn.Module):
    def __init__(self, args, additive_hidden_dims):
        super().__init__()
        self.pos_layer = PPEG(dim=512, has_cls=False)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.n_classes = args.classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)

        self.hidden_dims = additive_hidden_dims
        self.hidden_activation = torch.nn.ReLU()
        self.additive_function = Sum()
        self.model = self.build_model(512)
        self.cls_head = nn.Linear(512, 1)
        self.cls_token = nn.Parameter(torch.randn(1, args.classes, 512))
        self.is_cls = args.is_cls

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

    def forward(self, features):
        if self.is_cls:
            cls_tokens = self.cls_token.repeat(features.shape[0], 1, 1)
            features = torch.cat((cls_tokens, features), dim=1)

        h = features  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
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
        if self.is_cls:
            y_cls = self.cls_head(h[:, :self.n_classes, :]).squeeze()
            y_cls = F.softmax(y_cls ,dim=1) 
            patch_logits = self.model(h[:, self.n_classes:, :])
        else:
            patch_logits = self.model(h)

        y_ins = F.softmax(patch_logits / 0.1 ,dim=2)      #softmax with temperature
        y_bag = y_ins.sum(dim=1)
        y_bag /= y_bag.sum(axis=1, keepdims=True)    #正規化
        y_bag = F.softmax(y_bag / 0.1 ,dim=1)     #softmax with temperature

        results_dict = {'y_bag': y_bag}
        results_dict['y_cls'] = y_cls
        results_dict['y_ins'] = patch_logits
        return results_dict

    
class CountTransformerMILGraph(torch.nn.Module):
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

    def forward(self, images: torch.Tensor):
        batch_size, bag_size = images.shape[:2]
        shape = [-1] + list(images.shape[2:])  # merge batch and bag dim
        if self.fixed_bag_size and bag_size != self.fixed_bag_size:
            raise ValueError(
                f"Provided bag-size {bag_size} is inconsistent with expected bag-size {self.fixed_bag_size}"
            )
        images = images.view(shape)
        features = self.featurizer(images)

        features = features.view([batch_size, bag_size] + list(features.shape[1:]))  # separate batch and bag dim
        classifier_out_dict = self.classifier(features)
        bag_logits = classifier_out_dict['y_bag']
        bag_cls = classifier_out_dict['y_cls'] if 'y_cls' in classifier_out_dict else None

        patch_logits = classifier_out_dict['y_ins'] if 'y_ins' in classifier_out_dict else None
        # out = {}
        # out['value'] = bag_logits
        # if patch_logits is not None:
        #     out['patch_logits'] = patch_logits
        #  out['attention'] = attention
        patch_logits = patch_logits.reshape(-1, patch_logits.shape[-1])
        return bag_logits, patch_logits, bag_cls


