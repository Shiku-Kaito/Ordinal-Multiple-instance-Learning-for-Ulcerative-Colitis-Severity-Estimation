import torch
import torch.nn as nn
import torchvision

from typeguard import typechecked
from typing import Tuple, Optional


class Resnet(torch.nn.Module):
    @typechecked
    def __init__(self, pretrained: bool = True):

        super().__init__()

        self.pretrained = True

        self.model = self.build_model()

    def build_model(self):

        model = torchvision.models.resnet18(pretrained=self.pretrained)
        model.fc = nn.Identity()
        return model

    def forward(self, images):

        features = self.model(images)
        return features

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (224, 224, 3)


class ShuffleNetV2(torch.nn.Module):
    @typechecked
    def __init__(self, pretrained: bool = True):

        super().__init__()

        self.pretrained = pretrained

        self.model = self.build_model()

    def build_model(self):

        model = torchvision.models.resnet18(pretrained=self.pretrained)
        model.fc = nn.Identity()
        return model

    def forward(self, images):

        features = self.model(images)
        return features

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (224, 224, 3)


class StableSoftmax(torch.nn.Module):
    @typechecked
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.nn.LogSoftmax(dim=self.dim)(inputs).exp()

class Sum:
    def pool(self, features, **kwargs) -> int:
        return torch.sum(features, **kwargs)
