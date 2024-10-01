import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence

class Classification_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        
        self.head1 = nn.Linear(512, 4)

    def forward(self, x):
        ins_feat = self.feature_extractor(x)

        x = self.head1(ins_feat).squeeze()
        return {"y_ins": x}


