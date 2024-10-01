import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence

class K_rank_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        
        self.head1 = nn.Linear(512, 2)
        self.head2 = nn.Linear(512, 2)
        self.head3 = nn.Linear(512, 2)

    def forward(self, x):
        ins_feat = self.feature_extractor(x)

        x1 = self.head1(ins_feat).squeeze()
        x2 = self.head2(ins_feat).squeeze()
        x3 = self.head3(ins_feat).squeeze()

        return {"y_0vs123": x1, "y_01vs23":x2, "y_012vs3":x3}


