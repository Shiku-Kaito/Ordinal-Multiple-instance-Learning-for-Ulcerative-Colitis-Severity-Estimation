import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

class Output_max(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        self.reg_head = nn.Linear(512, 1)

    def forward(self, x, len_list):
        y = self.feature_extractor(x)
        y_ins = self.reg_head(y)
        slice_ini=0
        y_bags, y_bag_feat = [], []
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            y_bag = y_ins[slice_ini:(slice_ini+bag_len)]
            y_bag = y_bag.max(dim=0)[0]      
            y_bags.extend(y_bag)
            y_bag_feat.extend(y[slice_ini:(slice_ini+bag_len)][y_ins[slice_ini:(slice_ini+bag_len)].max(dim=0)[1]])
            slice_ini += bag_len
        y_bags = torch.stack((y_bags))
        y_bag_feat = torch.stack((y_bag_feat))
        
        ins_score = []
        slice_ini=0
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            ins_score.append(y_ins[slice_ini:(slice_ini+bag_len)].cpu().detach().numpy())
            slice_ini += bag_len
            
        return {"bag": y_bags, "ins": y_ins, "ins_feats": y, "ins_score": ins_score, "bag_feat": y_bag_feat}