import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

class Krank_MIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        self.reg_head1 = nn.Linear(512, 1)
        self.reg_head2 = nn.Linear(512, 1)
        self.reg_head3 = nn.Linear(512, 1)
        

    def forward(self, x, len_list):
        y = self.feature_extractor(x)
        y1_ins = self.reg_head1(y)
        y2_ins = self.reg_head2(y)
        y3_ins = self.reg_head3(y)
        
        slice_ini=0
        y1_bags, y2_bags, y3_bags = [], [], []
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            y1_bag, y2_bag, y3_bag = y1_ins[slice_ini:(slice_ini+bag_len)], y2_ins[slice_ini:(slice_ini+bag_len)], y3_ins[slice_ini:(slice_ini+bag_len)]
            y1_bag, y2_bag, y3_bag = y1_bag.max(dim=0)[0], y2_bag.max(dim=0)[0], y3_bag.max(dim=0)[0]    
            y1_bags.extend(y1_bag), y2_bags.extend(y2_bag), y3_bags.extend(y3_bag)
            slice_ini += bag_len
        y1_bags, y2_bags, y3_bags = torch.stack((y1_bags)), torch.stack((y2_bags)), torch.stack((y3_bags))
        
        return {"y_0vs123": y1_bags, "y_01vs23": y2_bags, "y_012vs3": y3_bags, "y_ins_0vs123": y1_ins, "y_ins_01vs23": y2_ins, "y_ins_012vs3": y3_ins}
