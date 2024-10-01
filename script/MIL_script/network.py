import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

class Feat_agg(nn.Module):
    def __init__(self, num_class, mode, p_val):
        super().__init__()
        self.MIL_f = mode
        self.MIL_p = p_val
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        self.classifier = nn.Linear(512, num_class)
        self.relu = nn.ReLU()

    def forward(self, x, len_list):
        y_features = self.feature_extractor(x)
        y_ins = self.classifier(y_features)  

        slice_ini=0
        bag_feats = []
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            y_feature = y_features[slice_ini:(slice_ini+bag_len)]
            
            if self.MIL_f == 'mean':
                bag_feature = y_feature.mean(dim=0)      
            elif self.MIL_f == 'max':
                bag_feature = torch.max(y_feature, dim=0)[0]
            elif self.MIL_f == 'p_norm':
                y_feature = self.relu(y_feature)
                bag_feature = torch.norm(y_feature, p=self.MIL_p, dim=0)#/y_feature.size(-1)
            elif self.MIL_f == 'LSE':
                y_feature = self.MIL_p * y_feature
                bag_feature = torch.logsumexp(y_feature, dim=0)#/y_feature.size(-1)
            else:
                print('None MIL_f')
            bag_feats.append(bag_feature)
            slice_ini += bag_len
        bag_feats = torch.stack((bag_feats))

        y_bag = self.classifier(bag_feats)  

        return {"bag": y_bag, "ins": y_ins}# , y_feature.reshape(-1, 512), bag_feature
    
    

class Output_agg(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        # self.MIL_p = p_val
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x, len_list):
        y = self.feature_extractor(x)
        y_ins = self.classifier(y)
        y_ins = F.softmax(y_ins, dim=1)      #softmax with temperature
        slice_ini=0
        y_bags = []
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            y_bag = y_ins[slice_ini:(slice_ini+bag_len)]

            y_bag = y_bag.mean(dim=0)      
            y_bag = F.softmax(y_bag ,dim=0)  
    
            y_bags.append(y_bag)
            slice_ini += bag_len
        y_bags = torch.stack((y_bags))
        return {"bag": y_bags, "ins": y_ins}