import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18

class IClassifier(nn.Module):
    def __init__(self, feature_size):
        super(IClassifier, self).__init__()
        self.feature_extractor = resnet18(pretrained=True)      
        self.feature_extractor.fc = nn.Sequential()
        self.fc = nn.Linear(feature_size, 1)
        
    def forward(self, x):
        # device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats) # N x C
        return feats, c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        # ### 1D convolutional layer that can handle multiple class (including binary)
        # self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        self.fcc = nn.Linear(512, 4)
        
    def forward(self, feats, c, len_list): # N x K, N x C
        device = feats.device
        slice_ini=0
        bag_feats, max_c = [], []
        for idx, bag_len in enumerate(len_list):            # loop at all 
            feat = feats[slice_ini:(slice_ini+bag_len)]
            one_c = c[slice_ini:(slice_ini+bag_len)]
            V = self.v(feat) # N x V, unsorted
            Q = self.q(feat).view(feat.shape[0], -1) # N x Q, unsorted
            # handle multiple classes without for loop
            _, m_indices = torch.sort(one_c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
            m_feats = torch.index_select(feat, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
            max_c.append(torch.index_select(one_c, dim=0, index=m_indices[0, :]) )
            q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
            A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
            A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V        
            bag_feats.append(B)
            slice_ini += bag_len
            
        bag_feats, max_c = torch.stack((bag_feats)), torch.stack((max_c))
        bag_feats = bag_feats.squeeze()
        C = self.fcc(bag_feats) # 1 x C x 
        return C, max_c, A, bag_feats
    
class DSMIL(nn.Module):
    def __init__(self, classes=4):
        super(DSMIL, self).__init__()
        self.i_classifier = IClassifier(feature_size=512)
        self.b_classifier = BClassifier(input_size=512, output_class=classes)
        
    def forward(self, x, len_list):
        feats, classes = self.i_classifier(x)
        prediction_bag, max_c, A, B = self.b_classifier(feats, classes, len_list)
        
        return {"IC":max_c, "BC":prediction_bag}#, A, B