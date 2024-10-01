import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class Multi_class_Attention(nn.Module):
    def __init__(self, num_class: int = 3, pool: str = "softmax", is_ins_pred=True):
        super().__init__()

        self.pool = pool

        L = 512
        D = 128
        K = num_class

        self.num_class = num_class
        self.is_ins_pred = is_ins_pred

        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()

        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())

        self.attention_c = nn.Linear(D, K)

        self.inst_clf = nn.Sequential(nn.Linear(L, num_class))
        self.bag_clf = nn.ModuleList(
            [nn.Linear(L, 1) for i in range(K)]
        )  # use an indepdent linear layer to predict each class

    def forward(self, x, len_list, attention=False):
        feats = self.feature_extractor(x)
        bag_res = self.inst2bagclf(feats, len_list, attention)
        # y_insts = self.inst_clf(feats)

        if self.is_ins_pred:
            ins_logits = torch.empty(len(feats), self.num_class).float()
            for c, layer in enumerate(self.bag_clf):
                ins_logits[:,c] = torch.squeeze(layer(feats))
            y_insts = F.softmax(ins_logits, dim=1)

        if attention:
            return {"bag": bag_res[0], "ins": y_insts, "bag_feat": bag_res[1], "ins_feat": feats, "att_weight": bag_res[2]}
        else:
            return {"bag": bag_res[0], "ins": y_insts, "bag_feat":bag_res[1], "ins_feat": feats}

    def inst2bagclf(self, feats, len_list, attention=False):
        start_idx = 0
        bag_feats = []
        y_bags = []
        attention_weight = []
        for length in len_list:
            x = feats[start_idx : start_idx + length]
            a = self.attention_a(x)
            b = self.attention_b(x)
            A = a.mul(b)
            A = self.attention_c(A)

            A = torch.transpose(A, 1, 0)  # KxN
            if self.pool == "sigmoid":
                A = F.sigmoid(A)  # sigmoid
            else:
                A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, x)  # KxL

            logits = torch.empty(1, self.num_class).float().to(M.device)
            for c, layer in enumerate(self.bag_clf):
                logits[0, c] = layer(M[c])
            y_bags.append(logits[0])
            bag_feats.append(M.transpose(1, 0))
            attention_weight.append(A)

            start_idx += length
        if attention:
            return torch.stack(y_bags), torch.stack(bag_feats), attention_weight
        else:
            return torch.stack(y_bags), torch.stack(bag_feats)


