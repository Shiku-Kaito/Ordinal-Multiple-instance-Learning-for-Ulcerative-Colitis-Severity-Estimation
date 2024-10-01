import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from statistics import mean
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from skimage import io
import glob
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler
import random
import copy
from utils import *

#toy
class all_one_vs_rest_LIMUC_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, bags, ins_label, bag_labels):
        np.random.seed(args.seed)
        self.bags_paths = bags
        self.ins_label = ins_label
        self.bag_labels = bag_labels
        self.len = len(self.bags_paths)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # bag_paths = np.load(self.bags_paths[idx])   # bagの内のinstanceのパスが格納されているファイルを読みコム
        bag_paths = self.bags_paths[idx]  # bagの内のinstanceのパスが格納されているファイルを読みコム

        bag = []
        for bag_path in bag_paths:
            img = Image.open(bag_path)
            img = np.array(img.resize((224, 224)))
            img = img / 255
            bag.append(img)
        bag = np.array(bag)
        bag = bag.transpose((0, 3, 1, 2))
        bag = torch.from_numpy(bag.astype(np.float32))

        max_label = self.bag_labels[idx]
        # 0 vs 1,2,3
        if max_label in set([0]):
            ovs123_bag_label = 0
        elif max_label in set([1,2,3]):
            ovs123_bag_label = 1
        # 0,1 vs 2,3
        if max_label in set([0,1]):
            o1vs23_bag_label = 0
        elif max_label in set([2,3]):
            o1vs23_bag_label = 1
        # 0,1,2 vs 3
        if max_label in set([0,1,2]):
            o12vs3_bag_label = 0
        elif max_label in set([3]):
            o12vs3_bag_label = 1
            
        ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = torch.tensor(int(ovs123_bag_label), dtype=torch.long), torch.tensor(int(o1vs23_bag_label), dtype=torch.long), torch.tensor(int(o12vs3_bag_label), dtype=torch.long)
        max_label = torch.tensor(int(max_label), dtype=torch.long)

        # ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = self.ins_label[idx], self.ins_label[idx], self.ins_label[idx], self.ins_label[idx]

        ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = self.ins_label[idx], [], [], []
        for ins_label in ins_labels:
            if ins_label in set([0]):
                ovs123_ins_label.append(0)
            elif ins_label in set([1,2,3]):
                ovs123_ins_label.append(1)
            # 0,1 vs 2,3
            if ins_label in set([0,1]):
                o1vs23_ins_label.append(0)
            elif ins_label in set([2,3]):
                o1vs23_ins_label.append(1)
            # 0,1,2 vs 3
            if ins_label in set([0,1,2]):
                o12vs3_ins_label.append(0)
            elif ins_label in set([3]):
                o12vs3_ins_label.append(1)

        ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = torch.tensor(int(ovs123_bag_label), dtype=torch.long), torch.tensor(int(o1vs23_bag_label), dtype=torch.long), torch.tensor(int(o12vs3_bag_label), dtype=torch.long)
        ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.from_numpy(np.array(ins_labels)), torch.from_numpy(np.array(ovs123_ins_label)), torch.from_numpy(np.array(o1vs23_ins_label)), torch.from_numpy(np.array(o12vs3_ins_label))
        len_list = len(bag)
        return {"bags": bag, "max_label": max_label, "0vs123_bag": ovs123_bag_label, "01vs23_bag": o1vs23_bag_label, "012vs3_bag": o12vs3_bag_label,
                "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                 "len_list": len_list}


def all_one_vs_rest_load_data_bags(args):  # LIMUC
    ######### load data #######
    train_bags = np.load('./bag_data/%s/%s/%d/train_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_ins_labels = np.load('./bag_data/%s/%s/%d/train_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bag_labels = np.load('./bag_data/%s/%s/%d/train_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    
    train_dataset = all_one_vs_rest_LIMUC_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, bag_labels=train_bag_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)
    return train_loader

def collate_fn_custom_for_multicls(batch):
    bags = []
    max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = [], [], [], []
    ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = [], [], [], []
    len_list = []

    for b in batch:
        bags.extend(b["bags"])
        max_label.append(b["max_label"]), ovs123_bag_label.append(b["0vs123_bag"]), o1vs23_bag_label.append(b["01vs23_bag"]), o12vs3_bag_label.append(b["012vs3_bag"])
        ins_labels.extend(b["ins_label"]), ovs123_ins_label.extend(b["0vs123_ins"]), o1vs23_ins_label.extend(b["01vs23_ins"]), o12vs3_ins_label.extend(b["012vs3_ins"])
        len_list.append(b["len_list"])

    max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = torch.stack(max_label, dim=0), torch.stack(ovs123_bag_label, dim=0), torch.stack(o1vs23_bag_label, dim=0), torch.stack(o12vs3_bag_label, dim=0)
    ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.stack(ins_labels, dim=0), torch.stack(ovs123_ins_label, dim=0), torch.stack(o1vs23_ins_label, dim=0), torch.stack(o12vs3_ins_label, dim=0)
    len_list = torch.tensor(len_list)
    bags = torch.stack(bags, dim=0)
    return {"bags": bags, "max_label": max_label, "0vs123_bag": ovs123_bag_label, "01vs23_bag": o1vs23_bag_label, "012vs3_bag": o12vs3_bag_label,
            "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                "len_list": len_list}