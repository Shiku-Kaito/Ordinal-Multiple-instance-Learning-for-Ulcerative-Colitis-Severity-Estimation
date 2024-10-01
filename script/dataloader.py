import os
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
                 "len_list": len_list, "bag_path":bag_paths}


def all_one_vs_rest_load_data_bags(args):  # LIMUC
    ######### load data #######
    test_bags = np.load('./bag_data/%s/%s/%d/test_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_ins_labels = np.load('./bag_data/%s/%s/%d/test_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_bag_labels = np.load('./bag_data/%s/%s/%d/test_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bags = np.load('./bag_data/%s/%s/%d/train_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_ins_labels = np.load('./bag_data/%s/%s/%d/train_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bag_labels = np.load('./bag_data/%s/%s/%d/train_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bags = np.load('./bag_data/%s/%s/%d/val_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_ins_labels = np.load('./bag_data/%s/%s/%d/val_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bag_labels = np.load('./bag_data/%s/%s/%d/val_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)

    # sampler
    bag_label_count = np.array([sum(train_bag_labels==0), sum(train_bag_labels==1), sum(train_bag_labels==2), sum(train_bag_labels==3)])
    class_weight = 1 / bag_label_count
    sample_weight = [class_weight[train_bag_labels[i]] for i in range(len(train_bag_labels))]
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_bag_labels), replacement=True)

    train_dataset = all_one_vs_rest_LIMUC_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, bag_labels=train_bag_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)
    val_dataset = all_one_vs_rest_LIMUC_Dataset(args=args, bags=val_bags, ins_label=val_ins_labels, bag_labels=val_bag_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)  
    test_dataset = all_one_vs_rest_LIMUC_Dataset(args=args, bags=test_bags, ins_label=test_ins_labels, bag_labels=test_bag_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)    
    return train_loader, val_loader, test_loader

def collate_fn_custom_for_multicls(batch):
    bags = []
    max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = [], [], [], []
    ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = [], [], [], []
    len_list = []
    bag_path = []

    for b in batch:
        bags.extend(b["bags"])
        max_label.append(b["max_label"]), ovs123_bag_label.append(b["0vs123_bag"]), o1vs23_bag_label.append(b["01vs23_bag"]), o12vs3_bag_label.append(b["012vs3_bag"])
        ins_labels.extend(b["ins_label"]), ovs123_ins_label.extend(b["0vs123_ins"]), o1vs23_ins_label.extend(b["01vs23_ins"]), o12vs3_ins_label.extend(b["012vs3_ins"])
        len_list.append(b["len_list"])
        bag_path.append(b["bag_path"])

    max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = torch.stack(max_label, dim=0), torch.stack(ovs123_bag_label, dim=0), torch.stack(o1vs23_bag_label, dim=0), torch.stack(o12vs3_bag_label, dim=0)
    ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.stack(ins_labels, dim=0), torch.stack(ovs123_ins_label, dim=0), torch.stack(o1vs23_ins_label, dim=0), torch.stack(o12vs3_ins_label, dim=0)
    len_list = torch.tensor(len_list)
    bags = torch.stack(bags, dim=0)
    return {"bags": bags, "max_label": max_label, "0vs123_bag": ovs123_bag_label, "01vs23_bag": o1vs23_bag_label, "012vs3_bag": o12vs3_bag_label,
            "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                "len_list": len_list, "bag_path":bag_path}



def make_soft_labal(r_t, Y=[0,1,2,3]):
    K = len(Y)
    encoded_vector = np.zeros(K)
    for i, r_i in enumerate(Y):
        encoded_vector[i] = np.exp(-abs(r_t-r_i))
    soft_label = encoded_vector / np.sum(encoded_vector)
    return soft_label   
#toy
class SORD_LIMUC_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, bags, ins_label, bag_labels):
        np.random.seed(args.seed)
        self.bags_paths = bags
        self.ins_label = ins_label
        self.bag_labels = bag_labels
        self.len = len(self.bags_paths)
        self.soft_label0 = make_soft_labal(r_t=0)
        self.soft_label1 = make_soft_labal(r_t=1)
        self.soft_label2 = make_soft_labal(r_t=2)
        self.soft_label3 = make_soft_labal(r_t=3)
        
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
        if max_label==0:
            max_label = self.soft_label0
        elif max_label==1:
            max_label = self.soft_label1
        elif max_label==2:
            max_label = self.soft_label2
        elif max_label==3:
            max_label = self.soft_label3
            
        max_label = torch.tensor(max_label)

        # ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = self.ins_label[idx], self.ins_label[idx], self.ins_label[idx], self.ins_label[idx]

        ins_labels = self.ins_label[idx]

        ins_labels = torch.from_numpy(np.array(ins_labels))
        len_list = len(bag)
        return {"bags": bag, "max_label": max_label, "ins_label": ins_labels, "len_list": len_list}


def SORD_load_data_bags(args):  # LIMUC
    ######### load data #######
    test_bags = np.load('./bag_data/%s/%s/%d/test_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_ins_labels = np.load('./bag_data/%s/%s/%d/test_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_bag_labels = np.load('./bag_data/%s/%s/%d/test_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bags = np.load('./bag_data/%s/%s/%d/train_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_ins_labels = np.load('./bag_data/%s/%s/%d/train_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bag_labels = np.load('./bag_data/%s/%s/%d/train_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bags = np.load('./bag_data/%s/%s/%d/val_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_ins_labels = np.load('./bag_data/%s/%s/%d/val_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bag_labels = np.load('./bag_data/%s/%s/%d/val_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)

    # sampler
    bag_label_count = np.array([sum(train_bag_labels==0), sum(train_bag_labels==1), sum(train_bag_labels==2), sum(train_bag_labels==3)])
    class_weight = 1 / bag_label_count
    sample_weight = [class_weight[train_bag_labels[i]] for i in range(len(train_bag_labels))]
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_bag_labels), replacement=True)

    train_dataset = SORD_LIMUC_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, bag_labels=train_bag_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_custom_for_SORD)
    val_dataset = all_one_vs_rest_LIMUC_Dataset(args=args, bags=val_bags, ins_label=val_ins_labels, bag_labels=val_bag_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)  
    test_dataset = all_one_vs_rest_LIMUC_Dataset(args=args, bags=test_bags, ins_label=test_ins_labels, bag_labels=test_bag_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)    
    return train_loader, val_loader, test_loader

def collate_fn_custom_for_SORD(batch):
    bags = []
    max_label = []
    ins_labels = []
    len_list = []

    for b in batch:
        bags.extend(b["bags"])
        max_label.append(b["max_label"])
        ins_labels.extend(b["ins_label"])
        len_list.append(b["len_list"])

    max_label = torch.stack(max_label, dim=0)
    ins_labels = torch.stack(ins_labels, dim=0)
    len_list = torch.tensor(len_list)
    bags = torch.stack(bags, dim=0)
    return {"bags": bags, "max_label": max_label, "ins_label": ins_labels, "len_list": len_list}



#toy
class regression_LIMUC_Dataset(torch.utils.data.Dataset):
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
        max_label = torch.tensor(float(max_label))
        max_label_CL = torch.tensor(int(max_label), dtype=torch.long)

        ins_labels = self.ins_label[idx]

        ins_labels = torch.from_numpy(np.array(ins_labels))
        len_list = len(bag)
        return {"bags": bag, "max_label": max_label, "ins_label": ins_labels, "len_list": len_list, "max_label_CL":max_label_CL, "bag_path":bag_paths}


def regression_load_data_bags(args):  # LIMUC
    ######### load data #######
    test_bags = np.load('./bag_data/%s/%s/%d/test_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_ins_labels = np.load('./bag_data/%s/%s/%d/test_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_bag_labels = np.load('./bag_data/%s/%s/%d/test_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bags = np.load('./bag_data/%s/%s/%d/train_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_ins_labels = np.load('./bag_data/%s/%s/%d/train_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bag_labels = np.load('./bag_data/%s/%s/%d/train_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bags = np.load('./bag_data/%s/%s/%d/val_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_ins_labels = np.load('./bag_data/%s/%s/%d/val_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bag_labels = np.load('./bag_data/%s/%s/%d/val_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)

    # sampler
    bag_label_count = np.array([sum(train_bag_labels==0), sum(train_bag_labels==1), sum(train_bag_labels==2), sum(train_bag_labels==3)])
    class_weight = 1 / bag_label_count
    sample_weight = [class_weight[train_bag_labels[i]] for i in range(len(train_bag_labels))]
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_bag_labels), replacement=True)

    train_dataset = regression_LIMUC_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, bag_labels=train_bag_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_custom_for_reg)
    val_dataset = regression_LIMUC_Dataset(args=args, bags=val_bags, ins_label=val_ins_labels, bag_labels=val_bag_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_reg)  
    test_dataset = regression_LIMUC_Dataset(args=args, bags=test_bags, ins_label=test_ins_labels, bag_labels=test_bag_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_reg)    
    return train_loader, val_loader, test_loader

def collate_fn_custom_for_reg(batch):
    bags = []
    max_label, max_label_CL = [], []
    ins_labels = []
    len_list = []
    bag_path = []

    for b in batch:
        bags.extend(b["bags"])
        max_label.append(b["max_label"])
        ins_labels.extend(b["ins_label"])
        len_list.append(b["len_list"])
        max_label_CL.append(b["max_label_CL"])
        bag_path.append(b["bag_path"])

    max_label = torch.stack(max_label, dim=0)
    ins_labels = torch.stack(ins_labels, dim=0)
    len_list = torch.tensor(len_list)
    bags = torch.stack(bags, dim=0)
    max_label_CL = torch.stack(max_label_CL, dim=0)
    return {"bags": bags, "max_label": max_label, "ins_label": ins_labels, "len_list": len_list, "max_label_CL":max_label_CL, "bag_path":bag_path}


#toy
class all_one_vs_rest_LIMUC_time_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, bags, ins_label, bag_labels):
        np.random.seed(args.seed)
        self.bags_paths = bags
        self.ins_label = ins_label
        self.bag_labels = bag_labels
        self.len = len(self.bags_paths)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        bag_paths = self.bags_paths[idx]  # bagの内のinstanceのパスが格納されているファイルを読みコム

        bag, frame_list = [], []
        for bag_path in bag_paths:
            img = Image.open(bag_path)
            img = np.array(img.resize((224, 224)))
            img = img / 255
            bag.append(img)

            frame = bag_path.split("_")[-1]
            frame = frame.split(".")[0]
            frame_list.append(int(frame))
        if min(frame_list)==0:
            print(min(frame_list))

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
 
        ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = torch.tensor(ovs123_bag_label, dtype=torch.float32), torch.tensor(o1vs23_bag_label, dtype=torch.float32), torch.tensor(o12vs3_bag_label, dtype=torch.float32)           
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

        ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.from_numpy(np.array(ins_labels)), torch.from_numpy(np.array(ovs123_ins_label)), torch.from_numpy(np.array(o1vs23_ins_label)), torch.from_numpy(np.array(o12vs3_ins_label))
        len_list = len(bag)
        frame_list = torch.tensor(frame_list)
        return {"bags": bag, "max_label": max_label, "0vs123_bag": ovs123_bag_label, "01vs23_bag": o1vs23_bag_label, "012vs3_bag": o12vs3_bag_label,
                "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                 "len_list": len_list,"frame_label":frame_list}

#toy
class all_one_vs_rest_LIMUC_time_regression_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, bags, ins_label, bag_labels):
        np.random.seed(args.seed)
        self.bags_paths = bags
        self.ins_label = ins_label
        self.bag_labels = bag_labels
        self.len = len(self.bags_paths)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        bag_paths = self.bags_paths[idx]  # bagの内のinstanceのパスが格納されているファイルを読みコム

        bag, frame_list = [], []
        for bag_path in bag_paths:
            img = Image.open(bag_path)
            img = np.array(img.resize((224, 224)))
            img = img / 255
            bag.append(img)

            frame = bag_path.split("_")[-1]
            frame = frame.split(".")[0]
            frame_list.append(int(frame))
        if min(frame_list)==0:
            print(min(frame_list))

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
 
        ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = torch.tensor(ovs123_bag_label, dtype=torch.float32), torch.tensor(o1vs23_bag_label, dtype=torch.float32), torch.tensor(o12vs3_bag_label, dtype=torch.float32)           
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

        ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.from_numpy(np.array(ins_labels)), torch.from_numpy(np.array(ovs123_ins_label)), torch.from_numpy(np.array(o1vs23_ins_label)), torch.from_numpy(np.array(o12vs3_ins_label))
        len_list = len(bag)
        frame_list = torch.tensor(frame_list)
        return {"bags": bag, "max_label": max_label, "0vs123_bag": ovs123_bag_label, "01vs23_bag": o1vs23_bag_label, "012vs3_bag": o12vs3_bag_label,
                "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                 "len_list": len_list,"frame_label":frame_list, "bag_path":bag_paths}


def all_one_vs_rest_load_LIMUC_regression_data_bags(args):  # LIMUC
    ######### load data #######
    test_bags = np.load('./bag_data/%s/%s/%d/test_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_ins_labels = np.load('./bag_data/%s/%s/%d/test_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_bag_labels = np.load('./bag_data/%s/%s/%d/test_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bags = np.load('./bag_data/%s/%s/%d/train_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_ins_labels = np.load('./bag_data/%s/%s/%d/train_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bag_labels = np.load('./bag_data/%s/%s/%d/train_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bags = np.load('./bag_data/%s/%s/%d/val_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_ins_labels = np.load('./bag_data/%s/%s/%d/val_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bag_labels = np.load('./bag_data/%s/%s/%d/val_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)

    # sampler
    bag_label_count = np.array([sum(train_bag_labels==0), sum(train_bag_labels==1), sum(train_bag_labels==2), sum(train_bag_labels==3)])
    class_weight = 1 / bag_label_count
    sample_weight = [class_weight[train_bag_labels[i]] for i in range(len(train_bag_labels))]
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_bag_labels), replacement=True)

    train_dataset = all_one_vs_rest_LIMUC_time_regression_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, bag_labels=train_bag_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)
    val_dataset = all_one_vs_rest_LIMUC_time_regression_Dataset(args=args, bags=val_bags, ins_label=val_ins_labels, bag_labels=val_bag_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)  
    test_dataset = all_one_vs_rest_LIMUC_time_regression_Dataset(args=args, bags=test_bags, ins_label=test_ins_labels, bag_labels=test_bag_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)    
    return train_loader, val_loader, test_loader

#toy
class Supervised_LIMUC_time_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, ins, ins_label):
        np.random.seed(args.seed)
        self.ins_paths = ins
        self.ins_label = ins_label
        self.len = len(self.ins_paths)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        ins_path = self.ins_paths[idx]  # bagの内のinstanceのパスが格納されているファイルを読みコム

        img = Image.open(ins_path)
        img = np.array(img.resize((224, 224)))
        img = img / 255

        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))

        ins_label = self.ins_label[idx]
        # 0 vs 1,2,3
        if ins_label in set([0]):
            ovs123_ins_label = 0
        elif ins_label in set([1,2,3]):
            ovs123_ins_label = 1
        # 0,1 vs 2,3
        if ins_label in set([0,1]):
            o1vs23_ins_label = 0
        elif ins_label in set([2,3]):
            o1vs23_ins_label = 1
        # 0,1,2 vs 3
        if ins_label in set([0,1,2]):
            o12vs3_ins_label = 0
        elif ins_label in set([3]):
            o12vs3_ins_label = 1
            
        ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.tensor(int(ovs123_ins_label), dtype=torch.long), torch.tensor(int(o1vs23_ins_label), dtype=torch.long), torch.tensor(int(o12vs3_ins_label), dtype=torch.long)
        ins_label = torch.tensor(int(ins_label), dtype=torch.long)
        
        return {"ins": img, "ins_label": ins_label, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label}


def supervised_load_LIMUC_time_data_bags(args):  # LIMUC
    ######### load data #######
    test_bags = np.load('./bag_data/%s/%s/%d/test_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_ins_labels = np.load('./bag_data/%s/%s/%d/test_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    test_bag_labels = np.load('./bag_data/%s/%s/%d/test_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    pre_train_bags = np.load('./bag_data/%s/%s/%d/train_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    pre_train_ins_labels = np.load('./bag_data/%s/%s/%d/train_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    train_bag_labels = np.load('./bag_data/%s/%s/%d/train_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    pre_val_bags = np.load('./bag_data/%s/%s/%d/val_bags.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    pre_val_ins_labels = np.load('./bag_data/%s/%s/%d/val_ins_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)
    val_bag_labels = np.load('./bag_data/%s/%s/%d/val_bag_labels.npy' % (args.dataset, args.data_type, args.fold), allow_pickle=True)

    # sampler
    train_ins, train_ins_labels = [], []
    for bag_idx in range(len(pre_train_bags)):
        train_ins.extend(pre_train_bags[bag_idx]), train_ins_labels.extend(pre_train_ins_labels[bag_idx])        
    val_ins, val_ins_labels = [], []
    for bag_idx in range(len(pre_val_bags)):
        val_ins.extend(pre_val_bags[bag_idx]), val_ins_labels.extend(pre_val_ins_labels[bag_idx])

    train_ins, train_ins_labels, val_ins, val_ins_labels = np.array(train_ins), np.array(train_ins_labels), np.array(val_ins), np.array(val_ins_labels)

    ins_label_count = np.array([sum(train_ins_labels==0), sum(train_ins_labels==1), sum(train_ins_labels==2), sum(train_ins_labels==3)])
    class_weight = 1 / ins_label_count
    sample_weight = [class_weight[train_ins_labels[i]] for i in range(len(train_ins_labels))]
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_ins_labels), replacement=True)

    train_dataset = Supervised_LIMUC_time_Dataset(args=args, ins=train_ins, ins_label=train_ins_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataset = Supervised_LIMUC_time_Dataset(args=args, ins=val_ins, ins_label=val_ins_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = all_one_vs_rest_LIMUC_time_Dataset(args=args, bags=test_bags, ins_label=test_ins_labels, bag_labels=test_bag_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)       
    return train_loader, val_loader, test_loader