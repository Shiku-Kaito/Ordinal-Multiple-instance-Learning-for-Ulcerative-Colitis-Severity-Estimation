import torch.utils.data as data
from PIL import Image
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
class POE_LIMUC_Dataset(torch.utils.data.Dataset):
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

        multi_hot_target = torch.zeros(4).long()
        multi_hot_target[list(range(max_label))] = 1
            
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
        return {"bags": bag, "max_label": max_label, "mh_label": multi_hot_target,
                "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                 "len_list": len_list}


def POE_load_data_bags(args):  # LIMUC
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

    train_dataset = POE_LIMUC_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, bag_labels=train_bag_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)
    val_dataset = POE_LIMUC_Dataset(args=args, bags=val_bags, ins_label=val_ins_labels, bag_labels=val_bag_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)  
    test_dataset = POE_LIMUC_Dataset(args=args, bags=test_bags, ins_label=test_ins_labels, bag_labels=test_bag_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom_for_multicls)    
    return train_loader, val_loader, test_loader

def collate_fn_custom_for_multicls(batch):
    bags = []
    max_label, mh_label = [], []
    ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = [], [], [], []
    len_list = []

    for b in batch:
        bags.extend(b["bags"])
        max_label.append(b["max_label"]), mh_label.append(b["mh_label"])
        ins_labels.extend(b["ins_label"]), ovs123_ins_label.extend(b["0vs123_ins"]), o1vs23_ins_label.extend(b["01vs23_ins"]), o12vs3_ins_label.extend(b["012vs3_ins"])
        len_list.append(b["len_list"])

    max_label, multi_hot_target = torch.stack(max_label, dim=0), torch.stack(mh_label, dim=0)
    ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = torch.stack(ins_labels, dim=0), torch.stack(ovs123_ins_label, dim=0), torch.stack(o1vs23_ins_label, dim=0), torch.stack(o12vs3_ins_label, dim=0)
    len_list = torch.tensor(len_list)
    bags = torch.stack(bags, dim=0)
    return {"bags": bags, "max_label": max_label, "mh_label": multi_hot_target,
            "ins_label": ins_labels, "0vs123_ins": ovs123_ins_label, "01vs23_ins": o1vs23_ins_label, "012vs3_ins": o12vs3_ins_label,
                "len_list": len_list}



class dataset_manger(data.Dataset):
    def __init__(self, images_root, data_file, transforms=None, num_output_bins=8):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        self.num_output_bins = num_output_bins
        with open(data_file) as fin:
            for line in fin:
                image_file, image_label = line.split()
                self.labels.append(int(image_label))
                self.images_file.append(image_file)

    def __getitem__(self, index):
        img_file, target = self.images_file[index], self.labels[index]
        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)

        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        multi_hot_target = torch.zeros(self.num_output_bins).long()
        multi_hot_target[list(range(target))] = 1

        return img, target, multi_hot_target

    def __len__(self):
        return len(self.labels)