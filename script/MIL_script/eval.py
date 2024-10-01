
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from utils import *

def eval_net(args, model, test_loader):
    fix_seed(args.seed)
    result_dict = {}
    ################## test ###################
    s_time = time()
    model.eval()
    ins_gt, bag_gt, ins_pred, bag_pred = [], [], [], []
    with torch.no_grad():
        for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
            bags, ins_label, bag_label = data["bags"], data["ins_label"], data["max_label"]
            bags, ins_label, bag_label = bags.to(args.device), ins_label.to(args.device), bag_label.to(args.device)

            y = model(x=bags, len_list=data["len_list"])

            bag_gt.extend(bag_label.cpu().detach().numpy())
            bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())

    bag_gt, bag_pred =np.array(bag_gt), np.array(bag_pred)
    bag_metric = calcurate_metrix(bag_pred, bag_gt)
    inst_metric = calcurate_metrix(ins_pred, ins_gt)

    result_dict["bag_acc"], result_dict["bag_kap"], result_dict["bag_macro-f1"] = bag_metric["acc"], bag_metric["kap"], bag_metric["macro-f1"]
    result_dict["ins_acc"], result_dict["ins_kap"], result_dict["ins_macro-f1"] = inst_metric["acc"], inst_metric["kap"], inst_metric["macro-f1"]
    return result_dict