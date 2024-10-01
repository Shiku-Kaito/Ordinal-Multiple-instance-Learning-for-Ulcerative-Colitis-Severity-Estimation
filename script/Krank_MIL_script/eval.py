
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
    
    s_time = time()
    model.eval()
    max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt, ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred = [], [], [], [], [], [], []
    ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred = [], [], []
    len_list = []
    with torch.no_grad():
        for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
            bags = data["bags"]
            max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = data["max_label"], data["0vs123_bag"], data["01vs23_bag"], data["012vs3_bag"]
            ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = data["ins_label"], data["0vs123_ins"], data["01vs23_ins"], data["012vs3_ins"]

            bags = bags.to(args.device)
            max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = max_label.to(args.device), ovs123_bag_label.to(args.device), o1vs23_bag_label.to(args.device), o12vs3_bag_label.to(args.device)
            ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = ins_labels.to(args.device), ovs123_ins_label.to(args.device), o1vs23_ins_label.to(args.device), o12vs3_ins_label.to(args.device)

            y = model(bags, data["len_list"])

            max_labels.extend(max_label.cpu().detach().numpy()), ovs123_bag_gt.extend(ovs123_bag_label.cpu().detach().numpy()), o1vs23_bag_gt.extend(o1vs23_bag_label.cpu().detach().numpy()), o12vs3_bag_gt.extend(o12vs3_bag_label.cpu().detach().numpy())
            ovs123_bag_pred.extend( torch.sigmoid(y["y_0vs123"]).cpu().detach().numpy()), o1vs23_bag_pred.extend(torch.sigmoid(y["y_01vs23"]).cpu().detach().numpy()), o12vs3_bag_pred.extend(torch.sigmoid(y["y_012vs3"]).cpu().detach().numpy())
            ovs123_ins_pred.extend(y["y_ins_0vs123"].cpu().detach().numpy()), o1vs23_ins_pred.extend(y["y_ins_01vs23"].cpu().detach().numpy()), o12vs3_ins_pred.extend(y["y_ins_012vs3"].cpu().detach().numpy())
            len_list.extend(data["len_list"])

    max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt =  np.array(max_labels), np.array(ovs123_bag_gt), np.array(o1vs23_bag_gt), np.array(o12vs3_bag_gt)
    ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred = np.array(ovs123_ins_pred), np.array(o1vs23_ins_pred), np.array(o12vs3_ins_pred)
    # ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred =  np.array(ovs123_bag_pred), np.array(o1vs23_bag_pred), np.array(o12vs3_bag_pred)
    # ovs123_bag_metric, o1vs23_bag_metric, o12vs3_bag_metric = calcurate_metrix(ovs123_bag_pred, ovs123_bag_gt), calcurate_metrix(o1vs23_bag_pred, o1vs23_bag_gt), calcurate_metrix(o12vs3_bag_pred, o12vs3_bag_gt)
    # test_ovs123_bag_cm, test_o1vs23_bag_cm, test_o12vs3_bag_cm = ovs123_bag_metric["cm"], o1vs23_bag_metric["cm"], o12vs3_bag_metric["cm"]

    bag_emsemble_label = output_krank_emsemble(ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred, len_list, args.emsemble_mode)
    bag_metric = calcurate_metrix(bag_emsemble_label, max_labels)


    result_dict["bag_acc"], result_dict["bag_kap"], result_dict["bag_macro-f1"] = bag_metric["acc"], bag_metric["kap"], bag_metric["macro-f1"]
    # result_dict["ins_acc"], result_dict["ins_kap"], result_dict["ins_macro-f1"] = inst_metric["acc"], inst_metric["kap"], inst_metric["macro-f1"]
    return result_dict