
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
    max_labels, ins_labels, ovs123_ins_gt, o1vs23_ins_gt, o12vs3_ins_gt, ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred = [], [], [], [], [], [], [], []
    len_lists = []
    with torch.no_grad():              
        for iteration, data in enumerate(test_loader): 
            bags = data["bags"]
            max_label = data["max_label"]
            ins_label, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = data["ins_label"], data["0vs123_ins"], data["01vs23_ins"], data["012vs3_ins"]

            bags = bags.to(args.device)
            max_label = max_label.to(args.device)
            ins_label, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = ins_label.to(args.device), ovs123_ins_label.to(args.device), o1vs23_ins_label.to(args.device), o12vs3_ins_label.to(args.device)

            y = model(bags)

            max_labels.extend(max_label.cpu().detach().numpy()), ins_labels.extend(ins_label.cpu().detach().numpy()), ovs123_ins_gt.extend(ovs123_ins_label.cpu().detach().numpy()), o1vs23_ins_gt.extend(o1vs23_ins_label.cpu().detach().numpy()), o12vs3_ins_gt.extend(o12vs3_ins_label.cpu().detach().numpy())
            ovs123_ins_pred.extend(y["y_0vs123"].argmax(1).cpu().detach().numpy()), o1vs23_ins_pred.extend(y["y_01vs23"].argmax(1).cpu().detach().numpy()), o12vs3_ins_pred.extend(y["y_012vs3"].argmax(1).cpu().detach().numpy())
            len_lists.extend(data["len_list"].cpu().detach().numpy())

        max_labels, ins_labels, ovs123_ins_gt, o1vs23_ins_gt, o12vs3_ins_gt =  np.array(max_labels), np.array(ins_labels), np.array(ovs123_ins_gt), np.array(o1vs23_ins_gt), np.array(o12vs3_ins_gt)
        ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred =  np.array(ovs123_ins_pred), np.array(o1vs23_ins_pred), np.array(o12vs3_ins_pred)
        ovs123_ins_metric, o1vs23_ins_metric, o12vs3_ins_metric = calcurate_metrix(ovs123_ins_pred, ovs123_ins_gt), calcurate_metrix(o1vs23_ins_pred, o1vs23_ins_gt), calcurate_metrix(o12vs3_ins_pred, o12vs3_ins_gt)

        ins_emsemble_label = emsemble(ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred)
        emsemble_ins_metric = calcurate_metrix(ins_emsemble_label, ins_labels)
        test_emsenble_ins_cm = emsemble_ins_metric["cm"]

        slice_ini=0
        bag_pred = []
        for bag_len in len_lists:
            bag_pred.append(ins_emsemble_label[slice_ini:(slice_ini+bag_len)].max())
            slice_ini += bag_len
        bag_pred = np.array(bag_pred)
        test_bag_metric = calcurate_metrix(bag_pred, max_labels)
        test_bag_metric_cm = test_bag_metric["cm"]

    save_confusion_matrix(cm=test_bag_metric_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_emsemble_bag.png") % (args.output_path, args.fold, args.seed),
                title='test: acc: %.4f, kapp: %.4f, macro-f1:%.4f' % (test_bag_metric["acc"], test_bag_metric["kap"], test_bag_metric["macro-f1"])) 

    result_dict["bag_acc"], result_dict["bag_kap"], result_dict["bag_macro-f1"] = test_bag_metric["acc"], test_bag_metric["kap"], test_bag_metric["macro-f1"]
    result_dict["ins_acc"], result_dict["ins_kap"], result_dict["ins_macro-f1"] = emsemble_ins_metric["acc"], emsemble_ins_metric["kap"], emsemble_ins_metric["macro-f1"]
    result_dict["0vs123bag_acc"], result_dict["0vs123bag_kap"], result_dict["0vs123bag_macro-f1"] = ovs123_ins_metric["acc"], ovs123_ins_metric["kap"], ovs123_ins_metric["macro-f1"]
    result_dict["01vs23bag_acc"], result_dict["01vs23bag_kap"], result_dict["01vs23bag_macro-f1"] = o1vs23_ins_metric["acc"], o1vs23_ins_metric["kap"], o1vs23_ins_metric["macro-f1"]
    result_dict["012vs3bag_acc"], result_dict["012vs3bag_kap"], result_dict["012vs3bag_macro-f1"] = o12vs3_ins_metric["acc"], o12vs3_ins_metric["kap"], o12vs3_ins_metric["macro-f1"]

    return result_dict