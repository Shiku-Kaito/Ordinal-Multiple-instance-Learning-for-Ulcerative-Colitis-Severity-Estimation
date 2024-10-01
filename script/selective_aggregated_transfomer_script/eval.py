
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
    max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt, ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred = [], [], [], [], [], [], []
    bag_feat, ins_feat = [], []
    bag_path, attw = [], []
    len_lists = []
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
            ovs123_bag_pred.extend(y["y_0vs123"].argmax(1).cpu().detach().numpy()), o1vs23_bag_pred.extend(y["y_01vs23"].argmax(1).cpu().detach().numpy()), o12vs3_bag_pred.extend(y["y_012vs3"].argmax(1).cpu().detach().numpy())

            bag_feat.extend(y["bag_feat"].cpu().detach().numpy())
            ins_feat.extend(y["ins_feats"])

            bag_path.extend(data["bag_path"])
            attw.extend(y["atten_weight"])
            len_lists.extend(data["len_list"])

    max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt =  np.array(max_labels), np.array(ovs123_bag_gt), np.array(o1vs23_bag_gt), np.array(o12vs3_bag_gt)
    ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred =  np.array(ovs123_bag_pred), np.array(o1vs23_bag_pred), np.array(o12vs3_bag_pred)
    ovs123_bag_metric, o1vs23_bag_metric, o12vs3_bag_metric = calcurate_metrix(ovs123_bag_pred, ovs123_bag_gt), calcurate_metrix(o1vs23_bag_pred, o1vs23_bag_gt), calcurate_metrix(o12vs3_bag_pred, o12vs3_bag_gt)
    test_ovs123_bag_cm, test_o1vs23_bag_cm, test_o12vs3_bag_cm = ovs123_bag_metric["cm"], o1vs23_bag_metric["cm"], o12vs3_bag_metric["cm"]

    bag_emsemble_label = emsemble(ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred)
    emsemble_bag_metric = calcurate_metrix(bag_emsemble_label, max_labels)
    test_emsenble_bag_cm = emsemble_bag_metric["cm"]

    # save_confusion_matrix(cm=test_emsenble_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_emsemble_bag.png") % (args.output_path, args.fold, args.seed),
    #             title='test: acc: %.4f, kapp: %.4f, macro-f1:%.4f' % (emsemble_bag_metric["acc"], emsemble_bag_metric["kap"], emsemble_bag_metric["macro-f1"])) 

    result_dict["bag_acc"], result_dict["bag_kap"], result_dict["bag_macro-f1"] = emsemble_bag_metric["acc"], emsemble_bag_metric["kap"], emsemble_bag_metric["macro-f1"]
    result_dict["ins_acc"], result_dict["ins_kap"], result_dict["ins_macro-f1"] = 0, 0, 0
    result_dict["0vs123bag_acc"], result_dict["0vs123bag_kap"], result_dict["0vs123bag_macro-f1"] = ovs123_bag_metric["acc"], ovs123_bag_metric["kap"], ovs123_bag_metric["macro-f1"]
    result_dict["01vs23bag_acc"], result_dict["01vs23bag_kap"], result_dict["01vs23bag_macro-f1"] = o1vs23_bag_metric["acc"], o1vs23_bag_metric["kap"], o1vs23_bag_metric["macro-f1"]
    result_dict["012vs3bag_acc"], result_dict["012vs3bag_kap"], result_dict["012vs3bag_macro-f1"] = o12vs3_bag_metric["acc"], o12vs3_bag_metric["kap"], o12vs3_bag_metric["macro-f1"]

    result_dict["max_labels"], result_dict["bag_pred"] = [], []
    result_dict["max_labels"].extend(max_labels), result_dict["bag_pred"].extend(bag_emsemble_label)
    
    result_dict["bag_feat"] =   bag_feat
    result_dict["ins_feat"] =   ins_feat
    result_dict["bag_path"] = bag_path
    result_dict["atten_weight"] = attw
    result_dict["len_lists"] = len_lists
    return result_dict
