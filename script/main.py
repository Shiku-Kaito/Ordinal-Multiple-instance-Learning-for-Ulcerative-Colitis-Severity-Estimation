import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from get_module import get_module
import torch.nn.functional as F

def main(args):
    fix_seed(args.seed) 
    train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader = get_module(args)
    args.output_path += '%s/%s/w_pretrain_lr=3e-6/%s/' % (args.dataset, args.data_type, args.mode) 
    make_folder(args)

    if args.is_evaluation == False:
        train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function)
        return
    else:    
        model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
        result_dict = eval_net(args, model, test_loader)   
    return result_dict


if __name__ == '__main__':
    results_dict = {"bag_acc":[], "bag_kap":[], "bag_macro-f1":[], "ins_acc":[], "ins_kap":[], "ins_macro-f1":[],
                    "0vs123bag_acc":[], "0vs123bag_kap":[], "0vs123bag_macro-f1":[],
                    "01vs23bag_acc":[], "01vs23bag_kap":[], "01vs23bag_macro-f1":[],
                    "012vs3bag_acc":[], "012vs3bag_kap":[], "012vs3bag_macro-f1":[], 
                    "max_labels":[], "bag_pred":[], "ins_labels":[],
                    "bag_feat":[], "ins_feat":[]}
    for fold in range(5):
    #    for seed in range(1):
        parser = argparse.ArgumentParser()
        # Data selectiion
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default='LIMUC',
                            type=str, help='LIMUC or private')
        parser.add_argument('--data_type', #書き換え
                            default="5-fold_in_test_balanced_time_order", type=str, help="5-fold_in_test_balanced or 5-fold_in_test_balanced_time_order")
        parser.add_argument('--classes', #書き換え
                            default=4, type=int, help="number of the sampled instnace")
        # Training Setup
        parser.add_argument('--num_epochs', default=1000, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--patience', default=100,
                            type=int, help='patience of early stopping')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--batch_size', default=32,
                            type=int, help='batch size for training.')
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='number of workers for training.')
        parser.add_argument('--lr', default=3e-6,
                            type=float, help='learning rate')        
        parser.add_argument('--is_evaluation', default=1,
                            type=int, help='1 or 0')                        
        # Module Selection
        parser.add_argument('--module',default='Selective_Aggregated_Transfomer', 
                            type=str, help="output_mean or output_max or Feat_agg or krank_mil or Selective_Aggregated_Transfomer or  Feat_agg or Output_max or Att_mil or multi-class_Att_mil or transfomer or one_vs_rest or one_token_OvsR_transfomer_script or shared_OvsR_transfomer")
        parser.add_argument('--mode',default='',    # don't write!
                            type=str, help="")                        
        # Save Path
        parser.add_argument('--output_path',
                            default='./result/', type=str, help="output file name")
        ### Module detail ####
        # Traditional MIL Paramter
        parser.add_argument('--feat_agg_method', default="max",
                            type=str, help='mean or max or p_norm or LSE')  
        parser.add_argument('--p_val',
                            default=4, type=int, help="1 or 4 or 8")
        # transfomer layer num
        parser.add_argument('--transfomer_layer_num', default=1, type=int, help='1 or 2 or 6 or 12')
        # shared_OvsR_transfomer
        parser.add_argument('--clstoken_mask', default=1, type=int, help='0 or 1')
        # Additive Parameter
        parser.add_argument('--add_agg_method', default="TransMIL",
                            type=str, help='Attention_MIL or TransMIL')
        # IBMIL parameter
        parser.add_argument('--c_path',  #nargs='+', 
                            default="./result/LIMUC/5-fold_in_test_balanced_time_order/w_pretrain_lr=3e-6/IBMIL_transMIL/datasets_deconf/fold=0_seed=0-train_bag_cls_agnostic_feats_proto_k=8.npy", type=str,help='directory to confounders')
        # cpl parameter
        parser.add_argument('--constraint', default='S-P', type=str, help='{S-P, S-B, H-L, H-S}')
        parser.add_argument('--metric_method', default='C', type=str, help='{E, C}')
        parser.add_argument('--cosine_scale', default=6., type=float)
        parser.add_argument('--tau', default=0.11, type=float)
        parser.add_argument('--loss_lam', default=6., type=float)
        # POE parameter
        parser.add_argument('--no-sto', action='store_true', default=False,
                    help='not using stochastic sampling when training or testing.')
        parser.add_argument('--distance', type=str, default='JDistance',
                            help='distance metric between two gaussian distribution')
        parser.add_argument('--alpha-coeff', type=float, default=1e-5, metavar='M',
                            help='alpha_coeff (default: 0)')
        parser.add_argument('--beta-coeff', type=float, default=1e-4, metavar='M',
                            help='beta_coeff (default: 1.0)')
        parser.add_argument('--margin', type=float, default=5, metavar='M',
                            help='margin (default: 1.0)')
        parser.add_argument('--main-loss-type', type=str,
                            default='reg', help='loss type in [cls, reg, rank].')
        parser.add_argument('--max-t', type=int, default=50,
                            help='number of samples during sto.')
        parser.add_argument('--num-output-neurons', type=int, default=1,
                            help='number of ouput neurons of your model, note that for `reg` model we use 1; `cls` model we use `num_output_classes`; and for `rank` model we use `num_output_class` * 2.')
        # k-rank mil
        parser.add_argument('--emsemble_mode',default='threshold', type=str, help="convert or threshold or sum")   
        args = parser.parse_args()

        args.c_path = "%s/%s/%s/w_pretrain_lr=3e-6/IBMIL_transMIL/datasets_deconf/fold=%d_seed=0-train_bag_cls_agnostic_feats_proto_k=8.npy" % (args.output_path, args.dataset, args.data_type, args.fold)

        # train
        if args.is_evaluation == False:
            main(args)
        # evalation
        else:
            result_dict = main(args)
            results_dict["bag_acc"].append(result_dict["bag_acc"]), results_dict["bag_kap"].append(result_dict["bag_kap"]), results_dict["bag_macro-f1"].append(result_dict["bag_macro-f1"])

            print("############ fold=%d ###########" % args.fold)
            print("@ Bag acc:%.5f, kap:%.5f, macro-f1:%.5f" % (float(result_dict["bag_acc"]), float(result_dict["bag_kap"]), float(result_dict["bag_macro-f1"])))
                 
    if args.is_evaluation == True:
        print("5-fold cross-validation")
        print("@Bag acc:%.5f±%.5f, kap:%.5f±%.5f, macro-f1:%.5f±%.5f" % ((np.array(results_dict["bag_acc"]).mean()), np.std(np.array(results_dict["bag_acc"])), (np.array(results_dict["bag_kap"]).mean()), np.std(np.array(results_dict["bag_kap"])), (np.array(results_dict["bag_macro-f1"]).mean()), np.std(np.array(results_dict["bag_macro-f1"]))))
        np.save("%s/test_metrics/test_metrics_log" % (args.output_path) , results_dict)