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


def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function):
    fix_seed(args.seed)
    log_dict = {"train_bag_acc":[], "train_bag_kap":[], "train_bag_f1":[], "train_emsemble_ins_acc":[], "train_emsemble_ins_kap":[], "train_emsemble_ins_f1":[], "train_loss":[],
                "val_bag_acc":[], "val_bag_kap":[], "val_bag_f1":[], "val_emsemble_ins_acc":[], "val_emsemble_ins_kap":[], "val_emsemble_ins_f1":[],  "val_loss":[],
                "test_bag_acc":[], "test_bag_kap":[], "test_bag_f1":[], "test_emsemble_ins_acc":[], "test_emsemble_ins_kap":[], "test_emsemble_ins_f1":[] }
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)


    best_val_kapp = -1
    cnt = 0
    for epoch in range(args.num_epochs):
        ############ train ###################
        s_time = time()
        model.train()
        max_labels, ins_labels,ins_pred = [], [], []
        losses = []
        for iteration, data in enumerate(train_loader): 
            bags = data["ins"]
            # max_label = data["max_label"]
            ins_label = data["ins_label"]

            bags = bags.to(args.device)
            # max_label = max_label.to(args.device)
            ins_label = ins_label.to(args.device)

            y = model(bags)
            loss = loss_function(y["y_ins"], ins_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ins_labels.extend(ins_label.cpu().detach().numpy())
            ins_pred.extend(y["y_ins"].argmax(1).cpu().detach().numpy())
            losses.append(loss.item())

        ins_labels, ins_pred =   np.array(ins_labels), np.array(ins_pred)
        emsemble_ins_metric = calcurate_metrix(ins_pred, ins_labels)

        log_dict["train_emsemble_ins_acc"].append(emsemble_ins_metric["acc"]), log_dict["train_emsemble_ins_kap"].append(emsemble_ins_metric["kap"]), log_dict["train_emsemble_ins_f1"].append(emsemble_ins_metric["macro-f1"])
        log_dict["train_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_loss"][-1], log_dict["train_emsemble_ins_acc"][-1], log_dict["train_emsemble_ins_kap"][-1], log_dict["train_emsemble_ins_f1"][-1]))
        
        ################# validation ####################
        s_time = time()
        model.eval()
        max_labels, ins_labels, ins_pred = [], [], []
        losses, losses1, losses2, losses3 = [], [], [], []
        with torch.no_grad():              
            for iteration, data in enumerate(val_loader): 
                bags = data["ins"]
                # max_label = data["max_label"]
                ins_label = data["ins_label"]

                bags = bags.to(args.device)
                # max_label = max_label.to(args.device)
                ins_label = ins_label.to(args.device)

                y = model(bags)
                loss = loss_function(y["y_ins"], ins_label)

                ins_labels.extend(ins_label.cpu().detach().numpy())
                ins_pred.extend(y["y_ins"].argmax(1).cpu().detach().numpy())
                losses.append(loss.item())

        ins_labels, ins_pred =   np.array(ins_labels), np.array(ins_pred)
        emsemble_ins_metric = calcurate_metrix(ins_pred, ins_labels)
        log_dict["val_emsemble_ins_acc"].append(emsemble_ins_metric["acc"]), log_dict["val_emsemble_ins_kap"].append(emsemble_ins_metric["kap"]), log_dict["val_emsemble_ins_f1"].append(emsemble_ins_metric["macro-f1"])
        log_dict["val_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_loss"][-1], log_dict["val_emsemble_ins_acc"][-1], log_dict["val_emsemble_ins_kap"][-1], log_dict["val_emsemble_ins_f1"][-1]))
        
        if args.is_test == True:
            ################## test ###################
            s_time = time()
            model.eval()
            max_labels, ins_labels, ins_pred = [], [], []
            len_lists = []
            with torch.no_grad():              
                for iteration, data in enumerate(test_loader): 
                    bags = data["bags"]
                    max_label = data["max_label"]
                    ins_label, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = data["ins_label"], data["0vs123_ins"], data["01vs23_ins"], data["012vs3_ins"]

                    bags = bags.to(args.device)
                    max_label = max_label.to(args.device)
                    ins_label = ins_label.to(args.device)

                    y = model(bags)

                    max_labels.extend(max_label.cpu().detach().numpy()), ins_labels.extend(ins_label.cpu().detach().numpy())
                    ins_pred.extend(y["y_ins"].argmax(1).cpu().detach().numpy())
                    len_lists.extend(data["len_list"].cpu().detach().numpy())

            max_labels, ins_labels, ins_pred =  np.array(max_labels), np.array(ins_labels), np.array(ins_pred)

            emsemble_ins_metric = calcurate_metrix(ins_pred, ins_labels)
            test_emsenble_ins_cm = emsemble_ins_metric["cm"]

            slice_ini=0
            bag_pred = []
            for bag_len in len_lists:
                bag_pred.append(ins_pred[slice_ini:(slice_ini+bag_len)].max())
                slice_ini += bag_len
            bag_pred = np.array(bag_pred)
            test_bag_metric = calcurate_metrix(bag_pred, max_labels)
            test_bag_metric_cm = test_bag_metric["cm"]
            
            log_dict["test_emsemble_ins_acc"].append(emsemble_ins_metric["acc"]), log_dict["test_emsemble_ins_kap"].append(emsemble_ins_metric["kap"]), log_dict["test_emsemble_ins_f1"].append(emsemble_ins_metric["macro-f1"])
            log_dict["test_bag_acc"].append(test_bag_metric["acc"]), log_dict["test_bag_kap"].append(test_bag_metric["kap"]), log_dict["test_bag_f1"].append(test_bag_metric["macro-f1"])

            e_time = time()
            logging.info('[Epoch: %d/%d (%ds)] @Test Bag acc: %.4f, kapp: %.4f, macro-f1: %.4f, @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                        (epoch+1, args.num_epochs, e_time-s_time, log_dict["test_bag_acc"][-1], log_dict["test_bag_kap"][-1], log_dict["test_bag_f1"][-1], log_dict["test_emsemble_ins_acc"][-1], log_dict["test_emsemble_ins_kap"][-1], log_dict["test_emsemble_ins_f1"][-1]))
            
        logging.info('===============================')

        # if epoch%10 == 0:
        #     torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-epoch=%d_model.pkl") % (args.output_path, args.fold, args.seed, epoch))


        if best_val_kapp < log_dict["val_emsemble_ins_kap"][-1]:
            best_val_kapp = log_dict["val_emsemble_ins_kap"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed))
            if args.is_test == True:
                save_confusion_matrix(cm=test_emsenble_ins_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_emsemble_ins.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_emsemble_ins_acc"][epoch], log_dict["test_emsemble_ins_kap"][epoch], log_dict["test_emsemble_ins_f1"][epoch])) 
                save_confusion_matrix(cm=test_bag_metric_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_bag.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_bag_acc"][epoch], log_dict["test_bag_kap"][epoch], log_dict["test_bag_f1"][epoch])) 
        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('[Best Epoch: %d/%d (%ds)] @Val Ins emsemble acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                    (best_epoch+1, args.num_epochs, e_time-s_time, log_dict["val_emsemble_ins_acc"][best_epoch], log_dict["val_emsemble_ins_kap"][best_epoch], log_dict["val_emsemble_ins_f1"][best_epoch]))
        if args.is_test == True:
            logging.info('[Best Epoch: %d/%d (%ds)] @Test Bag acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                        (best_epoch+1, args.num_epochs, e_time-s_time, log_dict["test_bag_acc"][best_epoch], log_dict["test_bag_kap"][best_epoch], log_dict["test_bag_f1"][best_epoch]))
            logging.info('[Best Epoch: %d/%d (%ds)] @Test Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                        (best_epoch+1, args.num_epochs, e_time-s_time, log_dict["test_emsemble_ins_acc"][best_epoch], log_dict["test_emsemble_ins_kap"][best_epoch], log_dict["test_emsemble_ins_f1"][best_epoch]))

        make_loss_graph(args,log_dict['train_loss'], log_dict['val_loss'], "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))

        make_bag_acc_graph(args, log_dict['train_emsemble_ins_acc'], log_dict['val_emsemble_ins_acc'], log_dict['test_emsemble_ins_acc'], "%s/acc_graph/fold=%d_seed=%d_ins-emsemble-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_emsemble_ins_kap'], log_dict['val_emsemble_ins_kap'], log_dict['test_emsemble_ins_kap'], "%s/acc_graph/fold=%d_seed=%d_ins-emsemble-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_emsemble_ins_f1'], log_dict['val_emsemble_ins_f1'], log_dict['test_emsemble_ins_f1'], "%s/acc_graph/fold=%d_seed=%d_ins-emsemble-macrof1-graph.png" % (args.output_path, args.fold, args.seed))

        make_bag_acc_graph(args, log_dict['test_bag_acc'], log_dict['test_bag_acc'], log_dict['test_bag_acc'], "%s/acc_graph/fold=%d_seed=%d_bag-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['test_bag_kap'], log_dict['test_bag_kap'], log_dict['test_bag_kap'], "%s/acc_graph/fold=%d_seed=%d_bag-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['test_bag_f1'], log_dict['test_bag_f1'], log_dict['test_bag_f1'], "%s/acc_graph/fold=%d_seed=%d_bag-macrof1-graph.png" % (args.output_path, args.fold, args.seed))
  
        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return