
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
    log_dict = {"train_bag_acc":[], "train_ins_acc":[], "train_bag_kap":[], "train_ins_kap":[], "train_bag_f1":[], "train_ins_f1":[], "train_mIoU":[], "train_loss":[], 
                "val_bag_acc":[], "val_ins_acc":[], "val_bag_kap":[], "val_ins_kap":[], "val_bag_f1":[], "val_ins_f1":[], "val_mIoU":[], "val_loss":[], 
                "test_bag_acc":[], "test_ins_acc":[], "test_bag_kap":[], "test_ins_kap":[], "test_bag_f1":[], "test_ins_f1":[], "test_mIoU":[], "test_loss":[]}
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
        ins_gt, bag_gt, ins_pred, bag_pred, losses = [], [], [], [], []
        for iteration, data in enumerate(train_loader): #enumerate(tqdm(train_loader, leave=False)):
            bags, ins_label, bag_label = data["bags"], data["ins_label"], data["max_label"]
            bags, ins_label, bag_label = bags.to(args.device), ins_label.to(args.device), bag_label.to(args.device)

            y = model(x=bags, len_list=data["len_list"])
            loss = loss_function(y["bag"], bag_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label.cpu().detach().numpy())
            ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
            losses.append(loss.item())

        ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
        bag_metric = calcurate_metrix(bag_pred, bag_gt)
        inst_metric = calcurate_metrix(ins_pred, ins_gt)
        train_ins_cm, train_bag_cm = inst_metric["cm"], bag_metric["cm"]

        log_dict["train_ins_acc"].append(inst_metric["acc"]), log_dict["train_ins_kap"].append(inst_metric["kap"]), log_dict["train_ins_f1"].append(inst_metric["macro-f1"])
        log_dict["train_bag_acc"].append(bag_metric["acc"]), log_dict["train_bag_kap"].append(bag_metric["kap"]), log_dict["train_bag_f1"].append(bag_metric["macro-f1"])
        log_dict["train_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, @bag acc: %.4f, kapp: %.4f, macro-f1: %.4f,  @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_loss"][-1], log_dict["train_bag_acc"][-1], log_dict["train_bag_kap"][-1], log_dict["train_bag_f1"][-1],
                      log_dict["train_ins_acc"][-1], log_dict["train_ins_kap"][-1], log_dict["train_ins_f1"][-1]))
        
        ################# validation ####################
        s_time = time()
        model.eval()
        ins_gt, bag_gt, ins_pred, bag_pred, losses = [], [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(val_loader): #enumerate(tqdm(val_loader, leave=False)):
                bags, ins_label, bag_label = data["bags"], data["ins_label"], data["max_label"]
                bags, ins_label, bag_label = bags.to(args.device), ins_label.to(args.device), bag_label.to(args.device)

                y = model(x=bags, len_list=data["len_list"])
                loss = loss_function(y["bag"], bag_label)

                ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label.cpu().detach().numpy())
                ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
                losses.append(loss.item())

        ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
        bag_metric = calcurate_metrix(bag_pred, bag_gt)
        inst_metric = calcurate_metrix(ins_pred, ins_gt)
        val_ins_cm, val_bag_cm = inst_metric["cm"], bag_metric["cm"]

        log_dict["val_ins_acc"].append(inst_metric["acc"]), log_dict["val_ins_kap"].append(inst_metric["kap"]), log_dict["val_ins_f1"].append(inst_metric["macro-f1"])
        log_dict["val_bag_acc"].append(bag_metric["acc"]), log_dict["val_bag_kap"].append(bag_metric["kap"]), log_dict["val_bag_f1"].append(bag_metric["macro-f1"])
        log_dict["val_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, @bag acc: %.4f, kapp: %.4f, macro-f1: %.4f,  @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_loss"][-1], log_dict["val_bag_acc"][-1], log_dict["val_bag_kap"][-1], log_dict["val_bag_f1"][-1],
                      log_dict["val_ins_acc"][-1], log_dict["val_ins_kap"][-1], log_dict["val_ins_f1"][-1]))

        ################## test ###################
        s_time = time()
        model.eval()
        ins_gt, bag_gt, ins_pred, bag_pred = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
                bags, ins_label, bag_label = data["bags"], data["ins_label"], data["max_label"]
                bags, ins_label, bag_label = bags.to(args.device), ins_label.to(args.device), bag_label.to(args.device)

                y = model(x=bags, len_list=data["len_list"])

                ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label.cpu().detach().numpy())
                ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())

        ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
        bag_metric = calcurate_metrix(bag_pred, bag_gt)
        inst_metric = calcurate_metrix(ins_pred, ins_gt)
        test_ins_cm, test_bag_cm = inst_metric["cm"], bag_metric["cm"]

        log_dict["test_ins_acc"].append(inst_metric["acc"]), log_dict["test_ins_kap"].append(inst_metric["kap"]), log_dict["test_ins_f1"].append(inst_metric["macro-f1"])
        log_dict["test_bag_acc"].append(bag_metric["acc"]), log_dict["test_bag_kap"].append(bag_metric["kap"]), log_dict["test_bag_f1"].append(bag_metric["macro-f1"])

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)]  @bag acc: %.4f, kapp: %.4f, macro-f1: %.4f,  @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                    (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_bag_acc"][-1], log_dict["val_bag_kap"][-1], log_dict["val_bag_f1"][-1],
                    log_dict["val_ins_acc"][-1], log_dict["val_ins_kap"][-1], log_dict["val_ins_f1"][-1]))
        logging.info('===============================')

        if best_val_kapp < log_dict["val_bag_kap"][-1]:
            best_val_kapp = log_dict["val_bag_kap"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed))
            save_confusion_matrix(cm=train_ins_cm, path=("%s/cm/fold=%d_seed=%d-cm_train_ins.png") % (args.output_path, args.fold, args.seed),
                        title='train: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["train_ins_acc"][epoch], log_dict["train_ins_kap"][epoch], log_dict["train_ins_f1"][epoch]))
            save_confusion_matrix(cm=train_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_train_bag.png") % (args.output_path, args.fold, args.seed),
                        title='train: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["train_bag_acc"][epoch], log_dict["train_bag_kap"][epoch], log_dict["train_bag_f1"][epoch]))
            save_confusion_matrix(cm=val_ins_cm, path=("%s/cm/fold=%d_seed=%d-cm_val_ins.png") % (args.output_path, args.fold, args.seed),
                        title='val: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["val_ins_acc"][epoch], log_dict["val_ins_kap"][epoch], log_dict["val_ins_f1"][epoch]))
            save_confusion_matrix(cm=val_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_val_bag.png") % (args.output_path, args.fold, args.seed),
                        title='val: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["val_bag_acc"][epoch], log_dict["val_bag_kap"][epoch], log_dict["val_bag_f1"][epoch]))
            if args.is_test == True:
                save_confusion_matrix(cm=test_ins_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_ins.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_ins_acc"][epoch], log_dict["test_ins_kap"][epoch], log_dict["test_ins_f1"][epoch]))
                save_confusion_matrix(cm=test_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_bag.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_bag_acc"][epoch], log_dict["test_bag_kap"][epoch], log_dict["test_bag_f1"][epoch]))
        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('best epoch: %d, @val bag acc: %.4f, kapp: %.4f, macro-f1: %.4f,  @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                        (best_epoch+1, log_dict["val_bag_acc"][best_epoch], log_dict["val_bag_kap"][best_epoch], log_dict["val_bag_f1"][best_epoch], 
                         log_dict["val_ins_acc"][best_epoch], log_dict["val_ins_kap"][best_epoch], log_dict["val_ins_f1"][best_epoch]))
        if args.is_test == True:
            logging.info('best epoch: %d, @test bag acc: %.4f, kapp: %.4f, macro-f1: %.4f,  @Ins acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                            (best_epoch+1, log_dict["test_bag_acc"][best_epoch], log_dict["test_bag_kap"][best_epoch], log_dict["test_bag_f1"][best_epoch], 
                            log_dict["test_ins_acc"][best_epoch], log_dict["test_ins_kap"][best_epoch], log_dict["test_ins_f1"][best_epoch]))

        make_loss_graph(args,log_dict['train_loss'], log_dict['val_loss'], "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_bag_acc'], log_dict['val_bag_acc'], log_dict['test_bag_acc'], "%s/acc_graph/fold=%d_seed=%d_bag-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_ins_acc_graph(args, log_dict['train_ins_acc'], log_dict['val_ins_acc'], log_dict['test_ins_acc'], "%s/acc_graph/fold=%d_seed=%d_ins-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_bag_kap'], log_dict['val_bag_kap'], log_dict['test_bag_kap'], "%s/acc_graph/fold=%d_seed=%d_bag-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_ins_acc_graph(args, log_dict['train_ins_kap'], log_dict['val_ins_kap'], log_dict['test_ins_kap'], "%s/acc_graph/fold=%d_seed=%d_ins-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_bag_f1'], log_dict['val_bag_f1'], log_dict['test_bag_f1'], "%s/acc_graph/fold=%d_seed=%d_bag-macrof1-graph.png" % (args.output_path, args.fold, args.seed))
        make_ins_acc_graph(args, log_dict['train_ins_f1'], log_dict['val_ins_f1'], log_dict['test_ins_f1'], "%s/acc_graph/fold=%d_seed=%d_ins-macrof1-graph.png" % (args.output_path, args.fold, args.seed))
        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return