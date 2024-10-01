
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
    loss_f_0vs123, loss_f_01vs23, loss_f_012vs3 = loss_function["0vs123"], loss_function["01vs23"], loss_function["012vs3"]
    log_dict = {"train_ovs123_bag_acc":[], "train_ovs123_bag_kap":[], "train_ovs123_bag_f1":[], "train_o1vs23_bag_acc":[], "train_o1vs23_bag_kap":[], "train_o1vs23_bag_f1":[], "train_o12vs3_bag_acc":[], "train_o12vs3_bag_kap":[], "train_emsemble_bag_f1":[], "train_emsemble_bag_acc":[], "train_emsemble_bag_kap":[], "train_o12vs3_bag_f1":[], "train_loss":[], "train_loss1":[], "train_loss2":[], "train_loss3":[],
                "val_ovs123_bag_acc":[], "val_ovs123_bag_kap":[], "val_ovs123_bag_f1":[], "val_o1vs23_bag_acc":[], "val_o1vs23_bag_kap":[], "val_o1vs23_bag_f1":[], "val_o12vs3_bag_acc":[], "val_o12vs3_bag_kap":[], "val_o12vs3_bag_f1":[], "val_emsemble_bag_f1":[], "val_emsemble_bag_acc":[], "val_emsemble_bag_kap":[],  "val_loss":[], "val_loss1":[], "val_loss2":[], "val_loss3":[],
                "test_ovs123_bag_acc":[], "test_ovs123_bag_kap":[], "test_ovs123_bag_f1":[], "test_o1vs23_bag_acc":[], "test_o1vs23_bag_kap":[], "test_o1vs23_bag_f1":[], "test_o12vs3_bag_acc":[], "test_o12vs3_bag_kap":[], "test_o12vs3_bag_f1":[],  "test_emsemble_bag_f1":[], "test_emsemble_bag_acc":[], "test_emsemble_bag_kap":[], }
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
        max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt, ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred = [], [], [], [], [], [], []
        losses, losses1, losses2, losses3 = [], [], [], []
        for iteration, data in enumerate(train_loader): 
            bags = data["bags"]
            max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = data["max_label"], data["0vs123_bag"], data["01vs23_bag"], data["012vs3_bag"]
            ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = data["ins_label"], data["0vs123_ins"], data["01vs23_ins"], data["012vs3_ins"]

            bags = bags.to(args.device)
            max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = max_label.to(args.device), ovs123_bag_label.to(args.device), o1vs23_bag_label.to(args.device), o12vs3_bag_label.to(args.device)
            ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = ins_labels.to(args.device), ovs123_ins_label.to(args.device), o1vs23_ins_label.to(args.device), o12vs3_ins_label.to(args.device)

            y = model(bags, data["len_list"])
            loss1, loss2, loss3 = loss_f_0vs123(y["y_0vs123"], ovs123_bag_label), loss_f_01vs23(y["y_01vs23"], o1vs23_bag_label), loss_f_012vs3(y["y_012vs3"], o12vs3_bag_label)
            loss = loss1 + loss2 + loss3 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            max_labels.extend(max_label.cpu().detach().numpy()), ovs123_bag_gt.extend(ovs123_bag_label.cpu().detach().numpy()), o1vs23_bag_gt.extend(o1vs23_bag_label.cpu().detach().numpy()), o12vs3_bag_gt.extend(o12vs3_bag_label.cpu().detach().numpy())
            ovs123_bag_pred.extend(y["y_0vs123"].argmax(1).cpu().detach().numpy()), o1vs23_bag_pred.extend(y["y_01vs23"].argmax(1).cpu().detach().numpy()), o12vs3_bag_pred.extend(y["y_012vs3"].argmax(1).cpu().detach().numpy())
            losses.append(loss.item()), losses1.append(loss1.item()), losses2.append(loss2.item()), losses3.append(loss3.item())

        max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt =  np.array(max_labels), np.array(ovs123_bag_gt), np.array(o1vs23_bag_gt), np.array(o12vs3_bag_gt)
        ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred =  np.array(ovs123_bag_pred), np.array(o1vs23_bag_pred), np.array(o12vs3_bag_pred)
        ovs123_bag_metric, o1vs23_bag_metric, o12vs3_bag_metric = calcurate_metrix(ovs123_bag_pred, ovs123_bag_gt), calcurate_metrix(o1vs23_bag_pred, o1vs23_bag_gt), calcurate_metrix(o12vs3_bag_pred, o12vs3_bag_gt)
        train_ovs123_bag_cm, train_o1vs23_bag_cm, train_o12vs3_bag_cm = ovs123_bag_metric["cm"], o1vs23_bag_metric["cm"], o12vs3_bag_metric["cm"]

        bag_emsemble_label = emsemble(ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred)
        emsemble_bag_metric = calcurate_metrix(bag_emsemble_label, max_labels)

        log_dict["train_ovs123_bag_acc"].append(ovs123_bag_metric["acc"]), log_dict["train_ovs123_bag_kap"].append(ovs123_bag_metric["kap"]), log_dict["train_ovs123_bag_f1"].append(ovs123_bag_metric["macro-f1"])
        log_dict["train_o1vs23_bag_acc"].append(o1vs23_bag_metric["acc"]), log_dict["train_o1vs23_bag_kap"].append(o1vs23_bag_metric["kap"]), log_dict["train_o1vs23_bag_f1"].append(o1vs23_bag_metric["macro-f1"])
        log_dict["train_o12vs3_bag_acc"].append(o12vs3_bag_metric["acc"]), log_dict["train_o12vs3_bag_kap"].append(o12vs3_bag_metric["kap"]), log_dict["train_o12vs3_bag_f1"].append(o12vs3_bag_metric["macro-f1"])
        log_dict["train_emsemble_bag_acc"].append(emsemble_bag_metric["acc"]), log_dict["train_emsemble_bag_kap"].append(emsemble_bag_metric["kap"]), log_dict["train_emsemble_bag_f1"].append(emsemble_bag_metric["macro-f1"])

        log_dict["train_loss"].append(np.array(losses).mean()), log_dict["train_loss1"].append(np.array(losses1).mean()), log_dict["train_loss2"].append(np.array(losses2).mean()), log_dict["train_loss3"].append(np.array(losses3).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, loss1: %.4f, loss2: %.4f, loss3: %.4f, @Bag acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_loss"][-1], log_dict["train_loss1"][-1], log_dict["train_loss2"][-1], log_dict["train_loss3"][-1], log_dict["train_emsemble_bag_acc"][-1], log_dict["train_emsemble_bag_kap"][-1], log_dict["train_emsemble_bag_f1"][-1]))
        
        ################# validation ####################
        s_time = time()
        model.eval()
        max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt, ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred = [], [], [], [], [], [], []
        losses, losses1, losses2, losses3 = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(val_loader): #enumerate(tqdm(val_loader, leave=False)):
                bags = data["bags"]
                max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = data["max_label"], data["0vs123_bag"], data["01vs23_bag"], data["012vs3_bag"]
                ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = data["ins_label"], data["0vs123_ins"], data["01vs23_ins"], data["012vs3_ins"]

                bags = bags.to(args.device)
                max_label, ovs123_bag_label, o1vs23_bag_label, o12vs3_bag_label = max_label.to(args.device), ovs123_bag_label.to(args.device), o1vs23_bag_label.to(args.device), o12vs3_bag_label.to(args.device)
                ins_labels, ovs123_ins_label, o1vs23_ins_label, o12vs3_ins_label = ins_labels.to(args.device), ovs123_ins_label.to(args.device), o1vs23_ins_label.to(args.device), o12vs3_ins_label.to(args.device)

                y = model(bags, data["len_list"])
                loss1, loss2, loss3 = loss_f_0vs123(y["y_0vs123"], ovs123_bag_label), loss_f_01vs23(y["y_01vs23"], o1vs23_bag_label), loss_f_012vs3(y["y_012vs3"], o12vs3_bag_label)
                loss = loss1 + loss2 + loss3 

                max_labels.extend(max_label.cpu().detach().numpy()), ovs123_bag_gt.extend(ovs123_bag_label.cpu().detach().numpy()), o1vs23_bag_gt.extend(o1vs23_bag_label.cpu().detach().numpy()), o12vs3_bag_gt.extend(o12vs3_bag_label.cpu().detach().numpy())
                ovs123_bag_pred.extend(y["y_0vs123"].argmax(1).cpu().detach().numpy()), o1vs23_bag_pred.extend(y["y_01vs23"].argmax(1).cpu().detach().numpy()), o12vs3_bag_pred.extend(y["y_012vs3"].argmax(1).cpu().detach().numpy())
                losses.append(loss.item()), losses1.append(loss1.item()), losses2.append(loss2.item()), losses3.append(loss3.item())

        max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt =  np.array(max_labels), np.array(ovs123_bag_gt), np.array(o1vs23_bag_gt), np.array(o12vs3_bag_gt)
        ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred =  np.array(ovs123_bag_pred), np.array(o1vs23_bag_pred), np.array(o12vs3_bag_pred)
        ovs123_bag_metric, o1vs23_bag_metric, o12vs3_bag_metric = calcurate_metrix(ovs123_bag_pred, ovs123_bag_gt), calcurate_metrix(o1vs23_bag_pred, o1vs23_bag_gt), calcurate_metrix(o12vs3_bag_pred, o12vs3_bag_gt)
        val_ovs123_bag_cm, val_o1vs23_bag_cm, val_o12vs3_bag_cm = ovs123_bag_metric["cm"], o1vs23_bag_metric["cm"], o12vs3_bag_metric["cm"]

        bag_emsemble_label = emsemble(ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred)
        emsemble_bag_metric = calcurate_metrix(bag_emsemble_label, max_labels)

        log_dict["val_ovs123_bag_acc"].append(ovs123_bag_metric["acc"]), log_dict["val_ovs123_bag_kap"].append(ovs123_bag_metric["kap"]), log_dict["val_ovs123_bag_f1"].append(ovs123_bag_metric["macro-f1"])
        log_dict["val_o1vs23_bag_acc"].append(o1vs23_bag_metric["acc"]), log_dict["val_o1vs23_bag_kap"].append(o1vs23_bag_metric["kap"]), log_dict["val_o1vs23_bag_f1"].append(o1vs23_bag_metric["macro-f1"])
        log_dict["val_o12vs3_bag_acc"].append(o12vs3_bag_metric["acc"]), log_dict["val_o12vs3_bag_kap"].append(o12vs3_bag_metric["kap"]), log_dict["val_o12vs3_bag_f1"].append(o12vs3_bag_metric["macro-f1"])
        log_dict["val_emsemble_bag_acc"].append(emsemble_bag_metric["acc"]), log_dict["val_emsemble_bag_kap"].append(emsemble_bag_metric["kap"]), log_dict["val_emsemble_bag_f1"].append(emsemble_bag_metric["macro-f1"])

        log_dict["val_loss"].append(np.array(losses).mean()), log_dict["val_loss1"].append(np.array(losses1).mean()), log_dict["val_loss2"].append(np.array(losses2).mean()), log_dict["val_loss3"].append(np.array(losses3).mean())

        e_time = time()
        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, loss1: %.4f, loss2: %.4f, loss3: %.4f, @Bag acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_loss"][-1], log_dict["val_loss1"][-1], log_dict["val_loss2"][-1], log_dict["val_loss3"][-1], log_dict["val_emsemble_bag_acc"][-1], log_dict["val_emsemble_bag_kap"][-1], log_dict["val_emsemble_bag_f1"][-1]))
        
        ################## test ###################
        s_time = time()
        model.eval()
        max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt, ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred = [], [], [], [], [], [], []
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

        max_labels, ovs123_bag_gt, o1vs23_bag_gt, o12vs3_bag_gt =  np.array(max_labels), np.array(ovs123_bag_gt), np.array(o1vs23_bag_gt), np.array(o12vs3_bag_gt)
        ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred =  np.array(ovs123_bag_pred), np.array(o1vs23_bag_pred), np.array(o12vs3_bag_pred)
        ovs123_bag_metric, o1vs23_bag_metric, o12vs3_bag_metric = calcurate_metrix(ovs123_bag_pred, ovs123_bag_gt), calcurate_metrix(o1vs23_bag_pred, o1vs23_bag_gt), calcurate_metrix(o12vs3_bag_pred, o12vs3_bag_gt)
        test_ovs123_bag_cm, test_o1vs23_bag_cm, test_o12vs3_bag_cm = ovs123_bag_metric["cm"], o1vs23_bag_metric["cm"], o12vs3_bag_metric["cm"]

        bag_emsemble_label = emsemble(ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred)
        emsemble_bag_metric = calcurate_metrix(bag_emsemble_label, max_labels)
        test_emsenble_bag_cm = emsemble_bag_metric["cm"]

        log_dict["test_ovs123_bag_acc"].append(ovs123_bag_metric["acc"]), log_dict["test_ovs123_bag_kap"].append(ovs123_bag_metric["kap"]), log_dict["test_ovs123_bag_f1"].append(ovs123_bag_metric["macro-f1"])
        log_dict["test_o1vs23_bag_acc"].append(o1vs23_bag_metric["acc"]), log_dict["test_o1vs23_bag_kap"].append(o1vs23_bag_metric["kap"]), log_dict["test_o1vs23_bag_f1"].append(o1vs23_bag_metric["macro-f1"])
        log_dict["test_o12vs3_bag_acc"].append(o12vs3_bag_metric["acc"]), log_dict["test_o12vs3_bag_kap"].append(o12vs3_bag_metric["kap"]), log_dict["test_o12vs3_bag_f1"].append(o12vs3_bag_metric["macro-f1"])
        log_dict["test_emsemble_bag_acc"].append(emsemble_bag_metric["acc"]), log_dict["test_emsemble_bag_kap"].append(emsemble_bag_metric["kap"]), log_dict["test_emsemble_bag_f1"].append(emsemble_bag_metric["macro-f1"])

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] @Test Bag emsemble acc: %.4f, kapp: %.4f, macro-f1: %.4f, @Bag kap: 0vs123: %.4f, 01vs23: %.4f, 012vs3: %.4f' %
                    (epoch+1, args.num_epochs, e_time-s_time, log_dict["test_emsemble_bag_acc"][-1], log_dict["test_emsemble_bag_kap"][-1], log_dict["test_emsemble_bag_f1"][-1], log_dict["test_ovs123_bag_kap"][-1], log_dict["test_o1vs23_bag_kap"][-1], log_dict["test_o12vs3_bag_kap"][-1]))
            
        logging.info('===============================')

        # if epoch%10 == 0:
        #     torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-epoch=%d_model.pkl") % (args.output_path, args.fold, args.seed, epoch))


        if best_val_kapp < log_dict["val_emsemble_bag_kap"][-1]:
            best_val_kapp = log_dict["val_emsemble_bag_kap"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed))
            if args.is_test == True:
                save_confusion_matrix(cm=test_ovs123_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_0vs123_bag.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_ovs123_bag_acc"][epoch], log_dict["test_ovs123_bag_kap"][epoch], log_dict["test_ovs123_bag_f1"][epoch]))
                save_confusion_matrix(cm=test_o1vs23_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_01vs23_bag.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_o1vs23_bag_acc"][epoch], log_dict["test_o1vs23_bag_kap"][epoch], log_dict["test_o1vs23_bag_f1"][epoch]))
                save_confusion_matrix(cm=test_o12vs3_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_012vs3_bag.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_o12vs3_bag_acc"][epoch], log_dict["test_o12vs3_bag_kap"][epoch], log_dict["test_o12vs3_bag_f1"][epoch])) 
                save_confusion_matrix(cm=test_emsenble_bag_cm, path=("%s/cm/fold=%d_seed=%d-cm_test_emsemble_bag.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, kapp: %.4f, macro-f1:n%.4f' % (epoch+1, log_dict["test_emsemble_bag_acc"][epoch], log_dict["test_emsemble_bag_kap"][epoch], log_dict["test_emsemble_bag_f1"][epoch])) 
        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('[Best Epoch: %d/%d (%ds)] @Val Bag emsemble acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                    (best_epoch+1, args.num_epochs, e_time-s_time, log_dict["val_emsemble_bag_acc"][best_epoch], log_dict["val_emsemble_bag_kap"][best_epoch], log_dict["val_emsemble_bag_f1"][best_epoch]))
        if args.is_test == True:
            logging.info('[Best Epoch: %d/%d (%ds)] @Test Bag emsemble acc: %.4f, kapp: %.4f, macro-f1: %.4f' %
                        (best_epoch+1, args.num_epochs, e_time-s_time, log_dict["test_emsemble_bag_acc"][best_epoch], log_dict["test_emsemble_bag_kap"][best_epoch], log_dict["test_emsemble_bag_f1"][best_epoch]))

        make_loss_graph(args,log_dict['train_loss'], log_dict['val_loss'], "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        make_loss_graph(args,log_dict['train_loss1'], log_dict['val_loss1'], "%s/loss_graph/fold=%d_seed=%d_loss1-graph.png" % (args.output_path, args.fold, args.seed))
        make_loss_graph(args,log_dict['train_loss2'], log_dict['val_loss2'], "%s/loss_graph/fold=%d_seed=%d_loss2-graph.png" % (args.output_path, args.fold, args.seed))
        make_loss_graph(args,log_dict['train_loss3'], log_dict['val_loss3'], "%s/loss_graph/fold=%d_seed=%d_loss3-graph.png" % (args.output_path, args.fold, args.seed))

        make_bag_acc_graph(args, log_dict['train_ovs123_bag_acc'], log_dict['val_ovs123_bag_acc'], log_dict['test_ovs123_bag_acc'], "%s/acc_graph/fold=%d_seed=%d_bag-0vs123-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_ovs123_bag_kap'], log_dict['val_ovs123_bag_kap'], log_dict['test_ovs123_bag_kap'], "%s/acc_graph/fold=%d_seed=%d_bag-0vs123-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_ovs123_bag_f1'], log_dict['val_ovs123_bag_f1'], log_dict['test_ovs123_bag_f1'], "%s/acc_graph/fold=%d_seed=%d_bag-0vs123-macrof1-graph.png" % (args.output_path, args.fold, args.seed))

        make_bag_acc_graph(args, log_dict['train_o1vs23_bag_acc'], log_dict['val_o1vs23_bag_acc'], log_dict['test_o1vs23_bag_acc'], "%s/acc_graph/fold=%d_seed=%d_bag-01vs23-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_o1vs23_bag_kap'], log_dict['val_o1vs23_bag_kap'], log_dict['test_o1vs23_bag_kap'], "%s/acc_graph/fold=%d_seed=%d_bag-01vs23-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_o1vs23_bag_f1'], log_dict['val_o1vs23_bag_f1'], log_dict['test_o1vs23_bag_f1'], "%s/acc_graph/fold=%d_seed=%d_bag-01vs23-macrof1-graph.png" % (args.output_path, args.fold, args.seed))

        make_bag_acc_graph(args, log_dict['train_o12vs3_bag_acc'], log_dict['val_o12vs3_bag_acc'], log_dict['test_o12vs3_bag_acc'], "%s/acc_graph/fold=%d_seed=%d_bag-012vs3-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_o12vs3_bag_kap'], log_dict['val_o12vs3_bag_kap'], log_dict['test_o12vs3_bag_kap'], "%s/acc_graph/fold=%d_seed=%d_bag-012vs3-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_o12vs3_bag_f1'], log_dict['val_o12vs3_bag_f1'], log_dict['test_o12vs3_bag_f1'], "%s/acc_graph/fold=%d_seed=%d_bag-012vs3-macrof1-graph.png" % (args.output_path, args.fold, args.seed))

        make_bag_acc_graph(args, log_dict['train_emsemble_bag_acc'], log_dict['val_emsemble_bag_acc'], log_dict['test_emsemble_bag_acc'], "%s/acc_graph/fold=%d_seed=%d_bag-emsemble-acc-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_emsemble_bag_kap'], log_dict['val_emsemble_bag_kap'], log_dict['test_emsemble_bag_kap'], "%s/acc_graph/fold=%d_seed=%d_bag-emsemble-kap-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_emsemble_bag_f1'], log_dict['val_emsemble_bag_f1'], log_dict['test_emsemble_bag_f1'], "%s/acc_graph/fold=%d_seed=%d_bag-emsemble-macrof1-graph.png" % (args.output_path, args.fold, args.seed))
        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return