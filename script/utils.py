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
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import glob


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def make_folder(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.exists(args.output_path + "/acc_graph"):
        os.mkdir(args.output_path + "/acc_graph")
    if not os.path.exists(args.output_path + "/cm"):
        os.mkdir(args.output_path + "/cm")
    if not os.path.exists(args.output_path + "/log_dict"):
        os.mkdir(args.output_path + "/log_dict")
    if not os.path.exists(args.output_path + "/loss_graph"):
        os.mkdir(args.output_path + "/loss_graph")
    if not os.path.exists(args.output_path + "/model"):
        os.mkdir(args.output_path + "/model")
    if not os.path.exists(args.output_path + "/test_metrics"):
        os.mkdir(args.output_path + "/test_metrics")
    return

def save_confusion_matrix(cm, path, title=''):
    plt.figure(figsize=(10, 8), dpi=300)
    cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', annot_kws={"size": 40})
    plt.xlabel('pred', fontsize=24)  # x軸ラベルの文字サイズを指定
    plt.ylabel('GT', fontsize=24)  # y軸ラベルの文字サイズを指定
    # sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.2f', annot_kws={"size": 36})
    # plt.xlabel('pred', fontsize=24)  # x軸ラベルの文字サイズを指定
    # plt.ylabel('GT', fontsize=24)  # y軸ラベルの文字サイズを指定
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def cal_OP_PC_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    OP = TP_c.sum() / (TP_c+FP_c).sum()
    PC = (TP_c/(TP_c+FP_c)).mean()
    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return OP, PC, mIoU


def cal_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return mIoU


def calcurate_metrix(preds, labels):
    acc = (np.array(preds) == np.array(labels)).sum() / len(preds)
    # f1 = f1_score(preds,labels, average="macro")
    # kap = cohen_kappa_score(preds, labels, weights="quadratic")
    # cm = confusion_matrix(preds, labels)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)
    kap = cohen_kappa_score(labels, preds, weights="quadratic")
    
    return {"acc": acc, "macro-f1": f1, "kap": kap, "cm": cm}

    
def make_loss_graph(args, keep_train_loss, keep_valid_loss, path):
    #loss graph save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(keep_train_loss, label = 'train')
    ax.plot(keep_valid_loss, label = 'valid')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("Losses")
    plt.legend()
    fig.savefig(path)
    plt.close() 
    return

def make_bag_acc_graph(args, train_bag_acc, val_bag_acc, test_bag_acc, path):
    #Bag level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_bag_acc, label = 'train bag acc')
    ax.plot(val_bag_acc, label = 'valid bag acc')
    ax.plot(test_bag_acc, label = 'test bag acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return

def make_ins_acc_graph(args, train_ins_acc, val_ins_acc, test_ins_acc, path):
    #instance level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_ins_acc, label = 'train instance acc')
    ax.plot(val_ins_acc, label = 'valid instans acc')
    ax.plot(test_ins_acc, label = 'test ins acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return


def make_PCA(args, ins_feature, ins_label, bag_feature, major_label):
    #PCA
    data = torch.cat([ins_feature, bag_feature], dim=0)
    data = PCA(n_components=2).fit_transform(data.cpu().clone().detach())
    #instance feature
    # ins_data2d = PCA(n_components=2).fit_transform(ins_feature.cpu().clone().detach())
    ins_data2d = data[:6400]
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(ins_label.max()+1):
        target = ins_data2d[ins_label == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=1, fontsize=14)
    plt.xlim(ins_data2d[:, 0].min(), ins_data2d[:, 0].max())
    plt.ylim(ins_data2d[:, 1].min(), ins_data2d[:, 1].max())
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "PCA_ins_feature_fold=" + str(args.fold)  + ".png"), dpi=400, bbox_inches="tight")
    plt.close()

    #bag feature
    # bag_data2d = PCA(n_components=2).fit_transform(bag_feature.cpu().clone().detach())
    bag_data2d = data[6400:]
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(major_label.max()+1):
        target = bag_data2d[major_label == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=1, fontsize=14)
    plt.xlim(ins_data2d[:, 0].min(), ins_data2d[:, 0].max())
    plt.ylim(ins_data2d[:, 1].min(), ins_data2d[:, 1].max())
    
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "PCA_bag_feature_fold=" + str(args.fold) + ".png"), dpi=400, bbox_inches="tight")
    plt.close()

# #tsne
    data = torch.cat([ins_feature, bag_feature], dim=0)
    data = TSNE(n_components=2).fit_transform(data.cpu().clone().detach())
    #instance feature
    # ins_data2d = PCA(n_components=2).fit_transform(ins_feature.cpu().clone().detach())
    ins_data2d = data[:6400]
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(ins_label.max()+1):
        target = ins_data2d[ins_label == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=1, fontsize=14)
    plt.xlim(ins_data2d[:, 0].min(), ins_data2d[:, 0].max())
    plt.ylim(ins_data2d[:, 1].min(), ins_data2d[:, 1].max())
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "TSNE_ins_feature_fold=" + str(args.fold) + ".png"), dpi=400, bbox_inches="tight")
    plt.close()

    #bag feature
    # bag_data2d = PCA(n_components=2).fit_transform(bag_feature.cpu().clone().detach())
    bag_data2d = data[6400:]
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(major_label.max()+1):
        target = bag_data2d[major_label == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=1, fontsize=14)
    plt.xlim(ins_data2d[:, 0].min(), ins_data2d[:, 0].max())
    plt.ylim(ins_data2d[:, 1].min(), ins_data2d[:, 1].max())
    
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "TSNE_bag_feature_fold=" + str(args.fold)  + ".png"), dpi=400, bbox_inches="tight")
    plt.close()
    return


def feature_visual_compare_train_val(args, train_ins_feature, train_ins_gt, val_ins_feature, val_ins_gt,epoch):
    if epoch%20 == 0:
    #PCA
        data = torch.cat([train_ins_feature, val_ins_feature], dim=0)
        data = PCA(n_components=2).fit_transform(data.cpu().clone().detach())
        #train instance feature
        train_data2d = data[:len(train_ins_feature)]
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(train_ins_gt.max()+1):
            target = train_data2d[train_ins_gt == i]
            ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=1, fontsize=14)
        plt.xlim(train_data2d[:, 0].min(), train_data2d[:, 0].max())
        plt.ylim(train_data2d[:, 1].min(), train_data2d[:, 1].max())
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "PCA_T_V_compare_T_fold=" + str(args.fold) + "_epoch=" + str(epoch) + ".png"), dpi=400, bbox_inches="tight")
        plt.close()

        #validation instance feature
        validation_data2d = data[len(train_ins_feature):]
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(train_ins_gt.max()+1):
            target = validation_data2d[val_ins_gt == i]
            ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=1, fontsize=14)
        plt.xlim(train_data2d[:, 0].min(), train_data2d[:, 0].max())
        plt.ylim(train_data2d[:, 1].min(), train_data2d[:, 1].max())
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "PCA_T_V_compare_V_fold=" + str(args.fold) + "_epoch=" + str(epoch) + ".png"), dpi=400, bbox_inches="tight")
        plt.close()

    # #tsne
        data = torch.cat([train_ins_feature, val_ins_feature], dim=0)
        data = TSNE(n_components=2).fit_transform(data.cpu().clone().detach())
        #train instance feature
        train_data2d = data[:len(train_ins_feature)]
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(train_ins_gt.max()+1):
            target = train_data2d[train_ins_gt == i]
            ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=1, fontsize=14)
        plt.xlim(train_data2d[:, 0].min(), train_data2d[:, 0].max())
        plt.ylim(train_data2d[:, 1].min(), train_data2d[:, 1].max())
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "TSNE_T_V_compare_T_fold=" + str(args.fold) + "_epoch=" + str(epoch) + ".png"), dpi=400, bbox_inches="tight")
        plt.close()

        #validation instance feature
        val_data2d = data[len(train_ins_feature):]
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(train_ins_gt.max()+1):
            target = val_data2d[val_ins_gt == i]
            ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=1, fontsize=14)
        plt.xlim(train_data2d[:, 0].min(), train_data2d[:, 0].max())
        plt.ylim(train_data2d[:, 1].min(), train_data2d[:, 1].max())
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.savefig((args.output_path+ "/feature_look/"+args.mode+"/" + "TSNE_T_V_compare_V_fold=" + str(args.fold) + "_epoch=" + str(epoch) + ".png"), dpi=400, bbox_inches="tight")
        plt.close()
    return

def emsemble(ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred):
    # baglabel_idx, bag_emsemble_label = np.arange(len(o12vs3_bag_pred)), np.zeros(len(o12vs3_bag_pred))
    # baglabel_0_idx = baglabel_idx[ovs123_bag_pred==0]  
    # baglabel_1_idx = baglabel_idx[(ovs123_bag_pred==1) * (o1vs23_bag_pred==0)]
    # baglabel_2_idx = baglabel_idx[(o1vs23_bag_pred==1) * (o12vs3_bag_pred==0)]
    # baglabel_3_idx = baglabel_idx[o12vs3_bag_pred==1]

    # bag_emsemble_label[baglabel_0_idx]=0
    # bag_emsemble_label[baglabel_1_idx]=1
    # bag_emsemble_label[baglabel_2_idx]=2
    # bag_emsemble_label[baglabel_3_idx]=3

    bag_emsemble_label = ovs123_bag_pred + o1vs23_bag_pred + o12vs3_bag_pred
    return bag_emsemble_label

def output_krank_emsemble(ovs123_ins_pred, o1vs23_ins_pred, o12vs3_ins_pred, len_list,  emsemble_mode):
    if emsemble_mode == "convert":
        p_0, p_1, p_2, p_3 = 1-ovs123_ins_pred, ovs123_ins_pred-o1vs23_ins_pred, o1vs23_ins_pred-o12vs3_ins_pred,  o12vs3_ins_pred
        p = np.column_stack((p_0, p_1, p_2, p_3))
        p=np.argmax(p, axis=1)
        slice_ini=0
        bag_preds=[]
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            bag_pred = p[slice_ini:(slice_ini+bag_len)].max()
            bag_preds.append(bag_pred)
            slice_ini += bag_len

    elif emsemble_mode == "threshold":
        pred_label_1 = (ovs123_ins_pred>0.2)*1
        pred_label_2 = (o1vs23_ins_pred>0.19)*1
        pred_label_3 = (o12vs3_ins_pred>0.22)*1
        p = pred_label_1+pred_label_2+pred_label_3
        slice_ini=0
        bag_preds=[]
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            bag_pred = p[slice_ini:(slice_ini+bag_len)].max()
            bag_preds.append(bag_pred)
            slice_ini += bag_len
            
    elif emsemble_mode == "sum":
        ins_pred = ovs123_ins_pred+o1vs23_ins_pred+o12vs3_ins_pred
        slice_ini=0
        bag_preds=[]
        for idx, bag_len in enumerate(len_list):            # loop at all bag
            bag_pred = ins_pred[slice_ini:(slice_ini+bag_len)].max()
            if bag_pred<0.1:
                bag_pred = 0
            elif bag_pred>=0.1 and bag_pred<1.04:
                bag_pred = 1
            elif bag_pred>=1.04 and bag_pred<2.02:
                bag_pred = 2
            elif bag_pred>=2.02:
                bag_pred = 3
            bag_preds.append(bag_pred)
            slice_ini += bag_len

    return np.array(bag_preds)

# def ORlabel_translation(pred_lablel):
#     pred_lablel_idx, translatd_lablel_idx = np.arange(len(pred_lablel)), np.arange(len(pred_lablel))
#     pred_label_0_idx = pred_lablel_idx[pred_lablel<=1.03]
#     pred_label_1_idx = pred_lablel_idx[(pred_lablel>1.03) * (pred_lablel<=1.97)]
#     pred_label_2_idx = pred_lablel_idx[(pred_lablel>1.97) * (pred_lablel<=2.79)]
#     pred_label_3_idx = pred_lablel_idx[pred_lablel>2.79]

#     translatd_lablel_idx[pred_label_0_idx] = 0
#     translatd_lablel_idx[pred_label_1_idx] = 1
#     translatd_lablel_idx[pred_label_2_idx] = 2
#     translatd_lablel_idx[pred_label_3_idx] = 3
#     return translatd_lablel_idx
#     return bag_emsemble_label

def ORlabel_translation(pred_lablel):
    pred_lablel_idx, translatd_lablel_idx = np.arange(len(pred_lablel)), np.arange(len(pred_lablel))
    # pred_label_0_idx = pred_lablel_idx[pred_lablel<=1.03]
    # pred_label_1_idx = pred_lablel_idx[(pred_lablel>1.03) * (pred_lablel<=1.97)]
    # pred_label_2_idx = pred_lablel_idx[(pred_lablel>1.97) * (pred_lablel<=2.79)]
    # pred_label_3_idx = pred_lablel_idx[pred_lablel>2.79]

    pred_label_0_idx = pred_lablel_idx[pred_lablel<=0.5]
    pred_label_1_idx = pred_lablel_idx[(pred_lablel>0.5) * (pred_lablel<=1.50)]
    pred_label_2_idx = pred_lablel_idx[(pred_lablel>1.50) * (pred_lablel<=2.50)]
    pred_label_3_idx = pred_lablel_idx[pred_lablel>2.50]

    translatd_lablel_idx[pred_label_0_idx] = 0
    translatd_lablel_idx[pred_label_1_idx] = 1
    translatd_lablel_idx[pred_label_2_idx] = 2
    translatd_lablel_idx[pred_label_3_idx] = 3
    return translatd_lablel_idx

def general_OvsR_emsemble(bag_pred_0vs123, bag_pred_1vs023, bag_pred_2vs013, bag_pred_3vs012):
    baglabel_idx, bag_emsemble_label = np.arange(len(bag_pred_0vs123)), np.zeros(len(bag_pred_0vs123))
    baglabel_0_idx = baglabel_idx[bag_pred_0vs123==0]  
    baglabel_1_idx = baglabel_idx[(bag_pred_0vs123==1) * (bag_pred_1vs023==0)]
    baglabel_2_idx = baglabel_idx[((bag_pred_0vs123==1) * (bag_pred_1vs023==1)) * (bag_pred_2vs013==0)]
    baglabel_3_idx = baglabel_idx[((bag_pred_0vs123==1) * (bag_pred_1vs023==1) )* (bag_pred_2vs013==1)]

    bag_emsemble_label[baglabel_0_idx]=0
    bag_emsemble_label[baglabel_1_idx]=1
    bag_emsemble_label[baglabel_2_idx]=2
    bag_emsemble_label[baglabel_3_idx]=3

    return bag_emsemble_label

def multiemsemble(ovs123_bag_pred, o1vs23_bag_pred, o12vs3_bag_pred):
    baglabel_idx, bag_emsemble_label = np.arange(len(o12vs3_bag_pred)), np.zeros(len(o12vs3_bag_pred))
    baglabel_0_idx = baglabel_idx[ovs123_bag_pred==0]  
    baglabel_1_idx = baglabel_idx[(ovs123_bag_pred==1) * (o1vs23_bag_pred==0)]
    baglabel_2_idx = baglabel_idx[(o1vs23_bag_pred==1) * (o12vs3_bag_pred==0)]
    baglabel_3_idx = baglabel_idx[o12vs3_bag_pred==1]

    bag_emsemble_label[baglabel_0_idx]=0
    bag_emsemble_label[baglabel_1_idx]=1
    bag_emsemble_label[baglabel_2_idx]=2
    bag_emsemble_label[baglabel_3_idx]=3

    return bag_emsemble_label