import argparse
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from glob import glob
from PIL import Image
from tqdm import tqdm
import os 
from utils import *
import random

def make_folder(args):
    for fold in range(args.fold_num):
        if not os.path.exists("%s/%d" % (args.output_path, fold)):
            os.mkdir("%s/%d" % (args.output_path, fold))
    return

def test_in_5fold_balanced_main(args):
    make_folder(args)
    c0_path_bags, c0_ins_labels_path, c0_bag_labels_path = np.array(sorted(glob(("%s/0/bags/*.npy" % args.data_path)))), np.array(sorted(glob("%s/0/ins_labels/*.npy" % args.data_path))), np.array(sorted(glob("%s/0/bag_labels/*.npy" % args.data_path)))
    c1_path_bags, c1_ins_labels_path, c1_bag_labels_path = np.array(sorted(glob("%s/1/bags/*.npy" % args.data_path))), np.array(sorted(glob("%s/1/ins_labels/*.npy" % args.data_path))), np.array(sorted(glob("%s/1/bag_labels/*.npy" % args.data_path)))
    c2_path_bags, c2_ins_labels_path, c2_bag_labels_path = np.array(sorted(glob("%s/2/bags/*.npy" % args.data_path))), np.array(sorted(glob("%s/2/ins_labels/*.npy" % args.data_path))), np.array(sorted(glob("%s/2/bag_labels/*.npy" % args.data_path)))
    c3_path_bags, c3_ins_labels_path, c3_bag_labels_path = np.array(sorted(glob("%s/3/bags/*.npy" % args.data_path))), np.array(sorted(glob("%s/3/ins_labels/*.npy" % args.data_path))), np.array(sorted(glob("%s/3/bag_labels/*.npy" % args.data_path)))

    idx = np.arange(len(c0_path_bags))
    np.random.shuffle(idx)
    c0_path_bags, c0_ins_labels_path, c0_bag_labels_path = c0_path_bags[idx], c0_ins_labels_path[idx], c0_bag_labels_path[idx]
    c0_ins_labels, c0_bag_labels = [], []
    for id in range(len(c0_bag_labels_path)):
        c0_ins_labels.append(np.load(c0_ins_labels_path[id])), c0_bag_labels.append(np.load(c0_bag_labels_path[id]))
    split_len = int(len(c0_path_bags)/args.fold_num)
    cls0_fold_dict = {'bag0': c0_path_bags[0:split_len] ,'ins_label0': c0_ins_labels[0:split_len], "bag_label0": c0_bag_labels[0:split_len], 
                 'bag1': c0_path_bags[split_len:(split_len*2)] ,'ins_label1': c0_ins_labels[split_len:(split_len*2)], "bag_label1": c0_bag_labels[split_len:(split_len*2)],
                 'bag2': c0_path_bags[(split_len*2):(split_len*3)] ,'ins_label2': c0_ins_labels[(split_len*2):(split_len*3)], "bag_label2": c0_bag_labels[(split_len*2):(split_len*3)],
                 'bag3': c0_path_bags[(split_len*3):(split_len*4)] ,'ins_label3': c0_ins_labels[(split_len*3):(split_len*4)], "bag_label3": c0_bag_labels[(split_len*3):(split_len*4)],
                 'bag4': c0_path_bags[(split_len*4):] ,'ins_label4': c0_ins_labels[(split_len*4):], "bag_label4": c0_bag_labels[(split_len*4):]}
    
    idx = np.arange(len(c1_path_bags))
    np.random.shuffle(idx)
    c1_path_bags, c1_ins_labels_path, c1_bag_labels_path = c1_path_bags[idx], c1_ins_labels_path[idx], c1_bag_labels_path[idx]
    c1_ins_labels, c1_bag_labels = [], []
    for id in range(len(c1_bag_labels_path)):
        c1_ins_labels.append(np.load(c1_ins_labels_path[id])), c1_bag_labels.append(np.load(c1_bag_labels_path[id]))
    split_len = int(len(c1_path_bags)/args.fold_num)
    cls1_fold_dict = {'bag0': c1_path_bags[0:split_len] ,'ins_label0': c1_ins_labels[0:split_len], "bag_label0": c1_bag_labels[0:split_len], 
                 'bag1': c1_path_bags[split_len:(split_len*2)] ,'ins_label1': c1_ins_labels[split_len:(split_len*2)], "bag_label1": c1_bag_labels[split_len:(split_len*2)],
                 'bag2': c1_path_bags[(split_len*2):(split_len*3)] ,'ins_label2': c1_ins_labels[(split_len*2):(split_len*3)], "bag_label2": c1_bag_labels[(split_len*2):(split_len*3)],
                 'bag3': c1_path_bags[(split_len*3):(split_len*4)] ,'ins_label3': c1_ins_labels[(split_len*3):(split_len*4)], "bag_label3": c1_bag_labels[(split_len*3):(split_len*4)],
                 'bag4': c1_path_bags[(split_len*4):] ,'ins_label4': c1_ins_labels[(split_len*4):], "bag_label4": c1_bag_labels[(split_len*4):]}

    idx = np.arange(len(c2_path_bags))
    np.random.shuffle(idx)
    c2_path_bags, c2_ins_labels_path, c2_bag_labels_path = c2_path_bags[idx], c2_ins_labels_path[idx], c2_bag_labels_path[idx]
    c2_ins_labels, c2_bag_labels = [], []
    for id in range(len(c2_bag_labels_path)):
        c2_ins_labels.append(np.load(c2_ins_labels_path[id])), c2_bag_labels.append(np.load(c2_bag_labels_path[id]))
    split_len = int(len(c2_path_bags)/args.fold_num)
    cls2_fold_dict = {'bag0': c2_path_bags[0:split_len] ,'ins_label0': c2_ins_labels[0:split_len], "bag_label0": c2_bag_labels[0:split_len], 
                 'bag1': c2_path_bags[split_len:(split_len*2)] ,'ins_label1': c2_ins_labels[split_len:(split_len*2)], "bag_label1": c2_bag_labels[split_len:(split_len*2)],
                 'bag2': c2_path_bags[(split_len*2):(split_len*3)] ,'ins_label2': c2_ins_labels[(split_len*2):(split_len*3)], "bag_label2": c2_bag_labels[(split_len*2):(split_len*3)],
                 'bag3': c2_path_bags[(split_len*3):(split_len*4)] ,'ins_label3': c2_ins_labels[(split_len*3):(split_len*4)], "bag_label3": c2_bag_labels[(split_len*3):(split_len*4)],
                 'bag4': c2_path_bags[(split_len*4):] ,'ins_label4': c2_ins_labels[(split_len*4):], "bag_label4": c2_bag_labels[(split_len*4):]
                 }

    idx = np.arange(len(c3_path_bags))
    np.random.shuffle(idx)
    c3_path_bags, c3_ins_labels_path, c3_bag_labels_path = c3_path_bags[idx], c3_ins_labels_path[idx], c3_bag_labels_path[idx]
    c3_ins_labels, c3_bag_labels = [], []
    for id in range(len(c3_bag_labels_path)):
        c3_ins_labels.append(np.load(c3_ins_labels_path[id])), c3_bag_labels.append(np.load(c3_bag_labels_path[id]))
    split_len = int(len(c3_path_bags)/args.fold_num)
    cls3_fold_dict = {'bag0': c3_path_bags[0:split_len] ,'ins_label0': c3_ins_labels[0:split_len], "bag_label0": c3_bag_labels[0:split_len], 
                 'bag1': c3_path_bags[split_len:(split_len*2)] ,'ins_label1': c3_ins_labels[split_len:(split_len*2)], "bag_label1": c3_bag_labels[split_len:(split_len*2)],
                 'bag2': c3_path_bags[(split_len*2):(split_len*3)] ,'ins_label2': c3_ins_labels[(split_len*2):(split_len*3)], "bag_label2": c3_bag_labels[(split_len*2):(split_len*3)],
                 'bag3': c3_path_bags[(split_len*3):(split_len*4)] ,'ins_label3': c3_ins_labels[(split_len*3):(split_len*4)], "bag_label3": c3_bag_labels[(split_len*3):(split_len*4)],
                 'bag4': c3_path_bags[(split_len*4):] ,'ins_label4': c3_ins_labels[(split_len*4):], "bag_label4": c3_bag_labels[(split_len*4):]
                 }

    for i in range(args.fold_num):
        output_path = "%s/%d/" % (args.output_path, i)
        test_bags, test_ins_label, test_bag_label = np.concatenate((cls0_fold_dict['bag%d'%((i)%5)], cls1_fold_dict['bag%d'%((i)%5)], cls2_fold_dict['bag%d'%((i)%5)], cls3_fold_dict['bag%d'%((i)%5)])), np.concatenate((cls0_fold_dict['ins_label%d'%((i)%5)], cls1_fold_dict['ins_label%d'%((i)%5)], cls2_fold_dict['ins_label%d'%((i)%5)], cls3_fold_dict['ins_label%d'%((i)%5)])), np.concatenate((cls0_fold_dict['bag_label%d'%((i)%5)], cls1_fold_dict['bag_label%d'%((i)%5)], cls2_fold_dict['bag_label%d'%((i)%5)], cls3_fold_dict['bag_label%d'%((i)%5)]))
        val_bags, val_ins_label, val_bag_label = np.concatenate((cls0_fold_dict['bag%d'%((1+i)%5)], cls1_fold_dict['bag%d'%((1+i)%5)], cls2_fold_dict['bag%d'%((1+i)%5)], cls3_fold_dict['bag%d'%((1+i)%5)])), np.concatenate((cls0_fold_dict['ins_label%d'%((1+i)%5)], cls1_fold_dict['ins_label%d'%((1+i)%5)], cls2_fold_dict['ins_label%d'%((1+i)%5)], cls3_fold_dict['ins_label%d'%((1+i)%5)])), np.concatenate((cls0_fold_dict['bag_label%d'%((1+i)%5)], cls1_fold_dict['bag_label%d'%((1+i)%5)], cls2_fold_dict['bag_label%d'%((1+i)%5)], cls3_fold_dict['bag_label%d'%((1+i)%5)]))
        train_bags = np.concatenate((cls0_fold_dict['bag%d'%((2+i)%5)], cls0_fold_dict['bag%d'%((3+i)%5)], cls0_fold_dict['bag%d'%((4+i)%5)], cls1_fold_dict['bag%d'%((2+i)%5)], cls1_fold_dict['bag%d'%((3+i)%5)], cls1_fold_dict['bag%d'%((4+i)%5)], cls2_fold_dict['bag%d'%((2+i)%5)], cls2_fold_dict['bag%d'%((3+i)%5)], cls2_fold_dict['bag%d'%((4+i)%5)], cls3_fold_dict['bag%d'%((2+i)%5)], cls3_fold_dict['bag%d'%((3+i)%5)], cls3_fold_dict['bag%d'%((4+i)%5)]))
        train_ins_label = np.concatenate((cls0_fold_dict['ins_label%d'%((2+i)%5)], cls0_fold_dict['ins_label%d'%((3+i)%5)], cls0_fold_dict['ins_label%d'%((4+i)%5)], cls1_fold_dict['ins_label%d'%((2+i)%5)], cls1_fold_dict['ins_label%d'%((3+i)%5)], cls1_fold_dict['ins_label%d'%((4+i)%5)], cls2_fold_dict['ins_label%d'%((2+i)%5)], cls2_fold_dict['ins_label%d'%((3+i)%5)], cls2_fold_dict['ins_label%d'%((4+i)%5)], cls3_fold_dict['ins_label%d'%((2+i)%5)], cls3_fold_dict['ins_label%d'%((3+i)%5)], cls3_fold_dict['ins_label%d'%((4+i)%5)]))
        train_bag_label = np.concatenate((cls0_fold_dict['bag_label%d'%((2+i)%5)], cls0_fold_dict['bag_label%d'%((3+i)%5)], cls0_fold_dict['bag_label%d'%((4+i)%5)], cls1_fold_dict['bag_label%d'%((2+i)%5)], cls1_fold_dict['bag_label%d'%((3+i)%5)], cls1_fold_dict['bag_label%d'%((4+i)%5)], cls2_fold_dict['bag_label%d'%((2+i)%5)], cls2_fold_dict['bag_label%d'%((3+i)%5)], cls2_fold_dict['bag_label%d'%((4+i)%5)], cls3_fold_dict['bag_label%d'%((2+i)%5)], cls3_fold_dict['bag_label%d'%((3+i)%5)], cls3_fold_dict['bag_label%d'%((4+i)%5)]))
        f = open((output_path + 'bag_info.txt'), 'w')

        idx = np.arange(len(test_bags))
        np.random.shuffle(idx)
        test_bags, test_ins_label, test_bag_label = test_bags[idx], test_ins_label[idx], test_bag_label[idx]

        idx = np.arange(len(val_bags))
        np.random.shuffle(idx)
        val_bags, val_ins_label, val_bag_label = val_bags[idx], val_ins_label[idx], val_bag_label[idx]

        idx = np.arange(len(train_bag_label))
        np.random.shuffle(idx)
        train_bags, train_ins_label, train_bag_label = train_bags[idx], train_ins_label[idx], train_bag_label[idx]

        np.save('%s/train_bags' % (output_path), train_bags), np.save('%s/train_ins_labels' % (output_path), train_ins_label), np.save('%s/train_bag_labels' % (output_path), train_bag_label)
        np.save('%s/val_bags' % (output_path), val_bags), np.save('%s/val_ins_labels' % (output_path), val_ins_label), np.save('%s/val_bag_labels' % (output_path), val_bag_label)
        np.save('%s/test_bags' % (output_path), test_bags), np.save('%s/test_ins_labels' % (output_path), test_ins_label),  np.save('%s/test_bag_labels' % (output_path), test_bag_label)

        f.write('$ Train data \n bag num , Bag_class_num: mayo0=%d, mayo1=%d, mayo2=%d, mayo3=%d \n' % (np.sum(train_bag_label==0), np.sum(train_bag_label==1), np.sum(train_bag_label==2), np.sum(train_bag_label==3)))
        f.write('$ Validation data \n bag num , Bag_class_num: mayo0=%d, mayo1=%d, mayo2=%d, mayo3=%d \n' % (np.sum(val_bag_label==0), np.sum(val_bag_label==1), np.sum(val_bag_label==2), np.sum(val_bag_label==3)))
        f.write('$ Test data \n bag num , Bag_class_num: mayo0=%d, mayo1=%d, mayo2=%d, mayo3=%d \n' % (np.sum(test_bag_label==0), np.sum(test_bag_label==1), np.sum(test_bag_label==2), np.sum(test_bag_label==3)))
        f.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--dataset', default='LIMUC', type=str)
    parser.add_argument('--data_path', default='./bag_data/LIMUC/bags_per_class', type=str)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--data_type', default='5-fold_in_test_balanced' , type=str)
    parser.add_argument('--output_path', default='./bag_data/LIMUC/' , type=str)
    parser.add_argument('--fold_num', default=5, type=int)

    args = parser.parse_args()

    args.output_path += args.data_type
    test_in_5fold_balanced_main(args)