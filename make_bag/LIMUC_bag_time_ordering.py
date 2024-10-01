# crossvalidationで分けられているBagを読み込んで時系列順に並び変え
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

def time_ordering(bags, ins_labels, txt_save_path):
    f = open((txt_save_path), 'w')

    ordered_bags, ordered_ins_labels = [], []
    for bag_idx in range(len(bags)):
        ins_paths = np.load(bags[bag_idx]) 
        ins_label = ins_labels[bag_idx]

        time_label = []
        for ins_idx in range(len(ins_paths)):
            ins_path = ins_paths[ins_idx]
            time = ins_path.split("_")[-1]
            time = time.split(".")[0]
            time_label.append(int(time))
        
        time_order_idx = np.argsort(time_label)  # 時系列で並べ替え
        ordered_ins_paths = ins_paths[time_order_idx]
        ordered_ins_label = ins_label[time_order_idx]
        ordered_bags.append(ordered_ins_paths), ordered_ins_labels.append(ordered_ins_label)

        f.write('$ bag_idx=%d \n' % (bag_idx))
        for i in range(len(ordered_ins_paths)):
            f.write("%s \n" % ordered_ins_paths[i])
    f.close()
    return np.array(ordered_bags), np.array(ordered_ins_labels)

def main(args):
    make_folder(args)

    for fold in range(5):
        train_bag_labels = np.load("%s/%d/train_bag_labels.npy" % (args.data_path, fold))
        train_ins_labels = np.load("%s/%d/train_ins_labels.npy" % (args.data_path, fold), allow_pickle=True)
        train_bags = np.load("%s/%d/train_bags.npy" % (args.data_path, fold), allow_pickle=True)
        time_ordered_train_bags, time_ordered_train_ins_labels = time_ordering(train_bags, train_ins_labels, txt_save_path="%s/%d/train_bags_time_order.txt" % (args.output_path, fold))
        np.save('%s/%d/train_bags' % (args.output_path, fold), time_ordered_train_bags)
        np.save('%s/%d/train_ins_labels' % (args.output_path, fold), time_ordered_train_ins_labels)
        np.save('%s%d//train_bag_labels' % (args.output_path, fold), train_bag_labels)

        val_bag_labels = np.load("%s/%d/val_bag_labels.npy" % (args.data_path, fold))
        val_ins_labels = np.load("%s/%d/val_ins_labels.npy" % (args.data_path, fold), allow_pickle=True)
        val_bags = np.load("%s/%d/val_bags.npy" % (args.data_path, fold), allow_pickle=True)
        time_ordered_val_bags, time_ordered_val_ins_labels = time_ordering(val_bags, val_ins_labels, txt_save_path="%s/%d/val_bags_time_order.txt" % (args.output_path, fold))
        np.save('%s/%d/val_bags' % (args.output_path, fold), time_ordered_val_bags)
        np.save('%s/%d/val_ins_labels' % (args.output_path, fold), time_ordered_val_ins_labels)
        np.save('%s/%d/val_bag_labels' % (args.output_path, fold), val_bag_labels)

        test_bag_labels = np.load("%s/%d/test_bag_labels.npy" % (args.data_path, fold))
        test_ins_labels = np.load("%s/%d/test_ins_labels.npy" % (args.data_path, fold), allow_pickle=True)
        test_bags = np.load("%s/%d/test_bags.npy" % (args.data_path, fold), allow_pickle=True)
        time_ordered_test_bags, time_ordered_test_ins_labels = time_ordering(test_bags, test_ins_labels, txt_save_path="%s/%d/test_bags_time_order.txt" % (args.output_path, fold))
        np.save('%s/%d/test_bags' % (args.output_path, fold), time_ordered_test_bags)
        np.save('%s/%d/test_ins_labels' % (args.output_path, fold), time_ordered_test_ins_labels)
        np.save('%s/%d/test_bag_labels' % (args.output_path, fold), test_bag_labels)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fold_num', default=5, type=int)

    parser.add_argument('--dataset', default='LIMUC', type=str)
    parser.add_argument('--data_path', default='./bag_data/LIMUC/5-fold_in_test_balanced/', type=str)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--output_path', default='./bag_data/LIMUC/5-fold_in_test_balanced_time_order/' , type=str)
    args = parser.parse_args()

    main(args)