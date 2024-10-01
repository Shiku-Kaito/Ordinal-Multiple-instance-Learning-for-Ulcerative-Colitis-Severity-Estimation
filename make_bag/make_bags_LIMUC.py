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
    for c in range(args.num_classes):
        if not os.path.exists("%s/bags_per_class/%d" % (args.output_path, c)):
            os.mkdir("%s/bags_per_class/%d" % (args.output_path, c))
        if not os.path.exists("%s/bags_per_class/%d/bags" % ((args.output_path, c))):
            os.mkdir("%s/bags_per_class/%d/bags" % ((args.output_path, c)))
        if not os.path.exists("%s/bags_per_class/%d/ins_labels" % ((args.output_path, c))):
            os.mkdir("%s/bags_per_class/%d/ins_labels" % ((args.output_path, c)))
        if not os.path.exists("%s/bags_per_class/%d/bag_labels" % ((args.output_path, c))):
            os.mkdir("%s/bags_per_class/%d/bag_labels" % ((args.output_path, c)))
    return

def analysis_def(args, inst_class_num, bag_size):
    make_histgram(data_list=bag_size, path="%s/data_analysis/bag_size_hist/hist.png" % args.output_path, x_max=150, x_min=0, x="bag size", y="bag num", normalize=False)
    for bag_id in inst_class_num.keys():
        ins_c = []
        for c in range(args.num_classes):
            ins_c.extend([c]*(inst_class_num[bag_id][c]))
        make_histgram(data_list=ins_c, path="%s/data_analysis/ins_distributions/%d.png" % (args.output_path, bag_id), x_max=4, x_min=0, x="class", y="instance num", normalize=False)
    return

def main(args):
    make_folder(args)
    bag_size, all_bag_class_num, inst_class_num = [], [0 for _ in range(args.num_classes)], {}
    patients_folders = [f for f in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, f))]
    patients_folders = sorted(list(map(int, patients_folders)))
    all_bags, all_ins_labels, all_bag_labels = [], [], []

    for patient_id, folder in enumerate(patients_folders):
        bag, ins_labels, bag_labels = [], [], []
        class_num = [0 for _ in range(args.num_classes)]
        class_folders = sorted([f for f in os.listdir(args.data_path+str(folder)) if os.path.isdir(os.path.join(args.data_path+str(folder), f))])
        for class_name, class_folder in enumerate(class_folders):
            class_num[class_name] = len(glob(os.path.join(args.data_path+str(folder)+"/"+class_folder+"/", "*.bmp")))
            ins_labels.extend([class_name for _ in range(class_num[class_name])])
            bag.extend(glob(os.path.join(args.data_path+str(folder)+"/"+class_folder+"/", "*.bmp")))
        max_label = np.where(np.array(class_num) > 0)[0].max()
        all_bag_labels.append(max_label), all_bags.append(bag), all_ins_labels.append(ins_labels)
        all_bag_class_num[max_label] += 1

        bag, ins_labels = np.array(bag), np.array(ins_labels)
        idx = np.arange(len(bag))
        np.random.shuffle(idx)
        bag, ins_labels = bag[idx], ins_labels[idx]

        np.save("%s/bags_per_class/%d/bags/bag_%d" %(args.output_path, max_label, all_bag_class_num[max_label]-1), bag)
        np.save("%s/bags_per_class/%d/ins_labels/ins_labels_%d" %(args.output_path, max_label, all_bag_class_num[max_label]-1), ins_labels)
        np.save("%s/bags_per_class/%d/bag_labels/bag_labels_%d" %(args.output_path, max_label, all_bag_class_num[max_label]-1), max_label)
        # data analysis
        inst_class_num[patient_id]= class_num
        bag_size.append(sum(class_num))
    analysis_def(args, inst_class_num, bag_size)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--dataset', default='LIMUC', type=str)
    parser.add_argument('--data_path', default='./org_data/LIMUC/patient_based_classified_images/', type=str)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--output_path', default='./bag_data/LIMUC/' , type=str)
    args = parser.parse_args()

    main(args)