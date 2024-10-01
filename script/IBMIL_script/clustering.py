import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
import _int_paths
from utils import *
from get_module import get_module
import torch.nn.functional as F
from ckustering_loader import all_one_vs_rest_load_data_bags
import faiss
import time

def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

    
def reduce(args, feats, k):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    prototypes = []
    semantic_shifts = []
    feats = feats.cpu().numpy()

    kmeans = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0)
                          for i in range(k)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(feats[assignments == i].T)
                           for i in range(k)])

    os.makedirs('./%s/datasets_deconf/' % args.output_path, exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes =  prototypes.reshape(-1, 512)
    print(prototypes.shape)
    print('./%s/datasets_deconf/train_bag_cls_agnostic_feats_proto_{k}.npy' % args.output_path)
    print(f'datasets_deconf/{args.dataset}/train_bag_cls_agnostic_feats_proto_{k}.npy')
    np.save('./%s/datasets_deconf/fold=%d_seed=%d-train_bag_cls_agnostic_feats_proto_k=%d.npy' % (args.output_path, args.fold, args.seed, k), prototypes)

    del feats


def main(args):
    fix_seed(args.seed) 
    _, _, model, _, _, _, _, _ = get_module(args)
    args.output_path += '%s/%s/w_pretrain_lr=3e-6/%s/' % (args.dataset, args.data_type, args.mode) 
    make_folder(args)

    ##data loader
    train_loader = all_one_vs_rest_load_data_bags(args)
    model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-pretrain_best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
    
    # forward
    feats_list = []
    model.eval()
    with torch.no_grad():
        for iteration, data in enumerate(train_loader): #enumerate(tqdm(test_loader, leave=False)):
            bags, ins_label, bag_label = data["bags"], data["ins_label"], data["max_label"]
            bags, ins_label, bag_label = bags.to(args.device), ins_label.to(args.device), bag_label.to(args.device)

            y = model(bags, data["len_list"])

            feats_list.append(y["Bag_feature"].cpu())
            
    bag_tensor = torch.cat(feats_list, dim=0)
    # bag_tensor_ag = bag_tensor.view(-1,args.feats_size)
    # for i in [2,4,8,16]:
    reduce(args, bag_tensor, 8)
        
    return 

if __name__ == '__main__':
    for fold in range(5):
    #    for seed in range(1):
        parser = argparse.ArgumentParser()
        # Data selectiion
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default='LIMUC',
                            type=str, help='LIMUC')
        parser.add_argument('--data_type', #書き換え
                            default="5-fold_in_test_balanced_time_order", type=str, help="")
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
        parser.add_argument('--is_test', default=1,
                            type=int, help='1 or 0')           
        parser.add_argument('--is_evaluation', default=0,
                            type=int, help='1 or 0')                        
        # Module Selection
        parser.add_argument('--module',default='IBMIL', 
                            type=str, help="Feat_agg or Output_agg or Att_mil or multi-class_Att_mil or transfomer or one_vs_rest or one_token_OvsR_transfomer_script or shared_OvsR_transfomer")
        parser.add_argument('--mode',default='',    # don't write!
                            type=str, help="")                        
        # Save Path
        parser.add_argument('--output_path',
                            default='./result/', type=str, help="output file name")
        ### Module detail ####
        # Traditional MIL Paramter
        parser.add_argument('--output_agg_method', default="mean",
                            type=str, help='mean')  
        parser.add_argument('--feat_agg_method', default="mean",
                            type=str, help='mean or max or p_norm or LSE')  
        parser.add_argument('--p_val',
                            default=4, type=int, help="1 or 4 or 8")
        # one_vs_rest Paramter
        parser.add_argument('--one_vs_rest_net',default='transfomer', 
                            type=str, help="Feat_mean or binary_Att_mil or multi-class_Att_mil or padded_Transmil or transfomer")
        parser.add_argument('--one', default="0,1,2",
                            type=str, help='one of one_vs_rest')
        # BIGtransfomer layer num
        parser.add_argument('--transfomer_layer_num', default=1, type=int, help='1 or 2 or 6 or 12')
        parser.add_argument('--loss_weight', default=0, type=int, help='0 or 1')
        parser.add_argument('--is_temp_softmax', default=0, type=int, help='0 or 1')
        parser.add_argument('--mask_topk', default=0.1, type=float, help='0 or 0.1 or 0.3 or 0.5 or 0.8')
        # shared_OvsR_transfomer
        parser.add_argument('--clstoken_mask', default=1, type=int, help='0 or 1')
        # GT mask mode
        parser.add_argument('--maskmode',default='hard', type=str, help="soft or hard")   
        parser.add_argument('--is_aug',default='1', type=int, help="0 or 1")
        parser.add_argument('--is_testmask',default='0', type=int, help="0 or 1")  
        # Bag augmentation
        parser.add_argument('--augmode',default='one-under-mix', type=str, help="under_random or one_under or hieral or equal_random or low_mask or high_mask") 
        parser.add_argument('--low_mask_ratio', default=0.5, type=float, help='')  
        parser.add_argument('--top_mask_ratio', default=0.1, type=float, help='')  
        parser.add_argument('--is_pretrain',default='0', type=int, help="0 or 1")  
        # Contrastive loss
        parser.add_argument('--contrastive',default='InfoNCE', type=str, help="SCOL or InfoNCE")   
        parser.add_argument('--psd_topk_ratio',default=0.5, type=float, help="SCOL or InfoNCE")   
        parser.add_argument('--psd_lowk_ratio',default=0.1, type=float, help="SCOL or InfoNCE")  
        parser.add_argument('--gt_leak',default=1, type=int, help="0 or 1")  
        parser.add_argument('--attn_choice',default="one_node", type=str, help="one_node or three_node")  
        # Iterative Contrastive loss 
        parser.add_argument('--warmup_epoch',default=0, type=int, help="50")  
        parser.add_argument('--is_consistency',default=1, type=int, help="")  
        # reject ratio
        parser.add_argument('--reject_ratio',default=0.2, type=float, help="50") 
        # positive boost augmentation
        parser.add_argument('--pos_boost_topk_ratio',default=0.5, type=float, help="") 
        parser.add_argument('--pos_boost_topk_mix',default=1, type=float, help="") 
        # Additive Parameter
        parser.add_argument('--add_agg_method', default="TransMIL",
                            type=str, help='Attention_MIL or TransMIL')
        # IBMIL parameter
        parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
        # time_order 
        parser.add_argument('--is_pe', default="0",
                            type=str, help='')
        args = parser.parse_args()
        
        result_dict = main(args)
