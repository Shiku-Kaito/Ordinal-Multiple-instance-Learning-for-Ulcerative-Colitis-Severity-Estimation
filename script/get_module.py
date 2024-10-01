import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from dataloader  import *
from torchvision.models import resnet18, resnet34, resnet50

from MIL_script.train import train_net as MIL_train_net
from MIL_script.network import Feat_agg, Output_agg
from MIL_script.eval import eval_net as MIL_eval_net

from Output_max_script.MIL_train import train_net as Outputmax_train_net
from Output_max_script.MIL_network import Output_max
from Output_max_script.eval import eval_net as Outputmax_eval_net

from Krank_MIL_script.train import train_net as krank_mil_train_net
from Krank_MIL_script.network import Krank_MIL
from Krank_MIL_script.eval import eval_net as krank_mil_eval_net

from multi_class_attMIL_script.multi_class_attMIL_train import train_net as multi_class_att_train_net
from multi_class_attMIL_script.multi_class_attMIL_network import Multi_class_Attention
from multi_class_attMIL_script.eval import eval_net as multi_class_att_eval_net

from transfomer_script.train import train_net as transformer_train_net
from transfomer_script.network import Transformer
from transfomer_script.eval import eval_net as transformer_eval_net

from transfomer_reg_script.train import train_net as reg_train_net
from transfomer_reg_script.network import Transformer_reg
from transfomer_reg_script.eval import eval_net as reg_eval_net

from DSMIL_script.train import train_net as dsmil_train_net
from DSMIL_script.network import DSMIL
from DSMIL_script.eval import eval_net as dsmil_eval_net

from IBMIL_script.train import train_net as ibmil_train_net
from IBMIL_script.network import TransMIL as IBMIL_transmil
from IBMIL_script.eval import eval_net as ibmil_eval_net

from selective_aggregated_transfomer_script.train import train_net as SATOMIL_network_train_net
from selective_aggregated_transfomer_script.eval import eval_net as SATOMIL_network_eval_net
from selective_aggregated_transfomer_script.network import Selective_Aggregated_Transfomer

from transfomer_krank_script.train import train_net as transfomer_krank_train_net
from transfomer_krank_script.eval import eval_net as transfomer_krank_eval_net
from transfomer_krank_script.network import Transfomer_Krank

from additive_script.additiveMIL_network import get_additive_mil_model_n_weights, get_transmil_model_n_weights, get_additive_transmil_model_n_weights
from additive_script.additiveMIL_train import train_net as add_train_net
from additive_script.eval import eval_net as add_eval_net

from transfomer_softlabel_script.train import train_net as transfomer_softlabel_train_net

from supervised.k_rank.k_rank_net import K_rank_net
from supervised.k_rank.train import train_net as K_rank_train_net
from supervised.k_rank.eval import eval_net as K_rank_eval_net

from supervised.classification.classification_net import Classification_net
from supervised.classification.train import train_net as classification_train_net
from supervised.classification.eval import eval_net as classification_eval_net

import transformer_cpl_script.cpl_utils as cpl_utils
from transformer_cpl_script.cpl.cpl_model import Transformer_CPL
from transformer_cpl_script.cpl_train_net import train_net  as cpl_train_net
from transformer_cpl_script.eval import eval_net  as cpl_eval_net
import transformer_cpl_script.cpl as cpl

from transfomer_POE_script.train import train_net as POE_train_net
from transfomer_POE_script.network import Transformer_POE
from transfomer_POE_script.eval import eval_net as POE_eval_net
from transfomer_POE_script.POE_dataloader import POE_load_data_bags
from transfomer_POE_script.POE_loss import ProbOrdiLoss

from losses import *

def get_module(args):
    if args.module ==  "Feat_agg":
        args.mode =  "Feat_%s" % args.feat_agg_method
        if args.feat_agg_method == "p_norm" or args.feat_agg_method == "LSE":
            args.mode += "=%s" % str(args.p_val)    
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Feat_agg(args.classes, args.feat_agg_method, args.p_val) 
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()  
        # Train net
        train_net = MIL_train_net
        eval_net = MIL_eval_net

    elif args.module ==  "output_mean":
        args.mode = "Output_mean"
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Output_agg(args.classes)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = MIL_train_net
        eval_net = MIL_eval_net

    elif args.module ==  "output_max":
        # args.mode = "Output_max_w_threshold"
        args.mode = "Output_max"
        # Dataloader
        train_loader, val_loader, test_loader = regression_load_data_bags(args) 
        # Model
        model = Output_max()
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.MSELoss()
        # Train net
        train_net = Outputmax_train_net
        eval_net = Outputmax_eval_net

    elif args.module == "multi-class_Att_mil":
        args.mode = "multi-class_Att_mil"
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Multi_class_Attention(args.classes)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = multi_class_att_train_net
        eval_net = multi_class_att_eval_net

    elif args.module ==  "additive_mil":
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        if args.add_agg_method == "Attention_MIL":
            args.mode = "additive_Attntionmil"
            model = get_additive_mil_model_n_weights(num_classes=4)
        elif args.add_agg_method == "TransMIL":
            args.mode = "additive_TransMIL"
            model = get_additive_transmil_model_n_weights(num_classes=4)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net 
        train_net = add_train_net
        eval_net = add_eval_net

    elif args.module ==  "transfomer":
        args.mode = "transfomer_layernum=%d" % (args.transfomer_layer_num)
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Transformer(num_classes=4, embed_dim=512, depth=args.transfomer_layer_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = transformer_train_net
        eval_net = transformer_eval_net

    elif args.module ==  "IBMIL":
        args.mode = "IBMIL_transMIL"
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = IBMIL_transmil(input_size=512, n_classes=4, confounder_path=args.c_path)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = ibmil_train_net
        eval_net = ibmil_eval_net

    elif args.module ==  "transfomer_reg":
        args.mode = "regression_transfomer_layernum=%d" % (args.transfomer_layer_num)
        # Dataloader
        train_loader, val_loader, test_loader = regression_load_data_bags(args) 
        # Model
        model = Transformer_reg(num_classes=4, embed_dim=512, depth=args.transfomer_layer_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.MSELoss()
        # Train net
        train_net = reg_train_net
        eval_net = reg_eval_net

    elif args.module ==  "transfomer_cpl":
        args.mode = "transfomer_cpl_layernum=%d_const=%s-%s_tau=%.2f" % (args.transfomer_layer_num, args.constraint, args.metric_method, args.tau)
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        
        # Model
        metric_method = cpl_utils.get_metric_method(args)
        if args.constraint == 'S-P':
            proxies_learner = cpl.BaseProxiesLearner(num_ranks=4, dim=512)
            criterion = cpl.SoftCplPoissonLoss(num_ranks=4, tau=args.tau, loss_lam=args.loss_lam)
        elif args.constraint == 'S-B':
            proxies_learner = cpl.BaseProxiesLearner(num_ranks=4, dim=512)
            criterion = cpl.SoftCplBinomialLoss(num_ranks=4, tau=args.tau, loss_lam=args.loss_lam)
        elif args.constraint == 'H-L':
            proxies_learner = cpl.LinearProxiesLearner(num_ranks=4, dim=512)
            criterion = cpl.HardCplLoss()
            metric_method = cpl.EuclideanMetric()
        elif args.constraint == 'H-S':
            proxies_learner = cpl.SemicircularProxiesLearner(num_ranks=4, dim=512)
            criterion = cpl.HardCplLoss()
            metric_method = cpl.CosineMetric(args.cosine_scale)
        else:
            raise NotImplementedError

        model = Transformer_CPL(num_classes=4, embed_dim=512, depth=args.transfomer_layer_num, 
                                proxies_learner=proxies_learner, metric_method=metric_method)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = criterion
        # Train net
        train_net = cpl_train_net
        eval_net = cpl_eval_net

    elif args.module ==  "transfomer_POE":
        args.mode = "transfomer_poe_layernum=%d_%s_dist=%s" % (args.transfomer_layer_num, args.main_loss_type, args.distance)
        # Dataloader
        train_loader, val_loader, test_loader = POE_load_data_bags(args) 
        
        # Model
        model = Transformer_POE(num_classes=4, embed_dim=512, depth=args.transfomer_layer_num, num_output_neurons=args.num_output_neurons)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = ProbOrdiLoss(distance=args.distance, alpha_coeff=args.alpha_coeff,
                             beta_coeff=args.beta_coeff, margin=args.margin, main_loss_type=args.main_loss_type)
        # Train net
        train_net = POE_train_net
        eval_net = POE_eval_net

    elif args.module ==  "dsmil":
        args.mode = "dsmil"
        # Dataloader
        train_loader, val_loader, test_loader = regression_load_data_bags(args) 
        # Model
        model = DSMIL(classes=4)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = {"IC_loss":nn.MSELoss(), "BC_loss":nn.CrossEntropyLoss()}
        # Train net
        train_net = dsmil_train_net
        eval_net = dsmil_eval_net

    elif args.module == "multi-class_Att_mil":
        args.mode = "multi-class_Att_mil"
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Multi_class_Attention(args.classes)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = multi_class_att_train_net
        eval_net = multi_class_att_eval_net


    elif args.module ==  "krank_mil":
        args.mode = "/krank_mil_emsemble=%s/" % (args.emsemble_mode)
        # Dataloader
        if args.dataset == "LIMUC":
            train_loader, val_loader, test_loader = all_one_vs_rest_load_LIMUC_regression_data_bags(args) 
        # Model
        model = Krank_MIL()
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = {"0vs123": nn.BCEWithLogitsLoss(), "01vs23": nn.BCEWithLogitsLoss(), "012vs3": nn.BCEWithLogitsLoss()}

        # Train net
        train_net = krank_mil_train_net
        eval_net = krank_mil_eval_net

    elif args.module ==  "Selective_Aggregated_Transfomer":
        args.mode = "Selective_Aggregated_Transfomer_layernum=%d_clsotken_mask=%d" % (args.transfomer_layer_num, args.clstoken_mask)
        # Dataloader
        if args.dataset == "LIMUC":
            train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Selective_Aggregated_Transfomer(num_classes=2, embed_dim=512, depth=args.transfomer_layer_num, clstoken_mask=args.clstoken_mask)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = {"0vs123": nn.CrossEntropyLoss(), "01vs23": nn.CrossEntropyLoss(), "012vs3": nn.CrossEntropyLoss()}
        # Train net
        train_net = SATOMIL_network_train_net
        eval_net = SATOMIL_network_eval_net

    elif args.module ==  "transfomer_krank":
        args.mode = "transfomer_krank_layernum=%d" % args.transfomer_layer_num
        # Dataloader
        train_loader, val_loader, test_loader = all_one_vs_rest_load_data_bags(args) 
        # Model
        model = Transfomer_Krank(num_classes=2, embed_dim=512, depth=args.transfomer_layer_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = {"0vs123": nn.CrossEntropyLoss(), "01vs23": nn.CrossEntropyLoss(), "012vs3": nn.CrossEntropyLoss()}

        # Train net
        train_net = transfomer_krank_train_net
        eval_net = transfomer_krank_eval_net

    elif args.module ==  "transfomer_softlabel":
        args.mode = "transfomer_softlabel_layernum=%d" % (args.transfomer_layer_num)
        # Dataloader
        train_loader, val_loader, test_loader = SORD_load_data_bags(args) 
        # Model
        model = Transformer(num_classes=4, embed_dim=512, depth=args.transfomer_layer_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = {"soft": cross_entropy_loss, "hard": nn.CrossEntropyLoss()}
        # Train net
        train_net = transfomer_softlabel_train_net
        eval_net = transformer_eval_net

    elif args.module ==  "supervised_k_rank":
        args.mode = "supervised_k_rank"
        # Dataloader
        if args.dataset == "LIMUC":
            train_loader, val_loader, test_loader = supervised_load_LIMUC_time_data_bags(args) 
        # Model
        model = K_rank_net()
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = {"0vs123": nn.CrossEntropyLoss(), "01vs23": nn.CrossEntropyLoss(), "012vs3": nn.CrossEntropyLoss()}
        # Train net
        train_net = K_rank_train_net
        eval_net = K_rank_eval_net   
    
    elif args.module ==  "supervised_classification":
        args.mode = "supervised_classification"
        # Dataloader
        if args.dataset == "LIMUC":
            train_loader, val_loader, test_loader = supervised_load_LIMUC_time_data_bags(args) 

        # Model
        model = Classification_net()
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = classification_train_net
        eval_net = classification_eval_net   

    else:
        print("Module ERROR!!!!!")

    return train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader