#!/bin/bash
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "output_mean" --is_evaluation 1 --device 'cuda:0'
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "output_max"  --is_evaluation 1 --device 'cuda:0'
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "Feat_agg" --feat_agg_method "mean" --is_evaluation 1 --device 'cuda:0'  
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "Feat_agg" --feat_agg_method "max" --is_evaluation 1 --device 'cuda:0'  
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "multi-class_Att_mil" --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer" --batch_size 32 --transfomer_layer_num 1 --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "dsmil" --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "additive_mil" --add_agg_method "TransMIL" --batch_size 32 --is_evaluation 1 --device 'cuda:0'
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "IBMIL" --is_evaluation 1 --device 'cuda:0' --c_path 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "krank_mil" --emsemble_mode "threshold"  --is_evaluation 1 --device 'cuda:0' 

python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer_reg" --batch_size 32 --transfomer_layer_num 1 --is_evaluation 1 --device 'cuda:0'
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer_softlabel" --batch_size 32 --transfomer_layer_num 1 --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer_POE" --main-loss-type 'reg'  --num-output-neurons 1  --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer_cpl" --constraint 'H-S' --metric_method 'C' --tau 0.13 --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer_cpl" --constraint 'S-P' --metric_method 'C' --tau 0.11 --is_evaluation 1 --device 'cuda:0' 

python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "transfomer_krank" --batch_size 32 --transfomer_layer_num 1 --clstoken_mask 1  --is_evaluation 1 --device 'cuda:0' 
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "Selective_Aggregated_Transfomer" --batch_size 32 --transfomer_layer_num 1 --clstoken_mask 1 --is_evaluation 0 --device 'cuda:0' 
