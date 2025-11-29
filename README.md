# Ordinal-Multiple-instance-Learning-for-Ulcerative-Colitis-Severity-Estimation
Ordinal Multiple-instance Learning for Ulcerative Colitis Severity Estimation with Selective Aggregated Transformer, in WACV2025🎉🎉!.
Shikui Kaito, Kaszuya nishimura, Daiki Suehiro, Kiyohito Tanaka, Ryoma Bise

![Alt Text](./method.jpg)

## 📑 Abstract
*Patient-level diagnosis of severity in ulcerative colitis (UC) is common in clinical, where the most severe score in a patient is recorded.
However, previous UC classification methods ({\it i.e.,} image-level estimation) mainly assumed the input was a single image. Thus, these methods can not utilize severity labels recorded in clinical practice.
In this paper, we propose a patient-level severity estimation method by a transformer with selective aggregator tokens, where a severity label is estimated from multiple images taken from a patient, similar to a clinical setting.
Our method can effectively aggregate features of severe parts from a set of images captured in each patient, and it facilitates improving the discriminative ability between adjacent severity classes.
Experiments demonstrate the effectiveness of the proposed method on two datasets compared with the state-of-the-art MIL methods.
Moreover, we evaluated our method in the clinical setting and confirmed that our method outperformed the previous image-level methods.
*

# ⬇️ Requirement
To set up their environment, please run:  
(we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda env create -n max_label -f max_label.yml
conda activate max_label
```

# 🌐 Download dataset
Please download the LIMUC dataset from here
```
https://zenodo.org/records/5827695#.Yi8GJ3pByUk
```
# 💻 Make dataset
You can create datasets by running the following code. 
```
python ./make_bag/make_bags_LIMUC.py
python ./make_bag/crossvalidation_LIMUC.py
python ./make_bag/LIMUC_bag_time_ordering.py
```

# 🔥 Training & Test for Selective Aggregated Transformer
After creating your python environment and Dataset which can be made by following above command, you can run Selective Aggregated Transformer code.
If you want to train a Selective aggregated transformer, please run following command. 5 fold training is automatically done in our code.
```
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "Selective_Aggregated_Transfomer" --batch_size 32 --transfomer_layer_num 1 --clstoken_mask 1 --is_evaluation 0 --device 'cuda:0' 
```
If you want to evaluate Selective aggregated transformer, please run following command. 5 fold trainevaluation is automatically done in our code.
```
python ./script/main.py --dataset "LIMUC" --data_type "5-fold_in_test_balanced_time_order" --module "Selective_Aggregated_Transfomer" --batch_size 32 --transfomer_layer_num 1 --clstoken_mask 1 --is_evaluation 1 --device 'cuda:0'
```
# 📊 Training & Test for comparison method
If you want to train the comparison methods, please run the following command.
```
bash ./script/train_comparison.sh
```

If you want to evaluate comparison method, please run following command.
```
bash ./script/eval_comparison.sh
```

# 🔍 Citation
If you find this repository helpful, please consider citing:
```
@InProceedings{Shiku_2025_WACV,
    author    = {Shiku, Kaito and Nishimura, Kazuya and Suehiro, Daiki and Tanaka, Kiyohito and Bise, Ryoma},
    title     = {Ordinal Multiple-Instance Learning for Ulcerative Colitis Severity Estimation with Selective Aggregated Transformer},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4290-4299}
}
```

# ✏️ Author
@ Shiku Kaito  
・ Contact: kaito.shiku@human.ait.kyushu-u.ac.jp
