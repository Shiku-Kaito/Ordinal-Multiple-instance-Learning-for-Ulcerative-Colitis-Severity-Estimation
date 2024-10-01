import torch
from torch import nn

from additive_script.additive_modules.common import Resnet
from additive_script.additive_modules.common import ShuffleNetV2
from additive_script.additive_modules.attention_mil import DefaultAttentionModule, DefaultClassifier, DefaultMILGraph
from additive_script.additive_modules.additive_mil import AdditiveClassifier
from additive_script.additive_modules.additive_transmil import TransformerMILGraph, AdditiveTransMIL
from additive_script.additive_modules.transmil import TransMIL
from additive_script.additive_modules.count_transmil import CountTransformerMILGraph, CountTransMIL

def get_attention_mil_model_n_weights():
    model = DefaultMILGraph(
        featurizer = ShuffleNetV2(),
        pointer=DefaultAttentionModule(hidden_activation = nn.LeakyReLU(0.2), hidden_dims=[256, 256], input_dims=1024),
        classifier=DefaultClassifier(hidden_dims=[256, 256], input_dims=1024, output_dims=2)
    )
    weights_path = "artifacts/nsclc/model_weights/wt_attention_mil.pth"
    return model, weights_path


def get_additive_mil_model_n_weights(num_classes):
    model = DefaultMILGraph(
        featurizer = Resnet(),
        pointer=DefaultAttentionModule(hidden_activation = nn.LeakyReLU(0.2), hidden_dims=[256, 256], input_dims=512),
        classifier=AdditiveClassifier(hidden_dims=[256, 256], input_dims=512, output_dims=num_classes)
    )
    return model


def get_transmil_model_n_weights(num_classes):
    model = TransformerMILGraph(
        featurizer = Resnet(),
        classifier = TransMIL(num_classes)
    )
    weights_path = None
    return model, weights_path


def get_additive_transmil_model_n_weights(num_classes):
    model = TransformerMILGraph(
        featurizer = Resnet(),
        classifier = AdditiveTransMIL(additive_hidden_dims=[256], n_classes=num_classes)
    )
    return model


def get_count_transmil_model_n_weights(arg):
    model = CountTransformerMILGraph(
        featurizer = Resnet(),
        classifier = CountTransMIL(additive_hidden_dims=[256], args=arg)
    )
    weights_path = "artifacts/nsclc/model_weights/wt_additive_transmil.pth"
    return model, weights_path


def load_torch_model(model, weights):
    state_dict = torch.load(weights, map_location=torch.device('cpu'))
    print(model.load_state_dict(state_dict))
    print("Model loading complete ...")


if __name__ == '__main__':
    print("Loading AttentionMIL model ...")
    load_torch_model(*get_attention_mil_model_n_weights())

    print("Loading AdditiveMIL model ...")
    load_torch_model(*get_additive_mil_model_n_weights())
    
    print("Loading TransMIL model ...")
    load_torch_model(*get_transmil_model_n_weights())

    print("Loading AdditiveTransMIL model ...")
    load_torch_model(*get_transmil_model_n_weights())

