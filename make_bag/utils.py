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


def make_histgram(data_list, path, x_max, x_min, x, y, normalize=True):
    if normalize==True:
        hist, edges = np.histogram(data_list, bins=20 ,range=(min,max), density=True)
        w = edges[1] - edges[0]
        hist = hist * w
        # グラフにプロット
        plt.bar(edges[:-1], hist, w)
        plt.savefig(path)
        plt.close()
    else:
        # グラフにプロット
        plt.hist(data_list, range=(x_min, x_max))
        plt.xlabel(x)
        plt.ylabel(y)
        plt.savefig(path)
        plt.close()
    return
