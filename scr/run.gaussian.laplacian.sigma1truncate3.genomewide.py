from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
# import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from math import log10
from torch.utils import data
from time import gmtime, strftime
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import convolve
import sys
import straw


def laplacian_filter(matrix):
    # 定义拉普拉斯算子（卷积核）
    kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    filtered_matrix = convolve(matrix, kernel, mode='reflect')

    return filtered_matrix



file_path = '/public1/xinyu/CohesinProject/ChIAPET/raw.data'
chr_genom = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X']


for chrN in chr_genom:
    print("reading: " + str(chrN))
## 转为2D矩阵
    df = utils.Hic_reader(os.path.join(file_path, 'GM12878_WT_ChIAPET_CTCF.intra_iPET_ALL.hic'), chr_list = [chrN])

    chrN="chr" + str(chrN)
    print("processing: " + chrN)
    df_2D_matrix = utils.HiCMatrix_2D(hic_data = df, chr = chrN)

    bin1_idx_df, bin2_idx_df = utils.get_signal_indice(signal_tmp = df, hic_2D_tmp = df_2D_matrix)

## Gaussian
    df_2D_gaussian = utils.smooth_gaussian(hic_2DMat = df_2D_matrix, sigma = 1, truncate = 3)
    df_2D_gaussian_sigma1truncate3 = [df_2D_gaussian[bin1_idx_df[i], bin2_idx_df[i]] for i in range(len(df))]
    df['gaussian'] = df_2D_gaussian_sigma1truncate3
    df['bin1_e'] = df['bin1'] + 5000
    df['bin2_e'] = df['bin2'] + 5000
    df['chromosome2'] = df['chromosome']
   # df[['chromosome', 'bin1', 'bin1_e', 'chromosome2', 'bin2', 'bin2_e', 'gaussian']].to_csv(data_type + "_" + chrN + ".gaussian_sigma1truncate3.bedpe", sep = '\t', index = False, header = False)

## 拉普拉斯平滑
    df_2D_laplacian = laplacian_filter(np.array(df_2D_gaussian))
    df_2D_laplacian[df_2D_laplacian < 0] = 0
    df_2D_laplacian_sum = np.array(df_2D_gaussian) + df_2D_laplacian
    df_2D_laplacian_sum_sigma1truncate3 = [df_2D_laplacian_sum[bin1_idx_df[i], bin2_idx_df[i]] for i in range(len(df))]
    df['laplacian'] = df_2D_laplacian_sum_sigma1truncate3
    df['bin1_e'] = df['bin1'] + 5000
    df['bin2_e'] = df['bin2'] + 5000
    df['chromosome2'] = df['chromosome']
    df[['chromosome', 'bin1', 'bin1_e', 'chromosome2', 'bin2', 'bin2_e', 'laplacian']].to_csv('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.test4/' + "WT_CTCF_" + chrN + ".gaussian_sigma1truncate3.laplacian.bedpe", sep = '\t', index = False, header = False)



