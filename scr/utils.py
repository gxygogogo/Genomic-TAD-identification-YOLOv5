# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
import straw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve1d,convolve
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed

# 读取单条染色体的hic数据
def read_single_chromosome(hic_path, chr):
    # 读取当前染色体的hic数据
    print('-------------------' + 'READING CHROMOSOME' + str(chr) + '--------------------')
    hic_tmp = straw.straw('NONE', hic_path, str(chr), str(chr), 'BP', 5000)
    # 将读取到的数据添加染色体的标签
    hic_tmp.append(['chr' + str(chr)] * len(hic_tmp[0]))
    # 将当前染色体的hic数据转换成DataFrame格式
    hic_tmp_df = pd.DataFrame(hic_tmp).T
    # 设置列名
    hic_tmp_df.columns = ['bin1', 'bin2', 'count', 'chromosome']
    return hic_tmp_df

# 读取所有染色体的hic数据
def Hic_reader(hic_path, chr_list):
    # 多核心并行读取所有染色体的数据
    hic_dfs = Parallel(n_jobs=-1)(delayed(read_single_chromosome)(hic_path, chr) for chr in chr_list)
    # 合并所有读取的数据
    hic_df = pd.concat(hic_dfs, ignore_index=True)
    return hic_df

# 处理信号点
def process_data(signal_df, SMC1A_df, SA1_df, SA2_df, signal_type):
    # 分解region列为chromosome, bin1, bin2
    signal_df_copy = signal_df.copy()
    signal_df_copy[['chromosome1', 'bin1_s', 'bin1_e', 'chromosome2', 'bin2_s', 'bin2_e']] = signal_df_copy['signal'].str.extract(r'(chr[^\_]+)_(\d+)_(\d+)_(chr[^\_]+)_(\d+)_(\d+)')
    signal_df_copy['bin1'] = signal_df_copy['bin1_s'].astype(int) - 1
    signal_df_copy['bin2'] = signal_df_copy['bin2_s'].astype(int) - 1

    signal_df_type = signal_df_copy[signal_df_copy['type'] == signal_type]

    # 提取信号
    if signal_type == 'SMC1A':
        merged_SMC1A = signal_df_type.merge(SMC1A_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SMC1A'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([merged_SMC1A, [], []])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [merged_SMC1A, [], []]
    elif signal_type == 'SA1':
        merged_SA1 = signal_df_type.merge(SA1_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA1'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([[], merged_SA1, []])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [[], merged_SA1, []]
    elif signal_type == 'SA2':
        merged_SA2 = signal_df_type.merge(SA2_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA2'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([[], [], merged_SA2])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [[], [], merged_SA2]
    elif signal_type == 'SMC1A_SA1':
        merged_SMC1A = signal_df_type.merge(SMC1A_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SMC1A'))
        merged_SA1 = signal_df_type.merge(SA1_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA1'))
        merged_SA2 = signal_df_type.merge(SA2_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA2'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([merged_SMC1A, merged_SA1, merged_SA2])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [merged_SMC1A, merged_SA1, merged_SA2]
    elif signal_type == 'SMC1A_SA2':
        merged_SMC1A = signal_df_type.merge(SMC1A_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SMC1A'))
        merged_SA1 = signal_df_type.merge(SA1_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA1'))
        merged_SA2 = signal_df_type.merge(SA2_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA2'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([merged_SMC1A, merged_SA1, merged_SA2])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [merged_SMC1A, merged_SA1, merged_SA2]
    elif signal_type == 'SA1_SA2':
        merged_SMC1A = signal_df_type.merge(SMC1A_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SMC1A'))
        merged_SA1 = signal_df_type.merge(SA1_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA1'))
        merged_SA2 = signal_df_type.merge(SA2_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA2'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([merged_SMC1A, merged_SA1, merged_SA2])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [merged_SMC1A, merged_SA1, merged_SA2]
    elif signal_type == 'SMC1A_SA1_SA2':
        merged_SMC1A = signal_df_type.merge(SMC1A_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SMC1A'))
        merged_SA1 = signal_df_type.merge(SA1_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA1'))
        merged_SA2 = signal_df_type.merge(SA2_df, left_on=['chromosome1', 'bin1', 'bin2'], right_on=['chromosome', 'bin1', 'bin2'], how='left', suffixes=('', '_SA2'))
        for index, (sublist_type, sublist_length) in enumerate(analyze_sublists([merged_SMC1A, merged_SA1, merged_SA2])):
            print(f"子列表 {index + 1}: 类型 = {sublist_type}, 大小 = {sublist_length}")
        return [merged_SMC1A, merged_SA1, merged_SA2]

# 分析子列表的类型和长度
def analyze_sublists(multi_list):
    result = []
    for sublist in multi_list:
        sublist_type = type(sublist)
        sublist_length = len(sublist)
        result.append((sublist_type, sublist_length))
    return result

# 获取信号在2D矩阵中的索引
def get_signal_indice(signal_tmp, hic_2D_tmp):
    columns_list = hic_2D_tmp.columns.to_list()
    columns_dict = {item: idx for idx, item in enumerate(columns_list)}
    
    bin1_indices = signal_tmp['bin1'].map(columns_dict).fillna(-1).astype(int).to_list()
    bin2_indices = signal_tmp['bin2'].map(columns_dict).fillna(-1).astype(int).to_list()
    
    return bin1_indices, bin2_indices

# 绘制 Hi-C 热图
def plot_HicMatrix(chromosome, start, end, hic_data, output, vmin = 0, vmax = 10, cmap = 'viridis', sns_if = True):
    ## 读入 Hic 矩阵
    hic_count = hic_data

    ## 提取相应区域的矩阵
    hic_count_df_1 = hic_count[(hic_count['chromosome']==chromosome) & (hic_count['bin1']>=start) & (hic_count['bin2']<=end)]
    hic_count_df_1 = hic_count_df_1[['bin1', 'bin2', 'count']]
    hic_count_df_1[['count']] = hic_count_df_1[['count']].astype(int)
    hic_count_df_1.columns = [0, 1, 2]

    ## 构建对称宽数据框
    axis_all_bin = list(set([x for x in hic_count_df_1[0]] + [x for x in hic_count_df_1[1]]))
    axis_all_bin.sort()
    matrix = np.zeros((len(axis_all_bin), len(axis_all_bin)), dtype=np.float32)
    df = pd.DataFrame(matrix, index = axis_all_bin, columns = axis_all_bin)
    for i in range(len(hic_count_df_1)):
        df.loc[hic_count_df_1.iloc[i,0], hic_count_df_1.iloc[i,1]] = hic_count_df_1.iloc[i,2]
        df.loc[hic_count_df_1.iloc[i,1], hic_count_df_1.iloc[i,0]] = hic_count_df_1.iloc[i,2]

    if sns_if == True:
        # 使用 seaborn 生成热图
        plt.figure(figsize = (10, 8))
        heatmap = sns.heatmap(np.array(df), cmap = cmap, vmin = vmin, vmax = vmax)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        plt.xlabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        plt.ylabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        # 设置边框
        ax = heatmap.axes
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        # 保存热图
        plt.savefig(output + '_sns.pdf', dpi = 400)
        # 关闭当前图像窗口
        plt.close()
    else:
        # 使用 imshow 生成热图
        plt.figure(figsize = (10, 8))
        plt.imshow(np.array(df), cmap = cmap, vmin = vmin, vmax = vmax, aspect = 'equal')
        plt.colorbar()
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        plt.xlabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        plt.ylabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        plt.savefig(output + '_imshow.pdf', dpi = 400)
        # 关闭当前图像窗口
        plt.close()


# 将长数据形式的HiC矩阵转换为宽数据形式的HiC矩阵
# def HiCMatrix_2D(hic_data, chr):
#     ## 读入 Hic 矩阵
#     hic_count = hic_data

#     ## 提取相应区域的矩阵
#     hic_count_df_1 = hic_count[hic_count['chromosome']==chr]
#     hic_count_df_1 = hic_count_df_1[['bin1', 'bin2', 'count']]
#     hic_count_df_1[['count']] = hic_count_df_1[['count']].astype(int)
#     hic_count_df_1.columns = [0, 1, 2]

#     ## 构建对称宽数据框
#     axis_all_bin = list(set([x for x in hic_count_df_1[0]] + [x for x in hic_count_df_1[1]]))
#     axis_all_bin.sort()
#     matrix = np.zeros((len(axis_all_bin), len(axis_all_bin)), dtype=np.float32)
#     df = pd.DataFrame(matrix, index = axis_all_bin, columns = axis_all_bin)
#     for i in range(len(hic_count_df_1)):
#         df.loc[hic_count_df_1.iloc[i,0], hic_count_df_1.iloc[i,1]] = hic_count_df_1.iloc[i,2]
#         df.loc[hic_count_df_1.iloc[i,1], hic_count_df_1.iloc[i,0]] = hic_count_df_1.iloc[i,2]

#     return df
## 将互作信息转为2D矩阵，单条染色体进行
def HiCMatrix_2D(hic_data, chr):
    # 提取特定染色体的数据
    hic_count_df_1 = hic_data[hic_data['chromosome'] == chr]
    hic_count_df_1 = hic_count_df_1[['bin1', 'bin2', 'count']]
    hic_count_df_1[['count']] = hic_count_df_1[['count']].astype(int)
    
    # 获取所有的bin
    axis_all_bin = list(set(hic_count_df_1['bin1']).union(set(hic_count_df_1['bin2'])))
    axis_all_bin.sort()
    
    # 初始化对称矩阵
    size = len(axis_all_bin)
    bin_index = {bin: i for i, bin in enumerate(axis_all_bin)}
    matrix = np.zeros((size, size), dtype = np.float32)
    
    # 填充矩阵数据
    bin1_indices = hic_count_df_1['bin1'].map(bin_index).to_numpy()
    bin2_indices = hic_count_df_1['bin2'].map(bin_index).to_numpy()
    counts = hic_count_df_1['count'].to_numpy()
    
    matrix[bin1_indices, bin2_indices] = counts
    matrix[bin2_indices, bin1_indices] = counts  # 确保矩阵对称
    
    # 转换为DataFrame
    df = pd.DataFrame(matrix, index = axis_all_bin, columns = axis_all_bin)
    
    return df


# 高斯滤波平滑
def smooth_gaussian(hic_2DMat, sigma = 1, truncate = 3):
    ## 读入 Hic 矩阵
    hic_matrix = hic_2DMat
    filtered_hic_matrix = gaussian_filter(hic_matrix, sigma = sigma, truncate = truncate)

    return filtered_hic_matrix

# 1D 高斯滤波函数
def gaussian_filter_1d(hic_2DMat, kernel_size = 3, sigma = 1, mode = 'reflect', axis = 0):
    # 确定核的范围
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    # 计算高斯函数值
    kernel_1d = np.exp(-0.5 * (x / sigma)**2)
    # 归一化
    kernel_1d /= kernel_1d.sum()
    gaussian_filtered_1D = convolve1d(hic_2DMat, kernel, mode = mode, axis = axis)

    return gaussian_filtered_1D

# 2D 高斯滤波函数
def gaussian_filter_2d(hic_2DMat, kernel_size = 3, sigma = 1, mode = 'reflect'):
    # 确定核的范围
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    # 计算高斯函数值
    kernel_1d = np.exp(-0.5 * (x / sigma)**2)
    # 归一化
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()
    gaussian_filtered_2D = convolve1d(hic_2DMat, kernel, mode = mode)

    return gaussian_filtered_2D
    

# # 高斯滤波平滑
# def smooth_gaussian_V2(hic_2DMat, sigma = 1):
#     ## 读入 Hic 矩阵
#     hic_matrix = hic_2DMat
#     filtered_hic_matrix = gaussian_filter(hic_matrix, sigma = sigma)

#     return filtered_hic_matrix

# 对矩阵进行分块
def get_submatrix(matrix, center_row, center_col, window_size = 5):
    half_window = window_size // 2
    rows, cols = matrix.shape

    # 计算子矩阵的起始和结束行/列
    row_start = max(0, center_row - half_window)
    row_end = min(rows, center_row + half_window + 1)
    col_start = max(0, center_col - half_window)
    col_end = min(cols, center_col + half_window + 1)

    # 创建一个全零矩阵，用于存储子矩阵
    submatrix = np.zeros((window_size, window_size))

    # 填充子矩阵
    submatrix[
        max(0, half_window - center_row):min(window_size, rows - center_row + half_window),
        max(0, half_window - center_col):min(window_size, cols - center_col + half_window)
    ] = matrix[row_start:row_end, col_start:col_end]
    
    return submatrix


# 使用核密度估计计算该信号的核密度
def kernel_density(matrix, bandwidth = 0.2, kernel = 'gaussian'):
    
    # 初始化 KernelDensity 对象
    kde = KernelDensity(bandwidth = bandwidth, kernel = kernel)

    # 计算每个信号点的密度
    density_matrix = np.zeros_like(matrix)
    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            # 获取以 (i, j) 为中心的 8x8 子矩阵
            submatrix = get_submatrix(matrix, i, j)
            flattened_submatrix = submatrix.flatten().reshape(-1, 1)

            # 对数据进行拟合
            kde.fit(flattened_submatrix)

            # 计算中心点的密度
            center_point = matrix[i, j].reshape(1, -1)
            log_density = kde.score_samples(center_point)
            density = np.exp(log_density)

            # 存储密度值
            density_matrix[i, j] = density
    return density_matrix



# def calculate_density(matrix, i, j, kde, window_size = 5):
#     submatrix = get_submatrix(matrix, i, j, window_size)
#     flattened_submatrix = submatrix.flatten().reshape(-1, 1)
    
#     # 对数据进行拟合
#     kde.fit(flattened_submatrix)
    
#     # 计算中心点的密度
#     center_point = np.array([[matrix[i, j]]])
#     log_density = kde.score_samples(center_point)
#     return np.exp(log_density)

# def kernel_density(matrix, bandwidth=0.2, kernel='gaussian', window_size=5):
#     # 初始化 KernelDensity 对象
#     kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
#     size = matrix.shape[0]
    
#     # 使用并行计算来加速密度计算
#     density_matrix = np.zeros_like(matrix)
#     results = Parallel(n_jobs=-1)(delayed(calculate_density)(matrix, i, j, kde, window_size) for i in range(size) for j in range(size))
    
#     for idx, density in enumerate(results):
#         i = idx // size
#         j = idx % size
#         density_matrix[i, j] = density
    
#     return density_matrix
