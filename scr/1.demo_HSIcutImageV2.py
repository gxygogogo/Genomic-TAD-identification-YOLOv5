import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import math
plt.ioff()
# import fanc
# import fanc.plotting 
# from scipy import ndimage as ndi
# import matplotlib.patches as patches
# from scipy.ndimage import zoom
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy import ndimage as ndi
# from skimage import restoration
# from skimage.morphology import square, closing
# import skimage.filters as filters
# from skimage.measure import label, regionprops
# from numpy import inf
# from chess.dbscan import dbscan, NOISE
# from scipy.stats import boxcox, mode
# from scipy.spatial.distance import cdist
# import random
# from collections import Counter
# from scipy.sparse import csr_matrix
# from scipy.io import savemat

from PIL import Image

#from sklearn.ensemble import IsolationForest
import os

def generate_mat(dataframe_tmp, data_mat, min_value, interal=5000):
    for indexi, data_i in dataframe_tmp.iterrows():
        value1 = data_i[6]
        row_ind = math.ceil(data_i[2] / interal) - min_value
        col_ind = math.ceil(data_i[5] / interal) - min_value
        data_mat[row_ind, col_ind] = value1
        # data_mat[col_ind, row_ind] = value1
    return data_mat

def generate_intervals(max_num, min_num, initial_interval, increase_step):
    segments = []
    current_start = min_num

    while current_start < max_num:
        current_end = current_start + initial_interval
        segments.append((current_start, current_end))

        current_start = current_end
        initial_interval += increase_step

    return segments


#plt.rcParams['text.usetex'] = True
# plt.style.use(['dark_background'])
# plt.style.use('classic')

prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib.colors import ListedColormap,LinearSegmentedColormap
#import cv2
colors = ['#000000','#FF0000','#FFD100']
new_cmap = ListedColormap(['#008000','#FFFFFF','#FF0000'])
cmap_germany = LinearSegmentedColormap.from_list('germany',colors,N=256)
cmap = ListedColormap(['white','black'])

winsize = "5kb"
wdir = "scr/data/laplacian/"


# gene_name= "WT"
gene_name= "WT_SMC1A"
# KO_signal_file = "SA2KO_DBSCANv3_pvalue.bedpe"
# KO2_signal_all = pd.read_csv(wdir + KO_signal_file, sep='\t', header=None)
# WT_signal_file = "WT_DBSCANv3_pvalue.bedpe"
WT_signal_file = gene_name + ".gaussian_sigma1truncate3.laplacian.bedpe"
WT_signal_all = pd.read_csv(wdir + WT_signal_file, sep='\t', header=None)


processed_path = "scr/recutHICs_LoadingD"

target_folder = 'scr/recutHICs_LoadingD'  # 替换为你的目标文件夹路径

chrnames = WT_signal_all[0].unique()
WT_df00_all = pd.DataFrame()
KO1_df00_all = pd.DataFrame()
KO2_df00_all = pd.DataFrame()
dense_df_all = pd.DataFrame()
dense_df_regionBox_all = pd.DataFrame()

chrnames=['chr1', 'chr2', 'chr3']

for chrname in chrnames:
    print(f'###### {chrname}')
    save_dir = os.path.join(target_folder, f'{gene_name}_{chrname}')
    # 确保目标文件夹存在
    os.makedirs(save_dir, exist_ok=True)

    dense_df_chri = pd.DataFrame()
    WT_signal = WT_signal_all[(WT_signal_all[0]==chrname)&(WT_signal_all[3]==chrname)]
    # KO2_signal = KO2_signal_all[(KO2_signal_all[0]==chrname)&(KO2_signal_all[3]==chrname)]
    # KO1_signal = KO1_signal_all[(KO1_signal_all[0]==chrname)&(KO1_signal_all[3]==chrname)]
    # CLS_signal = CLS_signal_all[(CLS_signal_all[0]==chrname)&(CLS_signal_all[3]==chrname)]
    # CTCF_signal = CTCF_signal_all[(CTCF_signal_all[0]==chrname)&(CTCF_signal_all[3]==chrname)]

    num1 = WT_signal.iloc[:,2].max()
    num2 = WT_signal.iloc[:,5].max()
    # num3 = CTCF_signal.iloc[:,2].max()
    # num4 = CTCF_signal.iloc[:,5].max()
    # num5 = CLS_signal.iloc[:,2].max()
    # num6 = CLS_signal.iloc[:,5].max()
    # num3 = KO2_signal.iloc[:,2].max()
    # num4 = KO2_signal.iloc[:,5].max()
    # num5 = KO1_signal.iloc[:,2].max()
    # num6 = KO1_signal.iloc[:,5].max()

    # max_num = math.ceil(max([num1,num2,num3,num4,num5,num6])/5000)
    # max_num = math.ceil(max([num1,num2,num3,num4,num5,num6])/5000)+5
    max_num = math.ceil(max([num1,num2])/5000)+5
    min_num = 0

    overlap_len = 5
    # interval_set = 400
    # filtered_regionID = np.array([(i, i+interval_set) for i in range(min_num,max_num, interval_set)])
    initial_interval = 50
    increase_step=0
    filtered_regionID = generate_intervals(max_num, min_num, initial_interval, increase_step)
    filtered_regionID = np.array(filtered_regionID)

    if filtered_regionID[-1][-1]>(max_num+overlap_len):
        filtered_regionID[-1][-1]=max_num
    max_num = filtered_regionID[-1][-1]+overlap_len

    # KO2_signal_mat = (np.zeros((max_num - min_num + 1, max_num - min_num + 1)))
    # KO1_signal_mat = (np.zeros((max_num - min_num + 1, max_num - min_num + 1)))
    WT_signal_mat = (np.zeros((max_num - min_num + 1, max_num - min_num + 1)))

    # if len(KO_signal):
    #     KO_signal_mat = generate_mat(KO_signal_id,KO_signal_mat,min_num, datatrans = 'log')
    if len(WT_signal):
        WT_signal_mat = generate_mat(WT_signal,WT_signal_mat,min_num)

    WT_signal_mat2 = WT_signal_mat.copy()
    WT_signal_mat2[WT_signal_mat2<=1]=0

    WT_signal_mat[WT_signal_mat > 6] = 6

    # filtered_regionID = np.array([ele for ele in regions_ID if ele.split('_')[0] == chrname])
    print('chr{} have total regions: {}'.format(chrname, len(filtered_regionID)))
    dense_df00_i = pd.DataFrame()
    dense_df00_i_1 = pd.DataFrame()
    R_count = 0
    max_cls_set=0
    last_end_ind = 0
    for R_ind,region_id in enumerate(filtered_regionID):
        print('now process chr{}_region{}'.format(chrname, R_ind))

        max_num_i = region_id[1]+overlap_len
        min_num_i = region_id[0]

        if max_num_i<last_end_ind:
            continue

        # if region_id[0]<16400 and region_id[1]>16400:
        #     print('debug')

        min_ii = min_num_i - min_num
        max_ii = max_num_i - min_num

        submat_all = WT_signal_mat[min_ii:max_ii + 1, min_ii:]
        flags_sub = np.sum(submat_all > 0, axis=0)
        nonzero_indices = np.nonzero(flags_sub)[0]
        if not nonzero_indices.size:
            continue
        submat_all2 = WT_signal_mat2[min_ii:max_ii + 1, min_ii:]
        flags_sub2 = np.sum(submat_all2 > 0, axis=0)
        nonzero_indices2 = np.nonzero(flags_sub2)[0]
        if nonzero_indices2.size:
            max_ii_col1 = np.max(nonzero_indices2)+min_ii
        else:
            max_ii_col1 = min_ii
        if (max_ii_col1+5) <= WT_signal_mat.shape[1]:
            max_ii_col1 = max_ii_col1 + 5

        max_ii_col1 = max(max_ii_col1,max_ii)
        max_ii_col = min(max_ii_col1,min_ii+400)

        WT_signal_mat_i = WT_signal_mat[min_ii:max_ii_col + 1, min_ii:max_ii_col + 1]  # 假设A是一个稀疏矩阵，这里使用稀疏矩阵的坐标和值的表示方式
        rows_WT, cols_WT = np.nonzero(WT_signal_mat_i)
        # KO2_signal_mat_i = KO2_signal_mat[min_ii:max_ii + 1, min_ii:max_ii_col + 1]  # 假设A是一个稀疏矩阵，这里使用稀疏矩阵的坐标和值的表示方式
        # rows_KO, cols_KO = np.nonzero(KO2_signal_mat_i)

        # 获取上三角部分（不包括对角线）
        upper_triangle = np.triu(WT_signal_mat_i, k=1)

        # 将上三角部分复制到下三角部分
        WT_signal_mat_i = upper_triangle + upper_triangle.T + np.diag(np.diag(WT_signal_mat_i))

        last_end_ind = max_ii_col

        interal = 5000
        end1s = interal * min_num_i
        if end1s==0:
            start1s=0
        else:
            start1s = end1s - interal + 1
        end2s = interal * (max_ii_col + min_num)
        start2s = end2s - interal + 1
        # np.save(os.path.join(processed_path, f'{chrname}_5kb_{start1s}_{end2s}.npy'), WT_signal_mat_i)

        # jpg_mat = WT_signal_mat_i.copy()
        # 检查矩阵是否为方阵
        if WT_signal_mat_i.shape[0] != WT_signal_mat_i.shape[1]:
            print(f"文件 {chrname}_5kb_{start1s}_{end2s}.jpg 中的矩阵不是方阵，跳过此文件")
            continue

        # min_val = np.min(WT_signal_mat_i)
        # max_val = np.max(WT_signal_mat_i)

        min_val = np.min(WT_signal_mat)
        max_val = np.max(WT_signal_mat)

        # 保存修改后的矩阵到目标文件夹中的 .npy 文件
        # np.save(os.path.join(save_dir, f'{chrname}_5kb_{start1s}_{end2s}.jpg'), jpg_mat)
        # img = Image.fromarray((255 * (WT_signal_mat_i - min_val) / (max_val - min_val)).astype(np.uint8))  # 转换为8位图像（0-255）
        # img = Image.fromarray((jpg_mat).astype(np.uint8))  # 转换为8位图像（0-255）
        # img.save(os.path.join(save_dir, f'{chrname}_5kb_{start1s}_{end2s}.jpg'), 'JPEG')

        ## 修改为png
        img = Image.fromarray((255 * (WT_signal_mat_i - min_val) / (max_val - min_val)).astype(np.uint8))  # 转换为8位图像（0-255）
        img.save(os.path.join(save_dir, f'{chrname}_5kb_{start1s}_{end2s}.png'), 'PNG')


