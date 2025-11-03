import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import math
plt.ioff()
from collections import Counter
import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
def generate_mat(dataframe_tmp, data_mat, min_value, interal=5000):
    for indexi, data_i in dataframe_tmp.iterrows():
        value1 = data_i[6]
        row_ind = math.ceil(data_i[2] / interal) - min_value
        col_ind = math.ceil(data_i[5] / interal) - min_value
        data_mat[row_ind, col_ind] = value1
        # data_mat[col_ind, row_ind] = value1
    return data_mat

def filtering_noiseP(cls_results,coords_list,diff_matrix):
    new_cls_results = []
    coords_list2=[]
    for i, coord in enumerate(coords_list):
        row, col = coord
        if diff_matrix[row, col]!=0:
            coords_list2.append([row, col])
            new_cls_results.append(cls_results[i])

    return new_cls_results,coords_list2

def merge_fun(dense_df00_i, dense_df00_i_1, max_cls_set):
    # dense_df00_i['idx'] = dense_df00_i['chrom1']+"_"+dense_df00_i['start1'].astype(str)+"_"+dense_df00_i['start2'].astype(str)
    # dense_df00_i_1['idx'] = dense_df00_i_1['chrom1']+"_"+dense_df00_i_1['start1'].astype(str)+"_"+dense_df00_i_1['start2'].astype(str)
    unique_cls = dense_df00_i['cls'].unique()
    unique_cls = unique_cls[unique_cls!=-1]
    for cls_i in unique_cls:
        dense_i_part = dense_df00_i[dense_df00_i['cls']==cls_i]
        dense_i_1_part = dense_df00_i_1[dense_df00_i_1['idx'].isin(dense_i_part['idx'])]
        if len(dense_i_1_part):
            counter = Counter(dense_i_1_part['cls'])
            cls_m, _ = counter.most_common(1)[0]
            # unique_cls2 = dense_i_1_part['cls'].unique()
            # dense_i_1_part = dense_df00_i_1[dense_df00_i_1['cls'].isin(unique_cls2) & dense_df00_i_1['cls'].duplicated(keep=False)]
            dense_i_1_part_re = dense_df00_i_1[dense_df00_i_1['cls']==cls_m]
            if len(dense_i_1_part_re)<len(dense_i_part):
                dense_df00_i_1 = dense_df00_i_1[dense_df00_i_1['cls'] != cls_m]
                if len(dense_i_1_part_re)>len(dense_i_1_part):
                    dense_i_add = dense_i_1_part_re[~dense_i_1_part_re['idx'].isin(dense_i_part['idx'])]
                    dense_i_add['cls']=cls_i
                    dense_df00_i=pd.concat([dense_df00_i,dense_i_add],ignore_index=True)
            else:
                dense_df00_i = dense_df00_i[dense_df00_i['cls'] != cls_i]

    if len(dense_df00_i_1):
        max_i_1 = dense_df00_i_1['cls'].max()
    else:
        max_i_1 = max_cls_set
    dense_df00_i['cls'] = dense_df00_i['cls'] + max_i_1
    return dense_df00_i, dense_df00_i_1

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def regionBox_generate(WT_sig_mat,chrname,min_num):

    ## detect: (x1,y1)   (x2,y2)
    ###
    WT = WT_sig_mat
    WT[WT > 6] = 6
    # WT = WT_sig_mat.astype(np.float32)
    upper_triangle = np.triu(WT, k=1)
    # 将上三角部分复制到下三角部分
    WT = upper_triangle + upper_triangle.T + np.diag(np.diag(WT))
    min_val = np.min(WT)
    max_val = np.max(WT)
    WT = ((WT - min_val) / (max_val - min_val)).astype(np.float32)  # 转换为8位图像（0-255）
    w='/public1/xinyu/CohesinProject/DeepLearning_Cohesin/Loading_domain_yolo/runs/train/exp21/weights/best.pt'
    weights = w
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    model = torch.jit.load(w, map_location=device) if 'torchscript' in w else attempt_load(weights, map_location=device)

    #model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location='cpu')
    stride = int(model.stride.max())
    WT1 = letterbox(WT, 480, stride=stride, auto=True)[0]
    WT1 = WT1.reshape(1, WT1.shape[0], WT1.shape[1])
    WT1 = np.ascontiguousarray(WT1)
    WT1 = torch.tensor(WT1)
    WT1 = WT1.unsqueeze(0)
    pred = model(WT1, augment=False, visualize=False).data.cpu().numpy()[0]
    pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)
    dense_regionBox_all = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'cls'])
    for i, det in enumerate(pred):
        print(len(det))
        if len(det):
            det[:, :4] = scale_coords(WT1.shape[2:], det[:, :4], WT.shape).round()
            det = det.numpy()
            print(det)
            for i in range(len(det)):
            # Rescale boxes from img_size to im0 size
                print(det[i])
                x1 = det[i][0]
                y1 = det[i][1]
                x2 = det[i][2]
                y2 = det[i][3]
                cls = det[i][5]
                # if cls == 0:
                #     x1 = y1 = min(x1, y1)
                interal = 5000
                end1_row = interal*(y1+min_num)
                if end1_row==0:
                    start1_row=0
                else:
                    start1_row = end1_row - interal + 1
                end1_col = interal*(x1+min_num)
                if end1_col==0:
                    start1_col=0
                else:
                    start1_col = end1_col - interal + 1


                end2_row = interal*(y2+min_num)
                start2_row = end2_row-interal+1
                end2_col = interal*(x2+min_num)
                start2_col = end2_col-interal+1

                box_x1 = start1_row
                box_x2 = end2_row
                box_y1 = start1_col
                box_y2 = end2_col


                new_row = {'chrom1': chrname, 'start1': box_x1, 'end1': box_x2, 'chrom2': chrname, 'start2': box_y1, 'end2': box_y2, 'cls': 1}
                area_new = (box_x2-box_x1+1)*(box_y2-box_y1+1)/5000/5000
                # if area_new>=4:
                dense_regionBox_all.loc[len(dense_regionBox_all)] = new_row
    return dense_regionBox_all

# IOU 计算函数
def calculate_iou(box1, box2):
    x1_max = max(box1['start1'], box2['start1'])
    y1_max = max(box1['start2'], box2['start2'])
    x2_min = min(box1['end1'], box2['end1'])
    y2_min = min(box1['end2'], box2['end2'])

    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    box1_area = (box1['end1'] - box1['start1']) * (box1['end2'] - box1['start2'])
    box2_area = (box2['end1'] - box2['start1']) * (box2['end2'] - box2['start2'])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# 合并重叠检测框函数
def merge_boxes(box1, box2):
    return {
        'chrom1': box1['chrom1'],
        'start1': min(box1['start1'], box2['start1']),
        'end1': max(box1['end1'], box2['end1']),
        'chrom2': box1['chrom2'],
        'start2': min(box1['start2'], box2['start2']),
        'end2': max(box1['end2'], box2['end2']),
        'cls': box1['cls']
    }

def regionBoxDeRep(df_curr,df_prev):
    # 阈值
    iou_threshold = 0.3

    # 复制 df_prev 以避免修改原始数据
    df_prev_copy = df_prev.copy()

    # 用于存储未重合的检测框
    new_boxes = []

    # 用于存储需要删除的当前检测框的索引
    delete_indices = []

    for curr_index, curr_box in df_curr.iterrows():
        merged = False
        for prev_index, prev_box in df_prev_copy.iterrows():
            iou = calculate_iou(curr_box, prev_box)
            if iou > iou_threshold:
                merged_box = merge_boxes(curr_box, prev_box)
                df_prev_copy.loc[prev_index] = merged_box
                merged = True
                delete_indices.append(curr_index)
                break
        if not merged:
            new_boxes.append(curr_box)

    # 删除当前检测框中重合的检测框
    df_curr.drop(delete_indices, inplace=True)

    # 将未重合的检测框添加到 df_prev_copy 中
    # df_prev_copy = pd.concat([df_prev_copy, pd.DataFrame(new_boxes, columns=columns)], ignore_index=True)
    return df_curr,df_prev_copy


def regionBox_enlarge(dense_box,OriSig_mat,CTCF_mat,min_num):
    chrnames = dense_box['chrom1'].unique()
    dense_regionBox_all = pd.DataFrame(columns = ['chrom1','start1','end1','chrom2','start2','end2','cls'])

    interal = 5000
    for chrname in chrnames:
        dense_box_chri = dense_box[dense_box['chrom1']==chrname]
        for indexi, data_i in dense_box_chri.iterrows():
            box_x1 = (data_i[1]-1+interal)/interal-min_num
            box_x2 = data_i[2]/interal-min_num
            box_y1 = (data_i[4]-1+interal)/interal-min_num
            box_y2 = data_i[5]/interal-min_num
            box_x1 = int(box_x1)
            box_x2 = int(box_x2)
            box_y1 = int(box_y1)
            box_y2 = int(box_y2)

            rows_len = box_x2 - box_x1 + 1
            cols_len = (box_y2 - box_y1 + 1)
            area_new = (box_x2 - box_x1 + 1) * (box_y2 - box_y1 + 1)
            while(rows_len<7 or cols_len<7):
                flag_row1=0
                flag_row2 = 0
                flag_col1=0
                flag_col2 = 0
                if rows_len<7:
                    flag_row1 = np.sum(OriSig_mat[box_x1-1,box_y1:box_y2+1])
                    flag_row2 = np.sum(OriSig_mat[box_x2+1,box_y1:box_y2+1])
                    if flag_row1:
                        box_x1 = box_x1-1
                    if flag_row2:
                        box_x2 = box_x2+1

                if cols_len<7:
                    flag_col1 = np.sum(OriSig_mat[box_x1:box_x2+1,box_y1-1])
                    flag_col2 = np.sum(OriSig_mat[box_x1:box_x2+1,box_y2+1])
                    if flag_col1:
                        box_y1 = box_y1-1
                    if flag_col2:
                        box_y2 = box_y2+1
                box_x1 = int(box_x1)
                box_x2 = int(box_x2)
                box_y1 = int(box_y1)
                box_y2 = int(box_y2)
                rows_len = box_x2 - box_x1 + 1
                cols_len = (box_y2 - box_y1 + 1)
                if (flag_row1==0 and flag_row2==0 and cols_len>=5) or (flag_col1==0 and flag_col2==0 and rows_len>=5):
                    break
                if (flag_row1==0 and flag_row2==0) and (flag_col1==0 and flag_col2==0):
                    break

            # box_x1 = int(box_x1)-1
            # box_x2 = int(box_x2)+1
            # box_y1 = int(box_y1)-1
            # box_y2 = int(box_y2)+1
            # if ((rows_len >= 4 and cols_len >= 6) or (rows_len >= 6 and cols_len >= 4) or (
            #         rows_len >= 5 and cols_len >= 5)) and (
            #         np.sum(CTCF_mat[box_x1:box_x2 + 1, box_y1:box_y2 + 1]) > 0):
            if ((rows_len >= 3 and cols_len >= 5) or (rows_len >= 5 and cols_len >= 3) or (
                    rows_len >= 4 and cols_len >= 4)) and (
                    np.sum(CTCF_mat[box_x1:box_x2 + 1, box_y1:box_y2 + 1]) > 0):
                end1s = interal * (box_x2 + min_num)
                start1s = interal * (box_x1 + min_num) - interal + 1
                end2s = interal * (box_y2 + min_num)
                start2s = interal * (box_y1 + min_num) - interal + 1

                new_row = {'chrom1': chrname, 'start1': start1s, 'end1': end1s, 'chrom2': chrname, 'start2': start2s, 'end2': end2s, 'cls': 1}
                dense_regionBox_all.loc[len(dense_regionBox_all)] = new_row
    return dense_regionBox_all

def deReplicating(WT_df_all):
    # WT_df_all['idx'] = WT_df_all['chrom1']+"_"+WT_df_all['start1'].astype(str)+"_"+WT_df_all['start2'].astype(str)
    WT_df_all_new = WT_df_all.groupby('idx').agg({'chrom1':'first','start1':'first','end1':'first',
                                                  'chrom2':'first','start2':'first','end2':'first',
                                                  'cls':'first'}).reset_index()
    return WT_df_all_new

def generate_intervals(max_num, min_num, initial_interval, increase_step):
    segments = []
    current_start = min_num

    while current_start < max_num:
        current_end = current_start + initial_interval
        segments.append((current_start, current_end))

        current_start = current_end
        initial_interval += increase_step

    return segments



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
import cv2
colors = ['#000000','#FF0000','#FFD100']
new_cmap = ListedColormap(['#008000','#FFFFFF','#FF0000'])
cmap_germany = LinearSegmentedColormap.from_list('germany',colors,N=256)
cmap = ListedColormap(['white','black'])


# KO_signal_file = "SA2KO_DBSCANv3_pvalue.bedpe"
# KO2_signal_all = pd.read_csv(wdir + KO_signal_file, sep='\t', header=None)
# WT_signal_file = "WT_DBSCANv3_pvalue.bedpe"
WT_signal_file = "/public1/xinyu/CohesinProject/DeepLearning_Cohesin/Loading_domain_yolo/hic/WT_SMC1A.gaussian_sigma1truncate3.laplacian.bedpe"
WT_signal_all = pd.read_csv(WT_signal_file, sep='\t', header=None)


processed_path = "./cutHICs"

chrnames = WT_signal_all[0].unique()
WT_df00_all = pd.DataFrame()
KO1_df00_all = pd.DataFrame()
KO2_df00_all = pd.DataFrame()
dense_df_all = pd.DataFrame()
dense_df_regionBox_all = pd.DataFrame()

# chrnames=['chr2']

for chrname in chrnames:
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
    # filtered_regionID = np.array([ele for ele in regions_ID if ele.split('_')[0] == chrname])
    print('chr{} have total regions: {}'.format(chrname, len(filtered_regionID)))
    dense_df00_i = pd.DataFrame()
    dense_df00_i_1 = pd.DataFrame()
    R_count = 0
    max_cls_set=0
    last_end_ind = 0
    iter_flag = 0
    # filtered_regionID = filtered_regionID[np.where((filtered_regionID[:, 0] > 4280) & (filtered_regionID[:, 0] < 4550))]
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
        if (max_ii_col1+5)<=WT_signal_mat.shape[1]:
            max_ii_col1 = max_ii_col1 + 5

        max_ii_col1 = max(max_ii_col1,max_ii)
        max_ii_col = min(max_ii_col1,min_ii+400)

        WT_signal_mat_i = WT_signal_mat[min_ii:max_ii_col + 1, min_ii:max_ii_col + 1]  # 假设A是一个稀疏矩阵，这里使用稀疏矩阵的坐标和值的表示方式
        rows_WT, cols_WT = np.nonzero(WT_signal_mat_i)
        # KO2_signal_mat_i = KO2_signal_mat[min_ii:max_ii + 1, min_ii:max_ii_col + 1]  # 假设A是一个稀疏矩阵，这里使用稀疏矩阵的坐标和值的表示方式
        # rows_KO, cols_KO = np.nonzero(KO2_signal_mat_i)
        rows, cols = WT_signal_mat_i.shape
        print(WT_signal_mat_i.shape)
        if rows!=cols:
            continue
        if len(rows_WT):
            dense_df_regionBox = regionBox_generate(WT_signal_mat_i,chrname,min_num_i)
            if len(dense_df_regionBox):
                iter_flag = iter_flag+1
                if iter_flag>1:
                    dense_df_regionBox,dense_df_regionBox_pre = regionBoxDeRep(dense_df_regionBox,dense_df_regionBox_pre)
                    dense_df_regionBox_all = pd.concat([dense_df_regionBox_all, dense_df_regionBox_pre], ignore_index=True)
                dense_df_regionBox_pre = dense_df_regionBox.copy()
    dense_df_regionBox_all = pd.concat([dense_df_regionBox_all, dense_df_regionBox_pre], ignore_index=True)
dense_df_regionBox_all.to_csv('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/Loading_domain_yolo/Dense_region_WT_SMC1A_20240828_v2.bedpe', sep='\t', index=False, header=False)

