# Genomic TAD identification using YOLOv5
## 为什么使用YOLOv5识别TAD？
Hi-C 矩阵（或其衍生矩阵，比如 .laplacian.bedpe → contact map）其实就是一张图像：
* X、Y 轴分别是基因组位置；
* 每个像素点的值是两个基因组片段之间的相互作用频率；
* 在可视化中，TAD 表现为沿主对角线的一个高信号方块块（triangle/square block）；
* 任务目标：识别这些沿对角线的“方块”及其边界。
因为 TAD 在 Hi-C contact map 中具有“图像目标”特征，YOLOv5 能自动学习其几何模式与边界。

## Step1. 图像经过高斯和拉普拉斯平滑
使用这种处理方式，图像的TAD特征会更加凸显，方便后续的识别。
```{shell}
python 0.run.gaussian.laplacian.sigma1truncate3.genomewide.py
```
![GL](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/G_L.png)

## Step2. 图像分割
### 该部分的作用
1. 把这些相互作用数据映射到一个二维矩阵；
2. 对矩阵按 5 kb 分辨率栅格化；
3. 按滑动窗口切成多个局部方阵；
4. 每个局部矩阵标准化到 0–255 灰度；
5. 保存为 .png 图像。

```{shell}
python 1.demo_HSIcutImageV2.py
```
![cutImg](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/cutImg.png)

## Step3. 训练标签的标注
使用labelimg进行TAD标签的标注，在labelimg中指定TAD标签标号(1,2,3...)和颜色，保存为每张图片中该标签的位置信息，是一个个的txt文件，和对应的图片一同放置于训练数据文件夹下。

## Step4. 训练
### 训练注意事项
使用YOLOv5的训练脚本进行训练，但是这里需要根据自己的数据进行一些调整。<br>
特别需要注意，需要在自己的 yaml 文件中指定train, val, test的路径。
![train](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/train.png)

### 训练batch
```{shell}
python 2.train_demo.py
```
![batch](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/batch.png)

### 训练结果
![loss](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/scr/runs/train/LoadingDomain.V6/results.png)

## Step5. 验证
### 流程
```{shell}
python 3.val_demo.py
```
1. 加载模型与数据集
* 读入 --weights（如 .../runs/train/LoadingDomain.V6/weights/best.pt）。
* 读入 --data（你的 LoadingDomain.yaml，里面给出 val 图片列表/目录、nc、names）。
2. 构建验证 dataloader
* 使用 create_dataloader(..., rect=True, pad=0.5) 保持长宽比评估。
* 默认三通道输入（BGR/RGB），尺寸由 --imgsz 控制（480）。
3. 前向 & NMS
* model(img) 得到预测，再做 non_max_suppression（阈值由 --conf-thres、--iou-thres）。
4. 与GT匹配 & 统计
* 计算每张图的正确匹配（IoU 阈值序列 0.50–0.95），累计得到 P/R、mAP@0.5、mAP@0.5:0.95。
* 生成混淆矩阵、可选保存 txt/json 预测结果和标注-预测对比图。

### 参数
* --data：数据集 YAML（你的 LoadingDomain.yaml）。
* --weights：待评估权重（通常是训练产出的 best.pt 或 last.pt）。
* --imgsz 480：验证分辨率；要与训练时设置基本一致。
* --conf-thres 0.01：评分阶段通常设很低，交由 NMS + AP 曲线综合评估。
* --iou-thres 0.5：NMS 的 IoU 阈值（不是 mAP 的 IoU），默认 0.5 合理。
* --batch-size 16：按显存调。
* --half：FP16 推理（仅 CUDA）。
* --single-cls：如果你的数据集就是单类别检测（例如只检测 “LoadingDomain”），可以打开；否则在 YAML 里 nc/names 要与真实类别数一致。
* --save-txt / --save-json / --save-conf：是否把预测保存成 txt/COCO-JSON，及是否附带置信度。
* --augment：测试时多尺度/翻转增强（通常关闭，保持可比性）。

### 验证batch
![val_batch](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/val_batch.png)

### 验证召回率
![recall](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/scr/runs/val/exp/PR_curve.png)

## Step6. 测试代码
这一部分代码分为在CPU上运行的版本和在GPU上运行的版本。
### 运行
```{shell}
python 4.Demo_testing_hic_CPU.py
python 4.Demo_testing_hic_GPU.py
```
## Step7. 预测TAD
### 关键流程
1. 读入 bedpe → generate_mat 把 (row_bin, col_bin) 放到 WT_signal_mat（按 5kb / bin；>6 截断）。
2. generate_intervals 产生每条染色体的子块区间（50-bin 步长，向右扩 overlap_len=5）。
3. 每个子块：
* 过滤空块（flags_sub），确定右边列终点 max_ii_col，得到方阵 WT_signal_mat_i；
* 对称化（上三角→对称），按染色体全局 min/max 归一化到 0-255；
* letterbox(..., 480) 得到网络输入，前向、NMS；
* 把检测框坐标按 5kb bin → 基因组坐标；
4. 邻近子块之间，按 IoU>0.3 合并（regionBoxDeRep）；
5. 汇总保存到 Results/Loading_domain_WT_SMC1A.V6.bedpe。
6. 产生的结果为每个TAD的位置，在Contact map中是一个个的方框的形式存在，可以结合JuiceBox进行效果的评估。
```{shell}
python 5.Demo_detect_hic_CPU_multiprocess.py
```
