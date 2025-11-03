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
![batch](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/batch.png)

### 训练结果
![loss](https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/batch.png)

