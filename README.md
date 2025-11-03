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
[!GL]('https://github.com/gxygogogo/Genomic-TAD-identification-YOLOv5/blob/main/img/G_L.png')

## Step2. 图像分割
