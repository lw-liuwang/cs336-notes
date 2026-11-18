# LayerNorm：先确定归一化轴，再计算方差

LayerNorm对输入的指定末尾特征轴求均值与方差，再作仿射变换。它不是把一批文本混在一起求总体统计，也不是自动使用N-1的样本方差。

本文为B起草、A补写的课程基础补充，来自固定PyTorch v2.9.0文档阅读与代数推导；不属于Coral导入材料，也不改原始provenance.json。

## 同一组[1,3]，两个分母给出不同变换

设gamma=1、beta=0，先令eps=0作数学比较。均值为2，平方偏差和为2：

| 方差定义 | 方差值 | 归一化结果 |
| --- | --- | --- |
| 除以N=2 | 1 | [-1,1] |
| 除以N-1=1 | 2 | [-1/sqrt(2),1/sqrt(2)] |

PyTorch LayerNorm采用除以N的定义，相当于var(correction=0)，固定版本文档也写成unbiased=False。目的是按当前特征定义变换，不是从一组独立观测构造未知总体方差的无偏估计。N-1不是对LayerNorm的通用“纠错”。

加入eps=1e-5后，输出为[-1/sqrt(1+1e-5),1/sqrt(1+1e-5)]≈[-.999995,.999995]。eps放在方差加法后开平方，不是sqrt(var)+eps。eps=0只是本例手算设定，常量特征或单元素时会有0/0；真实实现应保留合适的正eps。

## eps使输出方差不一定等于1

先保持gamma=1、beta=0。若归一化轴上的输入方差为v，归一化后按相同除以N口径计算的方差为v/(v+eps)，因为中心化不改变方差、除以sqrt(v+eps)会将方差除以v+eps。v>0且eps>0时，该值严格小于1；这与采用N还是N-1是两个不同问题。

近常量输入[.999,1.001]均值为1，平方偏差和为2×10^(-6)，所以v=10^(-6)。设置eps=10^(-5)，分母为sqrt(11×10^(-6))，两个输出为[-1/sqrt(11),1/sqrt(11)]，输出方差仅为1/11，约.090909，而不是1。若随后两个位置的gamma均为2、beta均为3，输出均值变成3、方差变成4/11；可学习仿射也不保证最终方差为1。

对前面的[1,3]，v=1，eps=1e-5时输出方差为1/(1+1e-5)≈.9999900001；对常量[1,1]，v=0且eps>0时输出全为0、方差0。eps在输入变化很小时尤其显著，不能把“归一化”理解成对任何输入都严格得到单位方差。

## normalized_shape同时决定统计轴与参数形状

对输入[B,T,H]，normalized_shape=H沿最后一个轴归一化；每个(b,t)位置独立计算H个特征的均值/方差。默认elementwise_affine=True、bias=True时，gamma/beta均为[H]，总计2H个标量参数，B和T不影响参数数量。

若改为normalized_shape=(T,H)，则沿最后两个轴共同归一化，每个batch样本只有一组统计量，gamma/beta形状变成[T,H]，总计2TH个标量。输入末尾维度必须与normalized_shape匹配。batch维通常不参加统计，但如果人为将其包含进所有末尾归一化轴，语义就改变了，不能仅看“LayerNorm”名字推断轴。

构造B=1、T=2、H=2的输入，两行分别为[1,3]、[5,9]。设eps=0且仿射为恒等：

- normalized_shape=2：两行均值为2、7，方差1、4，各自都输出[-1,1]。
- normalized_shape=(2,2)：四元素均值4.5，平方偏差和35，方差35/4=8.75；输出按原位置为[-3.5,-1.5,.5,4.5]/sqrt(8.75)，不再是逐行[-1,1]。

默认仿射下，两种配置参数总数分别为4和8。改变normalized_shape既改变统计范围也改变参数量，不能只改其中一个解释。bias=False会去掉beta；elementwise_affine=False会去掉这两套可学习仿射参数。

## 训练和推理的统计方式

LayerNorm在train/eval两种模式均使用当前输入的这些统计量，没有BatchNorm式的running mean/variance。这个结论不能推广成“模型切到eval不影响任何结果”，因为同一模型的dropout等其他模块仍会变化。

练习：输入[B,4,8]，normalized_shape=8与(4,8)在默认仿射下各有多少参数？答案：16与64；B都不参与这两种配置的参数计数。

来源：[PyTorch v2.9.0 normalization.py的LayerNorm文档与实现](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/modules/normalization.py)，已实际阅读归一化轴、biased variance、affine和train/eval说明。B起草方差分母、轴和参数计数，A补写eps影响输出方差及近常量反例。上述平方和、方差与参数量已独立算术核对，未运行PyTorch算子或训练实验。相关[技术问答#18](https://github.com/lw-liuwang/cs336-notes/discussions/18)。本篇由Codex辅助整理。
