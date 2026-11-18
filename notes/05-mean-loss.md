---
verification_status: pending-technical-review
title: CS336：多个位置的平均损失
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_section: lecture-02.md / Gradient accumulation：mean-reduced loss
content_scope: 课程基础补充；Python 平均值例子承接当前教学对话
created: 2026-10-30
updated: 2026-11-07
---
# 多个位置的平均损失

一段训练文本有多个预测位置，每个位置都能算出一个损失。为了用一个数代表这组预测，可以把参与训练的位置损失相加，再除以有效位置数。平均损失随后通过反向传播，合并这些位置对参数的影响。

这一章要弄清：哪些位置参与平均、分母为什么是有效位置数，以及这一个平均损失怎样汇总多处预测。

前置知识：[一个位置的交叉熵损失](04-single-token-loss.md)。

## 分子和分母要对应同一组目标

假设三个有效预测位置的损失为 [0.2,0.4,0.9]，每个位置等权。它们是已经算好的三个损失，不是同一位置三个候选 token 的概率。求和得到1.5，除以位置数3，平均损失为0.5。

![每个位置等权；平均得到一个标量，并没有减少预测位置数量。](../assets/05-mean-loss-1.svg)

每个位置等权；平均得到一个标量，并没有减少预测位置数量。

每个位置只贡献一个损失。训练可以对这个平均标量求梯度，不必为每个位置各更新一次参数。求平均中的除法也会传给梯度：平均三个损失，就是平均它们对同一参数的梯度贡献，前提是这些贡献在同一组参数上计算。

如果有填充位置或明确忽略的目标，它们的损失不进入分子，位置数也不进入分母。只扩大分母而不加入相应损失，会人为压低平均值。当前例子没有忽略位置，所以三个数全部参与。

## 平均选择了什么计数单位

本篇按有效 token 位置等权。文本长度不同时，一篇有2个有效位置、另一篇有8个，直接合并应除以10；先分别求每篇平均、再把两篇平均值等权平均，则让两篇各占一半，回答的是另一种权重设定。比较损失时应保持计数单位一致。

与原笔记的 Python 表达对应，`sum(losses)` 求出1.5，`len(losses)` 给出3，二者相除得到0.5。浮点结果有时显示成0.4000000000000001这类近似值，不表示平均规则改变。

## 类别权重会改变mean的分母

上面的有效位置数分母适用于无类别权重的token等权目标。固定PyTorch v2.9.0 CrossEntropyLoss，类别索引target、label_smoothing=0时，令V为非ignore_index目标集合，y_i为类别，w_c为类别权重，则：

$$L=\frac{\sum_{i\in V}w_{y_i}(-\log p_{i,y_i})}{\sum_{i\in V}w_{y_i}}.$$

假设两条有效目标的权重为1、3，未加权NLL为2、4，另有一条ignore_index目标。分子为1×2+3×4=14，分母为1+3=4，结果3.5；不是14/2=7，更不是14/3。忽略位置既不进分子，也不进分母。

如果两条目标各分在一个microbatch，各批weighted mean分别为2、4，简单平均得3，不能恢复3.5。正确做法是保留分子与权重和，整体14/4；或者将批均值按各批权重和的比例1/4、3/4累积。若反向传播按sum loss进行，统一使用全组有效权重和作分母，组内参数不变；结合DDP时另处理rank平均，见[梯度累积](16-gradient-accumulation.md)。

所有目标都忽略、或有效目标权重和为0时，不能按此式求均值。先判定共同分母，按训练策略一致跳过更新，避免NaN以及零梯度下动量/weight decay继续更新。权重应满足所需非负、有限口径；不能把没有定义的mean填成正常的零损失。

### 浮点soft target不是同一个带权mean

对于概率分布target q，单位置损失为-Σ_c w_c q_c log p_c，但reduction=mean按位置数平均，不按Σ_c w_c q_c归一化。即使q为one-hot，带权mean也可能与整数target不同：上述两条改为浮点one-hot后，sum仍为14，但mean为14/2=7。ignore_index仅适用于类别索引target；soft target要自行明确mask和分母。这里不讨论label_smoothing与类别权重同时启用的额外细节。

### soft target的显式mask例子

继续用浮点one-hot形式的两条有效目标，未归一化的带权损失分别为2、12。第三条是padding，先给它合法的有限logits和占位概率分布，再以reduction=none求每位置损失ℓ_i，使用mask m=[1,1,0]。若定义“有效位置等权的soft-target目标”，应计算Σ_i m_i ℓ_i/Σ_i m_i=14/2=7。先乘mask再对全部三个位置mean只得到14/3，错误地把padding放进分母。不要用ignore_index整数去填概率向量，也不能让padding先产生NaN再期待乘0消掉它。

如果明确要复现这两个整数标签的带权均值，则另定义分母D=Σ_i m_i Σ_c w_c q_{i,c}=1+3=4，结果才是14/4=3.5；这是自己选择的权重归一化目标，不是概率target分支默认的mean。跨microbatch应分别累积分子与所选分母，再整体相除；全组mask和或D为0时应按相应策略跳过更新。先决定要保持哪个目标，才能判断分母2还是4。

练习：权重和分别为2、6的两批，其weighted mean为1、5，整体weighted mean是多少？答案：(2×1+6×5)/(2+6)=4，而批等权平均是3。

来源：[v2.9.0 CrossEntropyLoss两类target的公式](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/modules/loss.py)、[LossNLL.cpp实现](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/LossNLL.cpp)。B起草带权mean、跨批次聚合与soft-target区别，A补写显式mask和两种分母算例。14/4=3.5、14/2=7和练习4已独立算术复核，未运行PyTorch训练。相关[问答#12](https://github.com/lw-liuwang/cs336-notes/discussions/12)；本节复核不等于整篇导入材料全部验证。

## 小练习

三个有效位置的损失为0.2、0.4、0.6，另外两个填充位置不参与训练。平均损失是多少，分母应为3还是5？

### 参考答案

平均损失为 $(0.2+0.4+0.6)/3=0.4$。分母为3，填充位置既不贡献损失，也不计入有效位置数；除以5会把损失错误地压低。


## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
