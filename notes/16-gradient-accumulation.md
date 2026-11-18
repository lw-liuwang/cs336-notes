---
verification_status: pending-technical-review
title: CS336：用梯度累积处理多个小批次
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_section: lecture-02.md / 第11节 Trading compute for memory / Gradient accumulation
content_scope: 课程基础补充；四个等大小小批次的平均损失与一次更新
created: 2026-10-30
updated: 2026-11-07
---
# 用梯度累积处理多个小批次

梯度累积是指：连续处理几个小批次，每批都计算梯度，但暂不更新参数；等这些梯度累加完成，再执行一次参数更新。

读完这一章，你应能排出四个小批次的前向、反向和一次更新，解释清梯度的位置与损失缩放为什么会影响结果。

前置知识：[清除旧梯度](08-zero-grad.md)、[训练循环与连续更新](09-training-loop.md)、[中间张量的存储与批次大小](14-activation-storage.md)。

## 四个小批次合成一次更新

假设每个小批次有 2 条文本，连续累积 4 批。这次更新共使用 $2 \times  4 = 8$ 条文本，执行 4 次 `backward()`，但只执行 1 次 `optimizer.step()`。

这里假设四批大小相同、每条文本的有效预测目标数量相同，且损失函数返回每批的平均损失。每批损失要先除以 4，使累积梯度对应这 8 条文本的整体平均，而不是四个平均损失之和。

下面的 `microbatches` 包含四批数据，每批提供输入 `micro_x` 和目标 `micro_y`。`model` 计算预测，`loss_fn` 计算该批的平均损失：

```python
optimizer.zero_grad()

for micro_x, micro_y in microbatches:
    loss = loss_fn(model(micro_x), micro_y) / 4
    loss.backward()

optimizer.step()
```

`optimizer.step()` 没有缩进，因此在四批数据处理完之后才执行。在这之前，各批前向计算使用同一组参数。

## 累加梯度，及时处理各批中间结果

`zero_grad()` 只在这一组小批次开始前调用。若每批都清除梯度，前面累积的结果就会丢失。

![箭头表示梯度贡献汇入同一存储；等大小、等权有效目标下，得到八条样本的平均贡献。](../assets/16-gradient-accumulation-1.svg)

箭头表示梯度贡献汇入同一存储；等大小、等权有效目标下，得到八条样本的平均贡献。

图中每批都及时执行反向传播，不必同时保留四批所需的中间结果。这样能减少同时保存的中间数据，但参数、梯度和优化器状态并不会因为分批而减少。总的前向与反向计算也没有因此消失。

上述整体平均梯度的说明，适用于损失可按样本独立相加、模型计算也满足相应条件的情况。跨样本运算或随机计算等因素可能使分批与一次处理大批次的数值结果不同。

## 清梯度的位置与平均梯度

如果将 `zero_grad()` 放到循环内每批开始处，第 2 批会清掉第 1 批的梯度，第 3 批会清掉第 2 批的梯度。循环外的 `step()` 只能使用最后一批留下的梯度。把清梯度放在整组小批次之前，才会保留各批累积的梯度。

损失乘一个常数，相应梯度也乘这个常数。因此，两批原始梯度为 2 和 6 时，每批损失除以 2，累积结果为 $2 / 2 + 6 / 2 = 4$，对应两批的平均梯度。这个等权平均依赖各批大小及有效预测目标数量相同的假设。

## 变长文本：按有效 token 数加权

当每批有效目标数不同时，批均值不能再简单除以批数。令第 $i$ 批有效 token 数为 $n_i$，批均值为 $L_i$，本次累积组实际有效总数为 $N=\sum_i n_i$。对可逐 token 相加、等权的损失：

$$L=\frac{\sum_i n_i L_i}{N},\qquad \nabla L=\sum_i\frac{n_i}{N}\nabla L_i.$$

计数固定且组内参数保持不变时，每批平均损失乘 $n_i/N$ 再反向传播即可。整组开始前清梯度，全部累积后才裁剪梯度和更新参数。

例如两批有 2 和 6 个有效 token，批均值分别是 1 和 3，则整体平均是 $(2\times1+6\times3)/8=2.5$，不是 $(1+3)/2=2$。若相应批均值对某个标量参数的导数为 2 和 6，累计导数是 $(2/8)\times2+(6/8)\times6=5$；等批平均错误地得到 4。损失的具体值不能决定其导数，这里导数是另给的假设条件。

组末尾不足额时，分母用本组**实际**有效 token 总数，不用计划容量或最大长度。若开始时不知道总数，可累计有效 token 损失之和的梯度，更新前统一除以实际 $N$；混合精度下还须正确处理 unscale，归一化后再裁剪。无需同时保留所有小批次计算图。

有效计数应排除 padding、`ignore_index` 和显式 mask 掉的目标，并与损失的 reduction 定义一致。单批有效计数为零时跳过该批，不对空集合求均值；整组 $N=0$ 时清除旧梯度、跳过 `optimizer.step()`，避免除零，也避免动量或 weight decay 仍改变参数。

这段公式针对无类别权重、逐 token 等权的目标。PyTorch 交叉熵使用类别权重时，其均值分母是有效目标权重之和；若目标是该带权均值，就应按相应权重和缩放。分布式训练还要结合跨 rank 的有效总数与 DDP 的梯度平均，本段不直接给出其缩放实现。跨样本耦合或随机前向也可能使分批与大批前向不完全相同。

参考：[PyTorch CrossEntropyLoss 的 reduction 与 ignore_index 定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[梯度累积与清零](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)。本节为代数推导和假设算例，未声称运行过 PyTorch 训练实验。

## DDP：rank均值不能直接当作全局token均值

若DDP在W个rank之间平均参数梯度，rank r在本次更新整组中有n_r个有效目标、这些目标的本地均值为L_r，令N=Σ_r n_r。我们要的是L=Σ_r(n_r/N)L_r；因此各rank应对(W n_r/N)L_r反向传播，抵消DDP额外的1/W。若逐microbatch使用sum loss S_{r,i}，则每批共同系数为W/N。

教学例子：W=2，两rank有效token为2、6，本地平均损失对同一标量参数的导数分别为2、6。不缩放时DDP给(2+6)/2=4；目标是(2×2+6×6)/8=5。对本地均值分别乘.5、1.5后，DDP平均得到(.5×2+1.5×6)/2=5。不能以本地n_r当全局分母，也不能在正确公式外再除W。

处理顺序为：在相同更新组上同步有效计数all_reduce(SUM)，确定共同N；整组开始前清梯度；按统一分母累积梯度；完成通信和AMP unscale后再裁剪、更新。使用no_sync时须按其要求将相应前向也纳入上下文，并保证最终反向执行同步；本节没有提供可直接用于任意通信配置的训练脚本。

### 两个microbatch怎样只同步一次

固定W=2，每个rank各处理两批。rank0两批各1个token，批均值梯度分别为2、4；rank1两批各3个token，批均值梯度分别为6、8。全局N=8，对每批sum loss统一乘W/N=1/4。rank0累积出(1×2+1×4)/4=1.5，rank1累积出(3×6+3×8)/4=10.5，最终DDP平均为6，等于八个token的整体平均(2+4+18+24)/8。

先在所有rank确定N并清梯度，第一批的前向和反向都放进no_sync上下文；第二批在上下文外做前向和反向，使这次反向同步先前累计的梯度。之后才unscale、裁剪和更新一次。每批已经按全组N归一化，不能再除microbatch数2；否则会错误地得到3。这里假定各rank按相同顺序完成两批、默认DDP平均，无join或自定义通信hook；批次数不齐需另行设计通信。

某rank没有有效token时，不能求空mean，也不能0×NaN。应从合法前向构造图连通零sum损失，仍参与其他rank需要的DDP通信，并核对unused-parameter与动态图设置。若全局N=0，所有rank一致跳过更新；单rank自行跳过backward可能导致collective不匹配。带类别权重时将token计数替换为目标定义的权重和；自定义通信hook若改变平均方式须重新推导。

零token反例：rank0整组无有效目标，rank1有6个且均值梯度为6。共同N=6，rank1的均值系数为2；rank0仍以图连通零sum完成所需通信，平均后(0+2×6)/2=6。若rank1直接backward未缩放均值，即便通信成功也只能得到3。

练习：两rank各有1、3个token，本地均值梯度分别为8、0，正确的全局梯度是多少？本地缩放系数是什么？答案：(1×8+3×0)/4=2；系数.5、1.5，通信平均后(.5×8+1.5×0)/2=2。

固定来源：[PyTorch v2.9.0 DDP文档源码](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/parallel/distributed.py)，说明跨节点梯度平均、sum loss的尺度区别及no_sync边界。本节由B起草公式和单批算例，A补写两批no_sync顺序、共同分母及零token反例；算例经本机算术核对，未运行分布式训练。其他原始章节仍按原复核状态管理。关联[完整技术问答](https://github.com/lw-liuwang/cs336-notes/discussions/11)。

## 小练习

每个小批次有 2 条文本，累积 3 批后再更新一次参数。共使用几条文本？反向传播和参数更新分别执行几次？

### 参考答案

共使用 `6` 条文本，执行 `3` 次 `backward()`、`1` 次 `optimizer.step()`。

每批都贡献梯度，但等三批梯度累积完成后才更新一次参数。若每批返回同权平均损失，本例相应将每批损失除以 3。

变长补充：两批有效 token 数为 3、1，批均值为 2、6，整体损失是多少？两次反向传播分别给批均值乘什么系数？

答案：$(3\times2+1\times6)/4=3$；分别乘 $3/4$、$1/4$。分母来自实际有效 token 总数 4。


## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
