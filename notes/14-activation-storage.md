---
verification_status: pending-technical-review
title: CS336：中间张量的存储与批次大小
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_section: lecture-02.md / 第9节 Training memory and compute
  inventory：activations 与 batch size
content_scope: 课程基础补充；仅计算一个 FP32 线性层输出张量的元素数据
created: 2026-10-30
updated: 2026-11-07
---
# 中间张量的存储与批次大小

前向计算产生的中间表示常称为激活张量。批次中的文本变多时，中间张量通常也会变大，但模型参数数量不因此改变。

读完这一章，你应能从输出形状算出一个激活张量的大小，并解释为什么增加批次条数会增加激活，却不增加权重数量。

前置知识：[参数与梯度的存储小计](13-parameter-gradient-storage.md)。

## 计算一个中间张量

下面的线性层将每个八维向量变成十维向量。在通常的默认 FP32 设置下：

```python
import torch

x = torch.zeros(2, 6, 8, dtype=torch.float32)
linear = torch.nn.Linear(8, 10, bias=False)
h = linear(x)
```

`h` 的形状为 `(2, 6, 10)`，包含 $2 \times  6 \times  10 = 120$ 个 FP32 元素，因此它的数据大小为 $120 \times  4 = 480$ 字节。

线性层仍然只有 $8 \times  10 = 80$ 个权重，各个文本和位置共享它们。

![两条激活容量采用相同尺度；单个 h 的数据量不能代表整次训练的全部显存。](../assets/14-activation-storage-1.svg)

两条激活容量采用相同尺度；单个 h 的数据量不能代表整次训练的全部显存。

若把批次条数 `B` 从 2 增加到 4，`h` 的形状变为 `(4, 6, 10)`，数据大小增加到 960 字节，权重数量仍为 80。

## 按批次条数换算

480 字节已经包含 2 条文本，不能直接把它当成每条文本的大小。每条文本的输出数据是 $480 \div  2 = 240$ 字节，因此 3 条文本为 $240 \times  3 = 720$ 字节，5 条为 $240 \times  5 = 1200$ 字节。`480 × 3` 对应原来两条文本的三倍，也就是 6 条文本。

## 单个张量与训练存储

为了反向传播，模型通常还会保存一些中间信息；具体保存哪些取决于计算操作，不是每个输出都必须保留。

这里算的只是 `h` 自身的元素数据大小，不是全部激活数据或训练过程的内存峰值。

## 小练习

其他设置不变，把批次条数改为 `B = 3`。`h` 的元素数据有多少字节？线性层的权重数量会改变吗？

### 参考答案

`h` 的数据大小是 $3 \times  6 \times  10 \times  4 = 720$ 字节。

权重数量不变，仍为 80。增加批次条数会增加本批中间数据，不会增加该线性层的参数。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
