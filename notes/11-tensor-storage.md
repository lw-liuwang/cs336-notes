---
verification_status: pending-technical-review
title: CS336：计算张量的元素数据大小
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_sections:
  - lecture-02.md / 第2节 Tensor semantics：numel 与 element_size
  - lecture-02.md / 第3节 Precision：格式与每元素字节数
content_scope: 课程基础补充；普通稠密张量的微型存储计算例子
created: 2026-10-30
updated: 2026-11-07
---
# 计算张量的元素数据大小

知道张量有多少个元素后，还需要知道每个元素占多少字节，才能算出元素数据大小。元素数量与字节数量是两个不同的量。

这一章要学会用元素总数乘每元素字节数计算数据大小，避免把“有多少个数”直接当成“占多少字节”。

前置知识：[计算张量的元素数量](10-tensor-numel.md)。

## 元素数量乘以每元素字节数

`dtype` 表示张量中数值的存储格式。比如，`torch.float32` 使用 FP32 格式，每个元素占 4 字节。PyTorch 的 `element_size()` 可以返回每个元素所占的字节数。

下面创建一个形状为 `(3, 4, 5)`、使用 FP32 格式的张量：

```python
import torch

x = torch.zeros(3, 4, 5, dtype=torch.float32)

data_bytes = x.numel() * x.element_size()
print(data_bytes)
```

`x.numel()` 返回 `60`，`x.element_size()` 返回 `4`，所以元素数据大小为 $60 \times  4 = 240$ 字节。

代码输出：

```text
240
```

这里的 60 是元素数量，不能直接当作 60 字节。即使元素的值全部为零，本例仍按每个 FP32 元素 4 字节计算。

| 维度计数 | 每个元素 | 数据主体 |
| --- | --- | --- |
| 3 × 4 × 5 = 60 个元素 | FP32：4 字节 | 60 × 4 = 240 字节 |

先数元素，再乘每元素字节数；此处不计对象、分配器、对齐等额外开销。

图中，形状决定元素数量，存储格式决定每个元素的字节数，两者相乘得到结果。本例计算的是新建普通稠密张量的元素数据，不包含梯度、优化器状态或 Python 对象等额外开销。

## 小练习

一个张量的形状是 `(2, 5, 8)`，使用 FP32 格式，每个元素占 4 字节。它的元素数据共占多少字节？

### 参考答案

共占 `320` 字节。$2 \times  5 \times  8 = 80$ 个元素，$80 \times  4 = 320$ 字节。

先计算元素数量，再乘每个元素的字节数。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
