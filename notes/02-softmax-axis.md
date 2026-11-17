---
verification_status: pending-technical-review
title: CS336：在词表维度上做 softmax
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_sections:
  - lecture-02.md / Tensor semantics
  - lecture-03.md / 7.1 Output z-loss：词表上的 softmax 公式
created: 2026-10-30
updated: 2026-11-07
---
# 在词表维度上做 softmax

语言模型的输出分数需要沿词表维度转换成概率。学会指定这个维度，就能保证每个输入位置各自得到一组候选 token 的概率。

这一章要弄清：每个位置的候选概率在哪里，以及 softmax 应沿哪一维计算，才能让各位置各自得到总和为 1 的概率。

前置知识：[线性层的输出维度与权重数量](01-linear-shapes.md)。

## 词表维度的位置

假设输出分数 `logits` 的形状是 `(2, 6, 10)`，表示两条文本、每条六个输入位置，每个位置有十个候选 token 的分数。

PyTorch 的维度索引从 0 开始。这三个维度分别是：

| 维度编号 | 长度 | 含义 |
|---|---:|---|
| `0` | 2 | 批次中的文本条数 |
| `1` | 6 | 每条文本的输入位置数量 |
| `2` | 10 | 词表中的候选 token 数量 |

`-1` 表示最后一个维度。因此，对这个三维张量来说，`dim=-1` 与 `dim=2` 指定的是同一个维度。使用 `-1` 可以直接表达“最后一个维度”，无需数前面有多少个维度。

## 每个位置分别转换

softmax 将同一组分数转换成总和为 1 的概率。这里，一条文本的一个位置对应十个分数，这十个分数构成一组。

| 归一化分组 | 分数所在的一行 | softmax 后 |
| --- | --- | --- |
| 文本 1 · 位置 1 | 10 个候选分数 | 10 个概率，合计 1 |
| 文本 1 · 位置 2 | 另一行 10 个分数 | 另一组概率，合计 1 |
| …… | 各位置分开处理 | 不跨位置混在一起求和 |

每一行对应一个预测位置；softmax 沿词表维度处理，不改变张量形状。

所有输入位置都会分别进行图中的处理。形状 `(2, 6, 10)` 中共有 $2 \times  6 = 12$ 组，每组都包含十个候选。softmax 改变这些数的值，但不改变张量形状。

## PyTorch 写法

下面用全为零的分数观察转换结果。`torch.zeros(2, 6, 10)` 创建指定形状的张量，并将每个数设为零。

```python
import torch

logits = torch.zeros(2, 6, 10)
probs = torch.softmax(logits, dim=-1)

print(probs.shape)
```

输出形状是：

```text
torch.Size([2, 6, 10])
```

`dim=-1` 指定沿最后一个维度进行 softmax。因为每个位置的十个候选分数完全相同，所以该位置的十个概率都为 `0.1`，合计为 1。其他位置也分别做相同的转换。

最后一维的 10 表示候选 token 数量。做完 softmax 后，模型仍然为每个位置保留十个候选概率，尚未选出要生成的 token。

## 小练习

一个输出分数张量的形状是 `(3, 4, 5)`，分别表示文本、位置与候选 token。沿候选维度做 softmax 后，输出形状是什么？哪些数构成一组总和为 1 的概率？

### 参考答案

输出形状仍然是 `(3, 4, 5)`。对每条文本的每个位置，最后一维的五个候选概率构成一组，总和为 1。共有 $3 \times  4 = 12$ 组，各组分别进行 softmax。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
