---
verification_status: pending-technical-review
title: CS336：按目标编号取出实际答案的概率
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_sections:
  - lecture-01.md / 第4节 Assignment 1 — Basics：cross-entropy loss
  - lecture-01.md / 第4节 Assignment 5 — Alignment：next-token supervision
content_scope: 课程基础补充；概率索引例子来自当前教学对话
created: 2026-10-30
updated: 2026-11-07
---
# 按目标编号取出实际答案的概率

模型为词表中的每个 token 分配概率后，训练需要取出实际答案对应的那一个概率。决定取哪一项的是目标 token 编号，不是模型认为哪一项最有可能。

这一章要学会沿一个目标编号找到对应概率，并分清“数据给出的正确答案”和“模型当前最想选的答案”。

前置知识：[沿词表维度计算 softmax](02-softmax-axis.md)。

## 编号与概率位置

假设词表中有 5 个 token，编号从 0 到 4。一个预测位置的概率按 token 编号顺序排列：

```python
probs = [0.1, 0.2, 0.4, 0.2, 0.1]
target_id = 1
```

`target_id` 保存原文中实际下一个 token 的编号。这里的编号是 1，所以要取概率列表中索引为 1 的元素：

```python
answer_probability = probs[target_id]
# 得到 0.2
```

Python 索引从 0 开始，索引 1 对应第二个元素。不要把“编号 1”误读成“取列表中的第一个值”。

这一步可以理解成沿编号查找：

![绿色格表示训练目标；最大概率 0.4 属于编号 2，不能替代目标概率。](../assets/03-target-probability-1.svg)

绿色格表示训练目标；最大概率 0.4 属于编号 2，不能替代目标概率。

图中的目标编号决定从列表中取哪一项。换一个目标编号，即使概率列表不变，取出的实际答案概率也可能不同。

## 实际答案不由最高概率决定

例子中，最高概率是 0.4，对应编号 2。它反映模型当前更偏向哪个候选，但这条训练样本的实际答案仍然是编号 1。因此，评价这次预测时要用 0.2，而不是 0.4。

模型给实际答案的概率越高，这个位置的交叉熵损失越小。按目标编号取概率，就是把模型输出与训练答案对应起来的关键一步。

## 小练习

模型为编号 0、1、2 分配的概率依次为 `[0.1, 0.6, 0.3]`，训练目标是编号 2。

实际答案对应哪个概率？应该直接选择最大的 0.6 吗？

### 参考答案

实际答案的概率是 `probs[2] = 0.3`，不应该选择 0.6。

目标编号 2 对应列表中的第三个元素。0.6 对应编号 1，只是模型当前给出的最高概率，并不是这个样本实际答案的概率。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
