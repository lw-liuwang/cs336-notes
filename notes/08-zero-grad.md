---
verification_status: pending-technical-review
title: CS336：清除旧梯度
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_section: lecture-02.md / 第10节：backward() 累积梯度与 optimizer.zero_grad()
content_scope: 课程基础补充；微型数值例子承接当前教学对话
created: 2026-10-30
updated: 2026-11-07
---
# 清除旧梯度

如果希望每次更新只使用本轮计算的梯度，需要先清除保存的旧梯度。PyTorch 用 `optimizer.zero_grad()` 完成这个操作。

这一章要弄清旧梯度为什么会混进新梯度，以及 zero_grad 应放在哪里，才能保留本轮需要的贡献。

前置知识：[用 SGD 执行一次参数更新](07-optimizer-step.md)。

## 清除的是梯度，不是参数

`loss.backward()` 会累积梯度，而不是直接覆盖已有梯度。`optimizer.step()` 更新参数后，也不会自动清除梯度。

例如，假设某参数保存的旧梯度是 `-2`，本轮新算出的梯度是 `3`。不清除旧梯度，保存的结果就是 $-2 + 3 = 1$；先清除，则本轮保存的是 `3`。这组数只用于演示累加规则。

沿用已经创建好的 `w` 和 `optimizer`，一次训练可以按以下顺序执行：

```python
optimizer.zero_grad()
loss = (w - 3) ** 2
loss.backward()
optimizer.step()
```

各步骤的关系是：

| 阶段 | grad 存储中的内容 | 参数 |
| --- | --- | --- |
| 上一轮结束后 | 仍可能留着旧梯度 | 上一轮新值 |
| 本轮清理 | 清空旧贡献 | 保持不变 |
| 本轮反向 | 写入本轮贡献 | 保持不变 |
| 本轮更新 | 供优化器读取 | 按梯度更新 |

按时间从上往下读；如果跳过清理，新反向通常会累加到旧贡献上。

清除后不会再把旧梯度混入本轮计算。`zero_grad()` 可以把参数的梯度设为 `None`，或按设置将已有梯度置为零；这里先理解它是在清除梯度。

## 小练习

参数刚更新为 `w = 2.5`，随后只清除保存的梯度。参数 `w` 会变成多少？

### 参考答案

仍然是 `2.5`。

`zero_grad()` 清除的是保存的梯度，不会清除或重置模型参数。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
