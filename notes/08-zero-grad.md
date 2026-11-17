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

## 没有梯度与零梯度，更新语义不同

以下固定核对 **PyTorch v2.9.0** 的 `SGD` 与 `Optimizer.zero_grad` 源码，只作源码阅读和手算，没有运行框架实验。`grad=None` 表示目前没有保存的梯度张量；`grad` 是零张量则表示有梯度张量、数值为零。两者都没有新增的非零损失梯度，却不一定触发相同更新。

SGD 的 `_init_group` 只将 `p.grad is not None` 的参数加入本次更新列表。因此该参数的梯度为 None 时，跳过的是该参数的更新，包括其权重衰减与动量；同一优化器的其他有梯度参数仍可更新。已有动量状态不因这次跳过而清空或自动衰减。

对比两份相互独立、初始状态相同的标量参数：$w=2$，已有动量 $b=3$，学习率 $\eta=0.1$、动量系数 $\mu=0.9$、权重衰减 $\lambda=0.1$，`dampening=0`、`nesterov=False`、`maximize=False`。用普通非稀疏、`foreach=False`、`fused=False` 的路径逐步看：

| 初始梯度 | 权重衰减后的更新梯度 | 更新后动量 | 更新后参数 |
| --- | --- | --- | --- |
| None | 该参数未进入更新列表 | $b'=3$ | $w'=2$ |
| 零张量 | $g'=0+\lambda w=0.2$ | $b'=\mu b+g'=0.9\times3+0.2=2.9$ | $w'=w-\eta b'=2-0.1\times2.9=1.71$ |

这里是 SGD 把权重衰减加入梯度的顺序，不是 AdamW 的解耦衰减。零梯度不取消历史动量；例如把本例 $\lambda$ 改为零，仍有 $b'=2.7$、$w'=1.73$。题设假定 buffer 已存在；若首次建立 buffer，SGD 的初始化路径不同，不能直接代入旧 $b=3$。

## 首次建立动量，与后续递推不是同一分支

PyTorch v2.9.0 的 `_single_tensor_sgd` 在 buffer 尚不存在时，直接把权重衰减后的梯度复制为初始 buffer；已有 buffer 时，才执行 $b'=\mu b+(1-\mathrm{dampening})g'$。所以首次初始化不能总是当成“旧 $b=0$ 后套递推式”，尤其是在 dampening 非零时。

仍取 $w=2$、损失梯度为零、$\lambda=0.1$、$\eta=0.1$、$\mu=0.9$，因此先得到 $g'=0.2$，关闭 Nesterov 和 maximize。下面每行是独立的初始状态，不是连续更新：

| 初始 buffer | dampening | 本次实际采用的动量路径 | 更新后参数 |
| --- | --- | --- | --- |
| 不存在 | 0 | 首次复制：$b'=g'=0.2$ | $2-0.1\times0.2=1.98$ |
| 不存在 | 0.5 | 仍首次复制：$b'=g'=0.2$ | $1.98$ |
| 已有 $b=3$ | 0 | $b'=0.9\times3+0.2=2.9$ | $1.71$ |
| 已有 $b=3$ | 0.5 | $b'=0.9\times3+0.5\times0.2=2.8$ | $1.72$ |

若在第二行错误地先设 $b=0$ 再套后续递推，会算成 $b'=0.1,w'=1.99$，与所核对实现不同。清除梯度不会删除旧 buffer，因此“第一次 backward”“第一次对这个参数执行有效 step”和“新建优化器后的状态”也要分别确认。这组补充仍是固定版本源码与手算，不是框架运行结果。

## set_to_none=False 不会补建缺失梯度

`zero_grad(set_to_none=False)` 先检查 `p.grad is not None`，只把已有梯度张量清零；原本为 None 的参数直接跳过，不分配一个新零张量。`set_to_none=True` 则把已有梯度设为 None。两者都不重置参数或优化器动量，后续 backward 是否为某参数创建/累计梯度取决于本次计算图。

因此，不应通过“清梯度”推断某个参数必然参加后续 step。条件分支或冻结方式可能让某些参数未收到梯度；排查时要同时看 `.grad` 是否存在、数值、优化器类型及已有状态。混合精度溢出跳步、closure、稀疏梯度及其他实现路径不在本标量算例内；实际浮点结果还可能舍入。

补充练习：沿用上述 $w=2,b=3$，把 weight_decay 设为零，零梯度能否保证参数不动？如果梯度原本为 None，再调用 `zero_grad(set_to_none=False)`，会变成哪个分支？

答案：不能保证，动量仍给出 $b'=2.7,w'=1.73$；缺失梯度仍是 None，因此若后续没有新的 backward 写入梯度，该参数继续跳过 step。

固定依据：[`torch/optim/sgd.py` v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/sgd.py) 的 `_init_group`、`_single_tensor_sgd`；[`torch/optim/optimizer.py` v2.9.0](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/optimizer.py) 的 `zero_grad` 及其 None/零梯度说明。数值为假设情境的独立手算，不是训练测量；保留全篇待技术复核状态。

## 小练习

参数刚更新为 `w = 2.5`，随后只清除保存的梯度。参数 `w` 会变成多少？

### 参考答案

仍然是 `2.5`。

`zero_grad()` 清除的是保存的梯度，不会清除或重置模型参数。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
