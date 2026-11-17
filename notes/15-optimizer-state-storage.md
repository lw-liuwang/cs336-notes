---
verification_status: pending-technical-review
title: CS336：优化器状态的存储
course: Stanford CS336 Spring 2026 — 入门基础
material_url: https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_revision: 98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e
material_section: lecture-02.md / 第9节 Training memory and compute inventory：Adam optimizer state
content_scope: 课程基础补充；假设参数、梯度及 Adam 两组状态均为 FP32
created: 2026-10-30
updated: 2026-11-07
---
# 优化器状态的存储

优化器更新参数时，除了读取当前梯度，还可以保存梯度历史的统计信息。这些额外保存的数据叫作优化器状态，也需要计入存储。

这一章要弄清 Adam 额外保存了哪两组数，并把参数、梯度、优化器状态四项存储分别算清。

前置知识：[参数与梯度的存储小计](13-parameter-gradient-storage.md)。

## Adam 保存的两组状态

Adam 是另一种常用的优化器。它用两组状态辅助更新参数：一组记录梯度历史的平滑统计，另一组记录梯度平方历史的平滑统计。可以先理解为，更新时同时参考当前梯度和过去的变化情况。

每组状态的形状都与对应参数张量相同。因此，80 个参数各对应 80 个第一组状态数值和 80 个第二组状态数值。这些状态是另外保存的数据。

## 四项数据分别计算

假设参数、梯度和两组状态都已经保存，全部使用 FP32，每个元素占 4 字节。对于 80 个参数：

| 数据 | 元素数量 | 数据大小 |
|---|---:|---:|
| 参数 | 80 | 320 字节 |
| 梯度 | 80 | 320 字节 |
| Adam 第一组状态 | 80 | 320 字节 |
| Adam 第二组状态 | 80 | 320 字节 |

两组优化器状态合计 `640` 字节，四项合计 $320 + 320 + 320 + 320 = 1280$ 字节。

![绿色两块是 Adam 的持久历史状态；例子不计激活、临时缓冲和混合精度主参数等。](../assets/15-optimizer-state-storage-1.svg)

绿色两块是 Adam 的持久历史状态；例子不计激活、临时缓冲和混合精度主参数等。

图中两组状态是独立于参数和梯度的额外数据。本例只计算这四项，不包括中间结果、额外参数副本、标量等，也不是训练峰值内存。

## 小练习

模型有 100 个参数，其他假设与例子相同。Adam 两组状态合计多少字节？参数、梯度及两组状态合计多少字节？

### 参考答案

两组状态合计 $100 \times  2 \times  4 = 800$ 字节。

四项数据合计 $100 \times  4 \times  4 = 1600$ 字节。

每组都有 100 个元素，两组优化器状态需要在参数与梯度之外另计空间。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
