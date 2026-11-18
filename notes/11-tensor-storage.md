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

## view和转置：逻辑大小不等于独占存储

构造连续float32张量a，形状[2,3]；b=a.view(3,2)，c=a.t()，d=a.clone()，四者同时存活。每个张量都有6个逻辑元素，即24字节，但view和transpose只改变元数据，不复制底层数据。

| 张量 | shape | stride | 逻辑字节数 | 底层数据storage |
| --- | --- | --- | --- | --- |
| a | [2,3] | (3,1) | 24 | A |
| b | [3,2] | (2,1) | 24 | A |
| c | [3,2] | (1,3) | 24 | A |
| d | [2,3] | (3,1) | 24 | D |

逻辑大小逐项相加为96字节；普通独立创建的A与D各24字节，去重storage数据小计只有48字节。c在这个非退化形状下不连续；e=c.contiguous()若继续存活，会另分配24字节，总计72字节。对已经连续的输入，contiguous可以直接返回原张量，并不保证分配。clone有独立storage，不能当view。

检查时可比较同设备张量的untyped_storage().data_ptr()和untyped_storage().nbytes()，并记录storage_offset()与stride()。张量自己的data_ptr()指向首个逻辑元素，切片offset非零时即使data_ptr不同也可能共享storage；空张量指针、外部buffer或复杂重叠区间又有边界，不能把指针相等测试当作任意分配器的完整别名算法。

### 首元素指针不同，仍然可能重叠

令s=a[:,1:]，r=a.view(-1)[2:3]，仍用上述连续float32张量。s形状[2,2]、stride=(3,1)、storage_offset=1，逻辑16字节；r形状[1]、stride=(1,)、storage_offset=2，逻辑4字节。若A的storage基址为P，a.data_ptr()=P、s.data_ptr()=P+4、r.data_ptr()=P+8，三个首元素指针不同，但三者的untyped_storage().data_ptr()都为P、nbytes()都为24。

更具体地，s[0,1]与r[0]都对应a[0,2]，所以这两片不仅共享storage，还共享一个元素。若只按各自data_ptr去重，会把同一块24字节重复记账。a/b/c/d都存活时再保留s和r，数据storage小计仍为48字节；若最终只保留r，逻辑只剩4字节，但题设整块24字节A仍被引用。这里讨论索引与存活关系，不建议对参与autograd的叶子view做原地写入验证。

例如只保留a的一元素切片，逻辑大小是4字节，但它仍可能使整个24字节storage存活。删除a这个名字也不意味着底层数据已释放：其他view仍引用它。更不能把上述48或72字节称为CUDA峰值显存；这里排除了分配器缓存/取整、元数据、上下文、梯度及workspace。

练习：在a/b/c/d均存活时删除b，storage小计改变吗？再删除d呢？答案：先仍为48字节，b只是A的一个别名；再删除d且没有其他引用时，题设数据storage小计降为24字节。这是可达数据存储的账本，不承诺allocator立刻把显存返还系统。

固定来源：[PyTorch v2.9.0 Tensor Views](https://github.com/pytorch/pytorch/blob/v2.9.0/docs/source/tensor_view.rst)的共享数据、transpose与contiguous说明。该文档已下载核对；本节24/48/72/96字节为结构推导，未运行显存测量。B起草view/transpose/clone存储表，A补写非零offset、首元素指针不同仍重叠的反例。关联[技术问答#13](https://github.com/lw-liuwang/cs336-notes/discussions/13)，全篇其他内容仍按原复核标记管理。

## 小练习

一个张量的形状是 `(2, 5, 8)`，使用 FP32 格式，每个元素占 4 字节。它的元素数据共占多少字节？

### 参考答案

共占 `320` 字节。$2 \times  5 \times  8 = 80$ 个元素，$80 \times  4 = 320$ 字节。

先计算元素数量，再乘每个元素的字节数。

## 来源与复核状态

本篇由个人 Coral 笔记导出，技术内容待协作复核。来源版本见 [导入清单](../provenance.json)。参考的[社区课程笔记](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)不是课程官方材料；本篇为基础知识补充解释。
