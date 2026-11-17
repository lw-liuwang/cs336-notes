# 首批可改进主题

这些是初始笔记尚未展开的边界或连接点，不表示已完成验证。领取时在共享仓库建立交接，给出目标文档、实际改动和验收条件。已有相关 PR 先续做，内容可以合并成完整主题。

| 主题 | 对应笔记 | 改动与验收方向 |
| --- | --- | --- |
| 不等有效 token 的梯度累积 | [16](notes/16-gradient-accumulation.md) | 两批有效 token 为 2 和 6、平均损失为 1 和 3；推导整体目标及每批缩放，给出梯度算例，对照错误的等权平均 |
| softmax 的数值稳定性 | [02](notes/02-softmax-axis.md) | 推导减去最大值为什么不改变概率，比较大 logits 的直接求指数与稳定形式 |
| logits 到交叉熵的直接计算 | [04](notes/04-single-token-loss.md) | 连接目标 logit、logsumexp 和负对数概率，解释为什么训练 API 可以直接收 logits |
| padding 与 mean 的分母 | [05](notes/05-mean-loss.md) | 结合 ignore_index 和实际目标计数，区分按 token 和按文本平均；处理没有有效目标的情况 |
| 线性层偏置与参数共享 | [01](notes/01-linear-shapes.md) | 区分权重数量、偏置数量、批次/序列长度对参数数量的影响，手算一个带 bias 的例子 |
| grad 为 None 和为零 | [08](notes/08-zero-grad.md) | 对照 zero_grad 的 set_to_none 行为，以及优化器对缺失梯度和零梯度的不同处理，注明版本与来源 |
| 梯度来自同一参数点 | [09](notes/09-training-loop.md) | 跟踪两轮平方损失的参数、损失和梯度；区分累积多批后更新与每批立即更新 |
| 逻辑大小与底层存储 | [11](notes/11-tensor-storage.md) | view、转置和共享 storage 的具体例子，解释 numel×element_size 为什么不是独占分配量 |
| dtype、精度和单位 | [12](notes/12-dtype-storage.md) | 区分 FP16/BF16 的相同字节数与不同数值范围，以及十进制 GB 和二进制 GiB |
| Adam 的状态生命周期 | [15](notes/15-optimizer-state-storage.md) | 说明状态何时建立；区分 FP32 示例、实现依赖的混合精度与主权重副本，不把估算当峰值实测 |
| 激活检查点与随机状态 | [17](notes/17-activation-checkpointing.md) | 从带 dropout 的示例解释重算一致性，注明 PyTorch checkpoint 版本与参数语义 |
| 一个训练更新的内存账本 | [13](notes/13-parameter-gradient-storage.md)、[14](notes/14-activation-storage.md)、[15](notes/15-optimizer-state-storage.md) | 用同一假设下的参数、梯度、状态和激活数量建立账本，分开持久占用与临时峰值 |

问答围绕上述实际疑问展开，回答者给出推导和引用，提问者验证后采纳。技术复核不是简单将状态改为 reviewed；需在 PR 说明核对了什么。
