# CS336 学习笔记

从张量维度、损失和梯度出发，理解训练循环和显存占用。笔记以原理、图解、手算和带答案的小练习为主。

这是个人学习与协作仓库，不是 Stanford 官方材料。首批十八篇来自 lw-liuwang 的 Coral 基础笔记，已转换为 GitHub 可读的 SVG 和 Markdown 表格，技术内容待逐篇复核。后续修改以本仓库为主，不自动回写 Coral。

## 阅读顺序

- [张量的三个维度](notes/00-foundations.md)
- [Linear 的输出维度与权重数量](notes/01-linear-shapes.md)
- [在词表维度上做 softmax](notes/02-softmax-axis.md)
- [按目标编号取出实际答案的概率](notes/03-target-probability.md)
- [一个位置的交叉熵损失](notes/04-single-token-loss.md)
- [多个位置的平均损失](notes/05-mean-loss.md)
- [反向传播计算梯度](notes/06-backward-gradient.md)
- [用 SGD 执行一次参数更新](notes/07-optimizer-step.md)
- [清除旧梯度](notes/08-zero-grad.md)
- [训练循环与连续更新](notes/09-training-loop.md)
- [计算张量的元素数量](notes/10-tensor-numel.md)
- [计算张量的元素数据大小](notes/11-tensor-storage.md)
- [转换 dtype 后的存储变化](notes/12-dtype-storage.md)
- [参数与梯度的存储小计](notes/13-parameter-gradient-storage.md)
- [中间张量的存储与批次大小](notes/14-activation-storage.md)
- [优化器状态的存储](notes/15-optimizer-state-storage.md)
- [用梯度累积处理多个小批次](notes/16-gradient-accumulation.md)
- [用重算减少中间数据存储](notes/17-activation-checkpointing.md)
- [LayerNorm的归一化轴与方差分母](notes/18-layernorm-variance-axes.md)

- [从两条梯度路径理解词嵌入与输出权重共享](notes/19-embedding-weight-tying.md)

- [从逐行归一化理解因果注意力与移位目标](notes/20-causal-attention-mask.md)

- [用二维旋转推导RoPE的相对位置与缓存偏移](notes/21-rope-relative-position.md)

- [用参数与逐元素账本理解SwiGLU的宽度预算](notes/22-swiglu-width-budget.md)

- [按查询头与键值头分别核算GQA的缓存和输出](notes/23-gqa-kv-cache.md)

- [从列切分与行切分推导前馈层张量并行通信](notes/24-tensor-parallel-ffn.md)

## 如何协作

共享[协作规范与交接队列](https://github.com/lw-liuwang/codex-collaboration)。B 提交 CS336 改进 PR，A 核对并合并；实际共同完成的内容保留共同署名。需要回答的问题放在本仓库 Discussions 的 Q&A 类别，交接在共享仓库 Issues 记录。

[首批可改进主题](BACKLOG.md)提供知识问题和验收方向，不按排版碎片拆分任务。[导入清单](provenance.json)记录原笔记 URI、版本和导出哈希；对方无需 Coral 即可阅读本仓库全部内容。

## 资料

- [Stanford CS336](https://cs336.stanford.edu/)：课程入口，具体年度安排以官网为准。
- [社区课程笔记的参考版本](https://github.com/narain1/stanford-cs336-2026-notes/tree/98492d17c0feaaf1ffbb9ecdc42154cd2349ae5e)：原始 Coral 笔记参考过的社区材料，非课程官方文本。
- [PyTorch 文档](https://docs.pytorch.org/docs/stable/index.html)：涉及 API 行为时核对对应版本。

初版笔记与协作文档由 OpenAI Codex 辅助整理。没有执行的代码不标为已运行；导出完成不等于所有推导和图已完成复核。暂未指定开源许可证。
