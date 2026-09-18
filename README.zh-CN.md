# LLM Inference Lab

**理解语言模型能复用什么，以及为什么。**

逐层观察 KV Cache，将增量解码与完整前缀重算逐项对照，再用 attention mask 解释不同模型家族的差异。一个足够小的确定性网络，让“缓存为什么成立”成为可以检查的数值问题。

[**进入实验室 ↗**](https://richardchen99.github.io/llm-inference-lab/) · [配套研究笔记](https://richardchen99.github.io/blog/llm-inference-lab-note/) · [English README](README.md) · [本地运行](#本地运行)

作者：**Richard Chen · 中国人民大学 / Renmin University of China** · [个人主页](https://richardchen99.github.io)

[![逐层 KV 缓存、执行阶段与数值等价检查](docs/assets/overview.jpg)](https://richardchen99.github.io/llm-inference-lab/)

*真实运行截图。首个续写 token 完成两层计算后，缓存路径与完整重算在显示精度下一致。*

## 四个相互连接的实验

| 模块 | 操作 | 证据 |
| --- | --- | --- |
| **Cache execution** | 逐步执行 Q/K/V、追加、历史读取、残差/FFN | 每一层只有在输入就绪后才更新 |
| **Numerical equivalence** | 对比独立实现的缓存与完整重算路径 | 检查最终隐藏状态的误差 |
| **Memory budget** | 改变上下文长度、KV heads、dtype 与 batch | 查看缓存张量内存如何变化 |
| **Model family atlas** | 切换三种模型家族 | 检查信息流与历史状态能否保持不变 |

句子与代码案例采用固定续写：`The cat sat → on the mat .` 和 `def square(x): → return x * x`，为数值检查提供可重复的输入。界面采用英文控件与中文解释。

## 从信息流到计算复用

![因果 attention、逐层 KV 复用、完整重算验证与内存估算的关系](docs/assets/architecture.png)

*原创框架图：因果结构解释复用，独立前向路径提供数值检查。[可编辑 SVG](docs/assets/architecture.svg) · [图片来源与状态](docs/assets/README.md)。*

## 跟随一个新 token

1. 选择句子案例，逐步执行 **prefill**，观察两层前缀 K/V。
2. 继续到 `on`。Layer 1 的 **Append K/V** 只向这一层追加当前 token。
3. 到达 **Read history**，查看新 Query 对历史 Key 与当前 Key 的权重。
4. 完成 Layer 1，再完成 Layer 2；两层结束后才显示最终隐藏状态及缓存/重算误差。
5. 查看模型家族：双向 encoder 追加输入后可能改变历史状态；因果 decoder 保留历史状态。Encoder–decoder 的固定源端 cross-attention K/V，与持续增长的目标端 self-attention K/V 含义不同。

模型测试要求两条计算路径的输出误差小于 **1e-12**。界面显示零表示当前显示精度下的一致，不代表所有生产实现都具有相同误差。

<details>
<summary><strong>展开内存预算与模型家族截图</strong></summary>

![32 层、8 个 KV heads、head dimension 128、上下文 4096、BF16、batch 1 的内存面板](docs/assets/memory.jpg)

*这组配置的 K/V 张量占用 512 MiB。*

![目标端 Query 通过 cross-attention 读取固定源端的 encoder-decoder 信息流](docs/assets/families.jpg)

*模型家族视图区分固定源端 cross-attention 与目标端因果 self-attention。*

</details>

## 数学机制与实现范围

在新的因果位置：

$$
z_t=\mathrm{softmax}\left(
\frac{q_t[K_{<t};k_t]^\top}{\sqrt{d_k}}
\right)[V_{<t};v_t].
$$

固定前缀、参数、位置规则与确定性前向计算时，因果 mask 保证新增未来 token 不改变历史位置的表示，因此可以逐层复用 K/V。**新 Query 仍需读取不断增长的历史**；缓存不会使 attention 成本成为常数。

演示网络为 **2 layers、4 dimensions、1 attention head**，包含固定 token/位置表示、Q/K/V 投影、因果 attention、残差与逐位置非线性 FFN。没有训练权重、LayerNorm 或 LM head；续写顺序预设，不从语言模型采样。

独立的内存面板采用 32 层与 head dimension 128：

$$
B_{\mathrm{KV}}=2LH_{\mathrm{KV}}d_hTsN.
$$

其中 $L$ 为层数， $H_{\mathrm{KV}}$ 为 KV head 数， $d_h$ 为 head dimension， $T$ 为缓存长度， $s$ 为每元素字节数， $N$ 为 batch size；系数二对应 Key 与 Value。

当 $L=32$、 $H_{\mathrm{KV}}=8$、 $d_h=128$、 $T=4096$、 $s=2$、 $N=1$ 时，结果为 **536,870,912 bytes = 512 MiB**。估算不包含模型权重、临时激活、分页/分配器开销及量化元数据；INT8 代表理想张量字节数。

## 本地运行

推荐 **Node.js 24**，最低支持 22.12。

```bash
git clone https://github.com/richardchen99/llm-inference-lab.git
cd llm-inference-lab
npm ci
npm run dev -- --host 127.0.0.1
```

```bash
npm test
npm run build
npm run preview -- --host 127.0.0.1
```

实验计算在浏览器内完成，无需推理服务、API 密钥或 GPU。技术栈为 React 19、TypeScript、Vite、Framer Motion 与 KaTeX。

## 实现与验证

| 入口 | 重点 |
| --- | --- |
| [`src/model.ts`](src/model.ts) | 完整/增量前向、执行轨迹、mask 与内存计算 |
| [`src/App.tsx`](src/App.tsx) | 逐层/阶段状态机、cache lanes、模型家族与参数控件 |
| [`src/shared.tsx`](src/shared.tsx) · [`src/style.css`](src/style.css) | 公式、动画、玻璃面板与 reduced-motion 支持 |
| [`tests/model.test.mjs`](tests/model.test.mjs) | 路径等价、因果历史不变、双向反例、内存单位与 mask |

`npm test` 编译模型后运行 Node test runner。[Pages 工作流](.github/workflows/deploy.yml) 使用 Node 24 测试、类型检查、构建并部署 `main`。Fork 后，将 Pages source 设为 **GitHub Actions** 即可部署。

## 阅读与引用

- Vaswani 等：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017，attention 与 encoder–decoder 架构。
- Hugging Face：[Cache strategies](https://huggingface.co/docs/transformers/kv_cache)，生产缓存实现与权衡。
- Ainslie 等：[GQA](https://arxiv.org/abs/2305.13245)，2023，grouped-query attention。
- [BERT](https://arxiv.org/abs/1810.04805) 与 [T5](https://arxiv.org/abs/1910.10683)，模型家族对照阅读。
- [配套研究笔记](https://richardchen99.github.io/blog/llm-inference-lab-note/)，中文实验导读。

用于课程或文章时，可链接本仓库并记录所用 commit。[CITATION.cff](CITATION.cff) 提供机器可读的软件署名信息。

## 系列实验室

| 项目 | 核心问题 |
| --- | --- |
| [Tokenizer Playground](https://github.com/richardchen99/tokenizer-playground) | 语料怎样变成可复用词表？ |
| [Transformer Architecture Lab](https://github.com/richardchen99/transformer-architecture-lab) | Attention 怎样把 token 表示转为上下文？ |
| [Position Encoding Lab](https://github.com/richardchen99/position-encoding-lab) | 位置怎样改变注意力几何？ |
| **LLM Inference Lab** | 什么条件下可以复用历史计算？ |
| [LLM RL Lab](https://github.com/richardchen99/llm-rl-lab) | 奖励怎样改变回答分布？ |

如果它对你的学习或教学有帮助，欢迎点亮 Star。也欢迎提交可复现的缓存实验与数值检查改进。
