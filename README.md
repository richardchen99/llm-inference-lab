# LLM Inference Lab

An interactive research lab by **Richard Chen · 中国人民大学 / Renmin University of China**.

[Live lab](https://richardchen99.github.io/llm-inference-lab/) · [Personal research](https://richardchen99.github.io)

把 **KV Cache** 与 **模型家族** 放在同一实验室：缓存能否复用，取决于模型允许哪些信息流动。

## Experiments

- **Cache workbench**：prefill 后逐 token、逐 layer 执行 Q/K/V 投影 → K/V 追加 → 历史读取 → 残差 / FFN；缓存格、注意力权重与最终隐藏状态按阶段同步出现。
- **Numerical witness**：两条独立路径分别做完整前缀重算与增量缓存，比较最后一个位置的真实数值误差。
- **Memory budget**：调节上下文长度、KV heads、batch size、dtype，观察缓存内存与 GQA / MQA 的影响。
- **Model family atlas**：切换 Encoder-only / Decoder-only / Encoder–Decoder，点击 Query token 查看 mask 和信息路径；第三类额外显示目标到源的 cross-attention。

提供句子 `The cat sat → on the mat .` 与代码 `def square(x): → return x * x` 两个固定续写示例。固定词序用于检查计算一致性，未调用模型采样生成。

## Try this

1. 单步执行 prefill，观察两层的前缀 K/V。
2. 新 token 首先进入 Layer 1；Append K/V 后，只新增该层的一格。
3. Read history 时该层缓存高亮，显示新 Query 对历史位置的权重。
4. Layer 1 完成后，Layer 2 使用它的输出继续计算。两层结束才显示最终隐藏状态及缓存 / 重算误差。
5. 改看 Encoder-only，注意新增输入可能改变历史隐藏状态；对比 Encoder–Decoder 的固定源端 KV 与不断增长的目标端 self-attention KV。

## Mathematical scope

$$
z_t=\operatorname{softmax}\left(
\frac{q_t[K_{<t};k_t]^\top}{\sqrt{d_k}}
\right)[V_{<t};v_t].
$$

固定前缀、参数、位置规则及确定性前向计算下，因果 mask 保证新增未来 token 不改变旧位置的表示；逐层递推即可复用每层 K/V。新 Query 仍需读取历史缓存，解码成本不是常数。

演示模型使用 **2 layers、4 dimensions、1 attention head**，含确定性 token / 位置表示、线性 Q/K/V 投影、因果 attention、残差和逐位置非线性变换。它是验证缓存机制的小网络，没有训练权重、LayerNorm 或 LM head，不等同于完整生产 Transformer。

内存面板独立采用 **32 layers、head dimension 128**：

$$
B_{\mathrm{KV}}=2LH_{\mathrm{KV}}d_hTbN.
$$

单位为 bytes，展示时换算 MiB / GiB。估算不含权重、临时激活、分页 / 分配器与量化元数据开销。INT8 选项表示理想张量字节数。

## Run locally

Use Node.js **24** (supported minimum: 22.12).

```bash
npm ci
npm run dev -- --host 127.0.0.1
npm test
npm run build
npm run preview -- --host 127.0.0.1
```

## Implementation

- `src/model.ts`：完整前向、增量前向、逐步轨迹、attention mask、内存计算。
- `src/App.tsx`：逐层 / 逐阶段状态机、cache lanes、mask atlas 与参数控件。
- `src/shared.tsx` / `src/style.css`：KaTeX、Framer Motion、浅色玻璃 UI、键盘操作与 reduced-motion 支持。
- `tests/model.test.mjs`：缓存与完整前向相等、因果历史不变及双向反例、内存单位、三类 attention mask。

所有计算在浏览器本地完成，无外部推理服务、密钥或 GPU 要求。`npm test` 编译计算模型并运行 Node test runner；`npm run build` 做类型检查与静态构建。

GitHub Actions 在 `main` 推送后以 Node 24 测试、构建和部署 GitHub Pages。首次部署需设置 Pages source 为 GitHub Actions。

## Research series

[Transformer Architecture Lab](https://richardchen99.github.io/transformer-architecture-lab/) ·
[Position Encoding Lab](https://richardchen99.github.io/position-encoding-lab/) ·
[Tokenizer Playground](https://richardchen99.github.io/tokenizer-playground/) ·
[LLM RL Lab](https://richardchen99.github.io/llm-rl-lab/)

## Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face: Cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [BERT](https://arxiv.org/abs/1810.04805)
- [T5](https://arxiv.org/abs/1910.10683)
