# 图片来源与实验状态

截图均采自实际运行的 [LLM Inference Lab](https://richardchen99.github.io/llm-inference-lab/)，使用公开的内置案例，仅包含应用内容，也用于[配套研究笔记](https://richardchen99.github.io/blog/llm-inference-lab-note/)。截图日期：**2026-09-18**。

| 文件 | 截图状态 |
| --- | --- |
| [`overview.jpg`](overview.jpg) | 句子案例；首个续写词元 `on`；两层计算完成；缓存与完整重算的误差显示为零 |
| [`memory.jpg`](memory.jpg) | 32 层、8 个 KV 头、每头维度 128、上下文 4,096、BF16、批大小 1；K/V 张量共占 512 MiB |
| [`families.jpg`](families.jpg) | 编码器—解码器架构；目标端 Query 2 通过交叉注意力读取源端 |

框架图采用统一布局，展示两层数值模型与独立等价性检查，并标明显存估算是采用另一组配置的配套实验。具体范围见 [README](../../README.md#数学机制与实现范围)。

- 中文版：[高清 PNG](architecture.png) · [可编辑 SVG](architecture.svg)
- 英文版：[高清 PNG](architecture.en.png) · [可编辑 SVG](architecture.en.svg)

PNG 宽度为 3,200 像素。框架图由矢量图形与文字绘制，截图来自真实界面，均未使用文生图模型。图片不包含编辑器窗口、浏览器边框、本机路径或私人账户信息；数值仅对应所述实验，不作为性能基准。
