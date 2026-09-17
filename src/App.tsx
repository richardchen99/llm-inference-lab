import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Shell,
  Panel,
  Formula,
  Tabs,
  Range,
  Stat,
  Bars,
  Tokens,
  Note,
  Matrix,
  Transport,
  usePlayback,
} from "./shared";
import { cacheBytes, trace, mask } from "./model";
const examples = {
  short: { prompt: ["The", "cat", "sat"], tail: ["on", "the", "mat", "."] },
  code: {
    prompt: ["def", "square", "(", "x", ")", ":"],
    tail: ["return", "x", "*", "x"],
  },
};
const families = [
  {
    id: "encoder-only",
    label: "Encoder-only",
    name: "BERT · read in both directions",
    tokens: ["The", "[MASK]", "sat", "on", "the", "mat"],
    copy: "适合分类、检索、掩码填空。每个位置可以读取整段输入。追加新词可能改变历史隐藏状态，因此不能把多层双向 encoder 的旧 KV 当作自回归 cache 原样复用。",
  },
  {
    id: "decoder-only",
    label: "Decoder-only",
    name: "GPT · continue a prefix",
    tokens: ["Transformer", "is", "powerful", "because"],
    copy: "每个位置只能读取自己及之前的 token。前缀固定、模型参数不变时，旧位置不受新增未来 token 影响，因此可以缓存每层历史 Key / Value。",
  },
  {
    id: "encoder-decoder",
    label: "Encoder–Decoder",
    name: "T5 · condition on a source",
    tokens: ["<BOS>", "猫", "坐", "在"],
    copy: "Encoder 双向编码源句；Decoder 对目标前缀使用 causal self-attention，同时用 cross-attention 读取源句。源端 K/V 可预计算，目标端 self-attention KV 逐步追加。",
  },
];
const phases = [
  "Project Q/K/V",
  "Append K/V",
  "Read history",
  "Residual + FFN",
];
export default function App() {
  const [example, setExample] = useState("short"),
    [layer, setLayer] = useState(0),
    [family, setFamily] = useState("decoder-only"),
    [query, setQuery] = useState(2),
    [heads, setHeads] = useState(8),
    [context, setContext] = useState(4096),
    [batch, setBatch] = useState(1),
    [bytes, setBytes] = useState(2);
  const ex = examples[example as keyof typeof examples],
    data = useMemo(() => trace(ex.prompt, ex.tail), [ex]);
  const play = usePlayback(1 + ex.tail.length * 8, example, 1100);
  const iteration =
      play.step < 2
        ? 0
        : Math.min(ex.tail.length, Math.floor((play.step - 2) / 8) + 1),
    phase = play.step < 2 ? -1 : (play.step - 2) % 4,
    activeLayer = play.step < 2 ? 0 : Math.floor(((play.step - 2) % 8) / 4);
  useEffect(() => {
    setLayer(activeLayer);
  }, [activeLayer, example]);
  const result = data[iteration];
  const layerSnap = (l: number) =>
    data[
      iteration === 0
        ? 0
        : activeLayer > l || (activeLayer === l && phase >= 1)
          ? iteration
          : iteration - 1
    ];
  const projected =
    iteration === 0 ? 0 : activeLayer >= layer ? iteration : iteration - 1;
  const cachedRows = ex.prompt.length + projected,
    fullRows =
      ex.prompt.length +
      Array.from(
        { length: projected },
        (_, i) => ex.prompt.length + i + 1,
      ).reduce((a, b) => a + b, 0);
  const weightsReady =
    play.step === 1 ||
    (iteration > 0 &&
      (activeLayer > layer || (activeLayer === layer && phase >= 2)));
  const outputReady =
    play.step === 1 || (iteration > 0 && activeLayer === 1 && phase === 3);
  const spec = families.find((f) => f.id === family)!,
    q = Math.min(query, spec.tokens.length - 1),
    source = ["The", "cat", "sat", "on", "the", "mat"];
  const selfMask = mask(
    family === "encoder-only" ? "encoder-only" : "decoder-only",
    spec.tokens.length,
  );
  const megabytes =
    cacheBytes(32, heads, 128, context, bytes, batch) / 1024 ** 2;
  return (
    <Shell
      slug="llm-inference-lab"
      title="LLM Inference Lab"
      subtitle="每生成一步，为什么不必把过去全部再算一遍？追踪两层因果网络的真实小矩阵，让缓存复用与三类模型的信息流一目了然。"
      sources={[
        ["Attention Is All You Need", "https://arxiv.org/abs/1706.03762"],
        [
          "Hugging Face · Cache strategies",
          "https://huggingface.co/docs/transformers/kv_cache",
        ],
        ["GQA · Grouped-Query Attention", "https://arxiv.org/abs/2305.13245"],
        ["BERT", "https://arxiv.org/abs/1810.04805"],
        ["T5", "https://arxiv.org/abs/1910.10683"],
      ]}
    >
      <div className="grid">
        <Panel
          title="Prefill once. Decode incrementally."
          eyebrow="01 / CACHE WORKBENCH"
        >
          <div className="row">
            <Tabs
              options={[
                { id: "short", label: "A sentence" },
                { id: "code", label: "A code prefix" },
              ]}
              value={example}
              onChange={setExample}
              label="Cache example"
            />
            <span className="badge">
              {play.step === 0
                ? "READY"
                : play.step === 1
                  ? "PREFILL"
                  : `DECODE ${iteration}`}
            </span>
          </div>
          <Tokens
            tokens={[...ex.prompt, ...ex.tail.slice(0, iteration)]}
            active={iteration > 0 ? ex.prompt.length + iteration - 1 : -1}
          />
          <div className="stageRail">
            {phases.map((p, i) => (
              <div key={p} className={i === phase ? "active" : ""}>
                <small>0{i + 1}</small>
                {p}
              </div>
            ))}
          </div>
          <div className="cacheLanes">
            {[0, 1].map((l) => {
              const snap = layerSnap(l);
              return (
                <div key={l}>
                  <button
                    className={layer === l ? "selected" : ""}
                    onClick={() => setLayer(l)}
                    aria-pressed={layer === l}
                  >
                    Layer {l + 1}
                  </button>
                  <div className="cacheSlots">
                    {snap.tokens.map((t, i) => (
                      <motion.div
                        layout="position"
                        initial={{ opacity: 0, y: 8 }}
                        animate={{
                          opacity: play.step ? 1 : 0.45,
                          y: 0,
                          backgroundColor:
                            iteration > 0 &&
                            i === ex.prompt.length + iteration - 1
                              ? "#f4e8cc"
                              : "#e5eff5",
                          boxShadow:
                            activeLayer === l && phase === 2
                              ? "0 0 0 1px #6fa8dc66"
                              : "0 0 0 1px #6fa8dc00",
                        }}
                        key={`${example}-${l}-${i}`}
                      >
                        <small>
                          {i} · {t}
                        </small>
                        <span>
                          K{" "}
                          {play.step
                            ? snap.cache[l].keys[i][0].toFixed(2)
                            : "—"}
                        </span>
                        <span>
                          V{" "}
                          {play.step
                            ? snap.cache[l].values[i][0].toFixed(2)
                            : "—"}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="legend">
            <span>
              <i />
              Reused prefix
            </span>
            <span>
              <i />
              New cache entries
            </span>
          </div>
          <Transport
            {...play}
            labels={Array.from({ length: play.total + 1 }, (_, s) =>
              s === 0
                ? "Input prefix"
                : s === 1
                  ? "Prefill · all prefix keys & values"
                  : `Token ${Math.floor((s - 2) / 8) + 1} · Layer ${Math.floor(((s - 2) % 8) / 4) + 1} · ${phases[(s - 2) % 4]}`,
            )}
          />
          <Note>
            固定续写用来演示增量计算，未调用语言模型生成。每个 token 先完成
            Layer 1，再进入 Layer 2；各层依次执行投影、追加、读取和残差 /
            FFN。每格显示 K/V 第一维，上一轮新增项在下一轮成为历史。
          </Note>
        </Panel>
        <Panel
          title="Reuse has a numerical witness"
          eyebrow="02 / SAME OUTPUT, LESS RECOMPUTATION"
        >
          <Note>
            {play.step === 0
              ? "推进一步后，prefill 会建立历史状态。"
              : phase === 0
                ? "只投影新增位置的 Q/K/V，历史 K/V 不动。"
                : phase === 1
                  ? "新 Key / Value 追加到当前层的缓存中。"
                  : phase === 2
                    ? "新 Query 与该层全部历史 Key 匹配，再按权重读取 Value。"
                    : phase === 3
                      ? activeLayer === 0
                        ? "Layer 1 完成残差与非线性变换，将新位置的表示送入 Layer 2。"
                        : "两层计算已完成，核对完整重算的最后一个位置。"
                      : "Prefill 并行处理前缀，保存每层的 Key / Value。"}
          </Note>
          <Formula
            tex={String.raw`z_t=\operatorname{softmax}\!\left(\frac{q_t[K_{<t};k_t]^\top}{\sqrt{d_k}}\right)[V_{<t};v_t]`}
          />
          <p className="miniTitle">
            Layer {layer + 1} ·{" "}
            {weightsReady ? "Attention weights" : "Awaiting history read"}
          </p>
          <Bars
            items={result.weights[layer].map((v, i) => ({
              label: result.tokens[i],
              value: weightsReady ? v : 0,
              display: weightsReady ? `${(v * 100).toFixed(1)}%` : "—",
              active: i === result.tokens.length - 1,
            }))}
          />
          <div className="stats">
            <Stat
              label="MAX OUTPUT ERROR"
              value={outputReady ? result.error.toExponential(1) : "—"}
              detail="两层完成后：缓存 vs 完整重算"
            />
            <Stat
              label="K/V PROJECTION ROWS"
              value={play.step >= 1 ? `${cachedRows} / ${fullRows}` : "—"}
              detail={`累计：缓存 / 重算，Layer ${layer + 1}`}
            />
          </div>
          <div className="callout">
            <h3>Final hidden state</h3>
            <Bars
              signed
              items={result.hidden.map((v, i) => ({
                label: `H [${i}]`,
                value: outputReady ? v : 0,
                display: outputReady ? v.toFixed(4) : "—",
              }))}
            />
          </div>
          <Note>
            省掉的是历史投影与旧位置的前向计算，不是所有 attention 工作。每个新
            Query 仍需要读取随上下文增长的 K/V。
          </Note>
        </Panel>
      </div>
      <Panel title="The cost of remembering" eyebrow="03 / MEMORY BUDGET">
        <div className="grid equal">
          <div>
            <Formula
              tex={String.raw`B_{\mathrm{KV}}=2\,L\,H_{\mathrm{KV}}\,d_h\,T\,b\,N`}
            />
            <Note>
              2 对应 K 与 V；32 层、head dimension 128 固定。这里仅计算理想 KV
              张量内存，不包括权重、临时激活、分配器和分页开销。GQA / MQA
              通过减少 KV heads 降低缓存占用。
            </Note>
          </div>
          <div className="controls">
            <Range
              label="KV heads"
              min={1}
              max={32}
              value={heads}
              onChange={setHeads}
            />
            <Range
              label="Context tokens"
              min={512}
              max={32768}
              step={512}
              value={context}
              onChange={setContext}
            />
            <Range
              label="Batch size"
              min={1}
              max={8}
              value={batch}
              onChange={setBatch}
            />
            <label className="field">
              Cache dtype
              <select value={bytes} onChange={(e) => setBytes(+e.target.value)}>
                <option value={2}>BF16 / FP16 · 2 bytes</option>
                <option value={4}>FP32 · 4 bytes</option>
                <option value={1}>INT8 · 1 byte (ideal)</option>
              </select>
            </label>
          </div>
        </div>
        <div className="stats">
          <Stat
            label="KV CACHE ONLY"
            value={`${(megabytes / 1024).toFixed(2)} GiB`}
          />
          <Stat
            label="PER SEQUENCE"
            value={`${(megabytes / batch).toFixed(0)} MiB`}
          />
          <Stat
            label="RELATIVE TO 32 KV HEADS"
            value={`${((heads / 32) * 100).toFixed(1)}%`}
          />
        </div>
      </Panel>
      <Panel
        title="Three families. Three information paths."
        eyebrow="04 / MODEL FAMILY ATLAS"
      >
        <Tabs
          options={families}
          value={family}
          onChange={(v) => {
            setFamily(v);
            setQuery(2);
          }}
          label="Model family"
        />
        <div className="grid equal">
          <div className="stack">
            <h3>{spec.name}</h3>
            <Note>{spec.copy}</Note>
            <Tokens
              tokens={spec.tokens}
              active={q}
              onSelect={setQuery}
              label="Query token"
            />
            <svg
              viewBox="0 0 520 140"
              className="diagram"
              role="img"
              aria-label="Allowed attention paths"
            >
              {spec.tokens.map((t, i) => {
                const x = 42 + i * 82,
                  from = 42 + q * 82;
                return (
                  <g key={i}>
                    <motion.path
                      initial={false}
                      animate={{
                        opacity: selfMask[q][i] ? 1 : 0.08,
                        pathLength: selfMask[q][i] ? 1 : 0,
                      }}
                      d={`M${from} 32 Q${(x + from) / 2} 140 ${x} 32`}
                      fill="none"
                      stroke={i === q ? "#b8934f" : "#6fa8dc"}
                      strokeWidth="2"
                    />
                    <circle
                      cx={x}
                      cy="32"
                      r="8"
                      fill={i === q ? "#c8a96a" : "#d9e8f0"}
                    />
                    <text x={x} y="18" textAnchor="middle">
                      {i}
                    </text>
                  </g>
                );
              })}
              <text x="14" y="134" className="subtext">
                Query {q} → {selfMask[q].reduce((a, b) => a + b, 0)} visible
                positions
              </text>
            </svg>
          </div>
          <div className="stack">
            <h3>
              {family === "encoder-only"
                ? "Bidirectional self-attention"
                : "Causal self-attention"}
            </h3>
            <Matrix rows={selfMask} selected={q} />
            <Note>
              1 = 可读取，0 = 屏蔽。点击左侧 token，观察该行的信息读取范围。
            </Note>
            {family === "encoder-decoder" && (
              <>
                <h3>Cross-attention · target → source</h3>
                <Matrix
                  rows={mask("cross", spec.tokens.length, source.length)}
                  rowLabels={spec.tokens}
                  colLabels={source}
                  selected={q}
                />
                <Note>
                  目标 Query 可读取所有源位置；源句与目标句是两条不同的序列。
                </Note>
              </>
            )}
          </div>
        </div>
        <div className="tableScroll">
          <table className="compare">
            <thead>
              <tr>
                <th>Family</th>
                <th>Information flow</th>
                <th>Typical task</th>
                <th>Cache behavior</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Encoder-only</td>
                <td>双向 self-attention</td>
                <td>分类、检索、掩码填空</td>
                <td>输入变化时通常重新编码</td>
              </tr>
              <tr>
                <td>Decoder-only</td>
                <td>因果 self-attention</td>
                <td>自回归文本与代码生成</td>
                <td>各层历史 KV 增量复用</td>
              </tr>
              <tr>
                <td>Encoder–Decoder</td>
                <td>源端双向 + 目标端因果 + cross-attention</td>
                <td>翻译、条件生成、摘要</td>
                <td>源端 cross KV 固定，目标端 self KV 增长</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>
    </Shell>
  );
}
