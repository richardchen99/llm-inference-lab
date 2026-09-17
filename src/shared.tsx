import { useEffect, useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import {
  ArrowUpRight,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "./style.css";

export const labs = [
  ["position-encoding-lab", "Position Encoding", "01"],
  ["llm-inference-lab", "LLM Inference", "02"],
  ["tokenizer-playground", "Tokenizer", "03"],
  ["llm-rl-lab", "LLM RL", "04"],
];

export function Shell({
  slug,
  title,
  subtitle,
  children,
  sources,
}: {
  slug: string;
  title: string;
  subtitle: string;
  children: ReactNode;
  sources: [string, string][];
}) {
  return (
    <MotionConfig
      reducedMotion="user"
      transition={{ type: "spring", stiffness: 180, damping: 25 }}
    >
      <a className="skip" href="#experiment">
        Skip to experiment
      </a>
      <header className="topbar">
        <a className="brand" href="https://richardchen99.github.io">
          <Sparkles size={20} />
          <span>
            RICHARD CHEN<small>Intelligence, made visible.</small>
          </span>
        </a>
        <nav aria-label="Research labs">
          {labs.map(([id, name, num]) => (
            <a
              key={id}
              aria-current={id === slug ? "page" : undefined}
              href={`https://richardchen99.github.io/${id}/`}
            >
              <small>{num}</small>
              {name}
            </a>
          ))}
        </nav>
        <a
          className="textLink"
          href="https://richardchen99.github.io/transformer-architecture-lab/"
        >
          Transformer <ArrowUpRight size={14} />
        </a>
      </header>
      <main>
        <motion.section
          className="hero"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="eyebrow">
            <span className="dot" /> INTERACTIVE RESEARCH SERIES /{" "}
            {labs.find((l) => l[0] === slug)?.[2]}
          </div>
          <h1>{title}</h1>
          <p className="intro">{subtitle}</p>
          <div className="byline">
            <span>
              Built by <strong>Richard Chen</strong> · 中国人民大学
            </span>
            <a
              href="https://richardchen99.github.io"
              target="_blank"
              rel="noreferrer"
            >
              Personal research <ArrowUpRight size={13} />
            </a>
          </div>
        </motion.section>
        <div id="experiment">{children}</div>
        <section className="references">
          <span className="eyebrow">
            <BookOpen size={14} /> RESEARCH NOTES
          </span>
          <div>
            {sources.map(([name, href]) => (
              <a href={href} key={href} target="_blank" rel="noreferrer">
                {name}
                <ArrowUpRight size={14} />
              </a>
            ))}
          </div>
        </section>
      </main>
      <footer>
        <span>Richard Chen · Renmin University of China</span>
        <a href={`https://github.com/richardchen99/${slug}`}>
          Explore the project <ArrowUpRight size={13} />
        </a>
      </footer>
    </MotionConfig>
  );
}

export function Formula({ tex, label }: { tex: string; label?: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: true,
        throwOnError: true,
        trust: false,
        output: "htmlAndMathml",
      }),
    [tex],
  );
  return (
    <div className="equation">
      <div
        className="equationScroll"
        tabIndex={0}
        role="region"
        aria-label={label ?? "Mathematical formula"}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
export function Panel({
  title,
  eyebrow,
  children,
  className = "",
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panelTitle">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}
export function Tabs({
  options,
  value,
  onChange,
  label,
}: {
  options: { id: string; label: string }[];
  value: string;
  onChange: (s: string) => void;
  label: string;
}) {
  return (
    <div className="tabs" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.id}
          aria-pressed={value === o.id}
          className={value === o.id ? "selected" : ""}
          onClick={() => onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
export function Range({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = "",
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
  unit?: string;
}) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <label className="range">
      <span>
        {label}
        <b>
          {value}
          {unit}
        </b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{
          backgroundImage: `linear-gradient(to right, #c8a96a ${percent}%, #dfe5e8 ${percent}%)`,
        }}
        onChange={(e) => onChange(+e.target.value)}
      />
    </label>
  );
}
export function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail?: string;
}) {
  return (
    <div className="stat">
      <small>{label}</small>
      <strong>{value}</strong>
      {detail && <span>{detail}</span>}
    </div>
  );
}
export function Bars({
  items,
  signed = false,
}: {
  items: { label: string; value: number; active?: boolean; display?: string }[];
  signed?: boolean;
}) {
  const scale = signed
    ? Math.max(1, ...items.map((i) => Math.abs(i.value)))
    : Math.max(1, ...items.map((i) => i.value));
  return (
    <div className="bars">
      {items.map((i, index) => (
        <div
          className={`barRow ${i.active ? "active" : ""}`}
          key={`${i.label}-${index}`}
        >
          <span title={i.label}>{i.label}</span>
          <div className={signed ? "barTrack signed" : "barTrack"}>
            <motion.i
              initial={false}
              animate={{
                width: `${(Math.abs(i.value) / scale) * (signed ? 50 : 100)}%`,
                left: signed
                  ? `${i.value < 0 ? 50 - (Math.abs(i.value) / scale) * 50 : 50}%`
                  : "0%",
              }}
              className={i.value < 0 ? "negative" : ""}
            />
          </div>
          <b>{i.display ?? i.value.toFixed(3)}</b>
        </div>
      ))}
    </div>
  );
}
export function Tokens({
  tokens,
  active = -1,
  onSelect,
  label = "Tokens",
}: {
  tokens: string[];
  active?: number;
  onSelect?: (i: number) => void;
  label?: string;
}) {
  return (
    <div className="tokens" role="group" aria-label={label}>
      <AnimatePresence initial={false} mode="popLayout">
        {tokens.map((t, i) =>
          onSelect ? (
            <motion.button
              layout="position"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8 }}
              key={`${i}-${t}`}
              className={i === active ? "token active" : "token"}
              onClick={() => onSelect(i)}
              aria-pressed={i === active}
            >
              <small>{i}</small>
              {t}
            </motion.button>
          ) : (
            <motion.span
              layout="position"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8 }}
              key={`${i}-${t}`}
              className={i === active ? "token active" : "token"}
            >
              <small>{i}</small>
              {t}
            </motion.span>
          ),
        )}
      </AnimatePresence>
    </div>
  );
}
export function Transport({
  step,
  total,
  playing,
  onPlay,
  onStep,
  onReset,
  speed,
  onSpeed,
  labels,
}: {
  step: number;
  total: number;
  playing: boolean;
  onPlay: () => void;
  onStep: (n: number) => void;
  onReset: () => void;
  speed: number;
  onSpeed: (n: number) => void;
  labels?: string[];
}) {
  return (
    <div className="transport">
      <div className="transportButtons">
        <button
          className="iconButton"
          onClick={onReset}
          aria-label="Reset experiment"
        >
          <RotateCcw size={16} />
        </button>
        <button
          className="iconButton"
          disabled={step === 0}
          onClick={() => onStep(step - 1)}
          aria-label="Previous step"
        >
          <ChevronLeft size={17} />
        </button>
        <button className="primary" disabled={total === 0} onClick={onPlay}>
          {playing ? <Pause size={15} /> : <Play size={15} />}{" "}
          {playing ? "Pause" : step === total ? "Replay" : "Play"}
        </button>
        <button
          className="iconButton"
          disabled={step >= total}
          onClick={() => onStep(step + 1)}
          aria-label="Next step"
        >
          <ChevronRight size={17} />
        </button>
      </div>
      <div className="transportProgress">
        <span>{labels?.[step] ?? `Step ${step} / ${total}`}</span>
        <div>
          <motion.i
            animate={{ width: `${total ? (step / total) * 100 : 0}%` }}
          />
        </div>
      </div>
      <Range
        label="Speed"
        min={0.5}
        max={2}
        step={0.25}
        value={speed}
        onChange={onSpeed}
        unit="×"
      />
    </div>
  );
}
export function usePlayback(total: number, resetKey: string, delay = 1400) {
  const [step, setStep] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1);
  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [resetKey]);
  useEffect(() => {
    if (!playing || total === 0) return;
    if (step >= total) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => setStep((s) => Math.min(total, s + 1)),
      delay / speed,
    );
    return () => window.clearTimeout(timer);
  }, [step, playing, speed, total, delay]);
  return {
    step: Math.min(total, step),
    total,
    playing,
    speed,
    onSpeed: setSpeed,
    onPlay: () => {
      if (step >= total) setStep(0);
      setPlaying((v) => !v);
    },
    onStep: (n: number) => {
      setPlaying(false);
      setStep(Math.min(total, Math.max(0, n)));
    },
    onReset: () => {
      setPlaying(false);
      setStep(0);
    },
  };
}
export function Note({ children }: { children: ReactNode }) {
  return <p className="note">{children}</p>;
}
export function Matrix({
  rows,
  rowLabels,
  colLabels,
  selected = -1,
}: {
  rows: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  selected?: number;
}) {
  return (
    <div
      className="tableScroll"
      tabIndex={0}
      role="region"
      aria-label="Numeric matrix"
    >
      <table>
        <thead>
          <tr>
            <th scope="col">Q / K</th>
            {(rows[0] ?? []).map((_, i) => (
              <th key={i} scope="col">
                {colLabels?.[i] ?? i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={selected === i ? "selectedRow" : ""}>
              <th scope="row">{rowLabels?.[i] ?? i}</th>
              {r.map((v, j) => (
                <td
                  key={j}
                  style={{
                    background: Number.isFinite(v)
                      ? `rgba(111,168,220,${0.04 + Math.min(1, Math.abs(v)) * 0.35})`
                      : "rgba(200,111,106,.09)",
                  }}
                >
                  {Number.isFinite(v) ? v.toFixed(2) : "−∞"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
