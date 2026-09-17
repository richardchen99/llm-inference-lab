export const dot = (a: number[], b: number[]) =>
  a.reduce((s, v, i) => s + v * b[i], 0);
export function softmax(x: number[]) {
  const m = Math.max(...x),
    e = x.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0)),
    s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => (s ? v / s : 0));
}
export function embed(token: string, pos: number) {
  const seed = Array.from(token).reduce((s, c) => s + c.codePointAt(0)!, 0);
  return Array.from(
    { length: 4 },
    (_, d) =>
      Math.sin(seed * 0.021 + d * 0.6) * 0.4 + Math.cos(pos * 0.15 + d) * 0.15,
  );
}
export function project(x: number[], layer: number, kind: number) {
  return x.map((_, j) =>
    x.reduce(
      (s, v, i) =>
        s +
        v * (i === j ? 0.8 : Math.sin((i + 1) * (j + 2) + kind + layer) * 0.16),
      0,
    ),
  );
}
function finish(x: number[], z: number[]) {
  const r = x.map((v, i) => v + z[i]);
  return r.map(
    (v, i) => v + 0.2 * Math.tanh(v * 0.7 + r[(i + 1) % r.length] * 0.2),
  );
}
export type CacheLayer = { keys: number[][]; values: number[][] };
export function full(tokens: string[], causal = true) {
  let hidden = tokens.map(embed);
  const cache: CacheLayer[] = [];
  const weights: number[][][] = [];
  for (let layer = 0; layer < 2; layer++) {
    const q = hidden.map((v) => project(v, layer, 0)),
      keys = hidden.map((v) => project(v, layer, 1)),
      values = hidden.map((v) => project(v, layer, 2));
    const a = q.map((v, i) =>
      softmax(
        keys.map((k, j) => (causal && j > i ? -Infinity : dot(v, k) / 2)),
      ),
    );
    hidden = hidden.map((v, i) =>
      finish(
        v,
        v.map((_, d) => a[i].reduce((s, w, j) => s + w * values[j][d], 0)),
      ),
    );
    cache.push({ keys, values });
    weights.push(a);
  }
  return { hidden, cache, weights };
}
export function append(token: string, pos: number, previous: CacheLayer[]) {
  let x = embed(token, pos);
  const cache: CacheLayer[] = [],
    weights: number[][] = [];
  for (let layer = 0; layer < 2; layer++) {
    const q = project(x, layer, 0),
      key = project(x, layer, 1),
      value = project(x, layer, 2),
      keys = [...previous[layer].keys, key],
      values = [...previous[layer].values, value];
    const a = softmax(keys.map((k) => dot(q, k) / 2));
    x = finish(
      x,
      x.map((_, d) => a.reduce((s, w, j) => s + w * values[j][d], 0)),
    );
    cache.push({ keys, values });
    weights.push(a);
  }
  return { hidden: x, cache, weights };
}
export function trace(prompt: string[], continuation: string[]) {
  const initial = full(prompt);
  let cache = initial.cache;
  return [
    {
      tokens: prompt,
      cache,
      hidden: initial.hidden.at(-1)!,
      weights: initial.weights.map((l) => l.at(-1)!),
      error: 0,
    },
    ...continuation.map((token, i) => {
      const cached = append(token, prompt.length + i, cache);
      cache = cached.cache;
      const tokens = [...prompt, ...continuation.slice(0, i + 1)],
        ref = full(tokens).hidden.at(-1)!;
      return {
        tokens,
        cache,
        hidden: cached.hidden,
        weights: cached.weights,
        error: Math.max(...ref.map((v, d) => Math.abs(v - cached.hidden[d]))),
      };
    }),
  ];
}
export function cacheBytes(
  layers: number,
  kvHeads: number,
  headDim: number,
  tokens: number,
  bytes: number,
  batch = 1,
) {
  return 2 * layers * kvHeads * headDim * tokens * bytes * batch;
}
export function mask(family: string, n: number, m = n): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) =>
      family === "decoder-only" && j > i ? 0 : 1,
    ),
  );
}
