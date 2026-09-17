import test from "node:test";
import assert from "node:assert/strict";
import { model as m } from "../scripts/load-model.mjs";
test("Cached two-layer decoding matches full recomputation at every token", () => {
  for (const prompt of [
    ["The", "cat", "sat"],
    ["def", "f", "(", "x", ")"],
    ["金融", "研究"],
  ]) {
    const t = m.trace(prompt, ["on", "the", "mat", "."]);
    for (const step of t) {
      assert.ok(step.error < 1e-12);
      for (const row of step.weights)
        assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    }
    for (let i = 1; i < t.length; i++)
      for (let l = 0; l < 2; l++)
        assert.deepEqual(
          t[i].cache[l].keys.slice(0, -1),
          t[i - 1].cache[l].keys,
        );
  }
});
test("Future tokens do not alter causal prefix states, but alter bidirectional states", () => {
  const a = m.full(["one", "two"]),
    b = m.full(["one", "two", "three"]);
  assert.deepEqual(a.hidden, b.hidden.slice(0, 2));
  assert.notDeepEqual(
    m.full(["one", "two"], false).hidden,
    m.full(["one", "two", "three"], false).hidden.slice(0, 2),
  );
});
test("Cache byte accounting includes both K/V, all layers and batches", () => {
  assert.equal(m.cacheBytes(32, 8, 128, 4096, 2), 536870912);
  assert.equal(m.cacheBytes(32, 8, 128, 4096, 2, 3), 1610612736);
});
test("Family masks preserve causal and cross-attention shapes", () => {
  assert.deepEqual(m.mask("decoder-only", 3), [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
  ]);
  assert.deepEqual(m.mask("cross", 2, 3), [
    [1, 1, 1],
    [1, 1, 1],
  ]);
});
