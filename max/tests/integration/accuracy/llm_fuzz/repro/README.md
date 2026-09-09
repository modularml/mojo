# CENG-781 — MiniMax-M3 adaptive-thinking empty-think repro

Reproduction kit for [CENG-781](https://linear.app/modularml/issue/CENG-781):
MiniMax-M3 (MXFP8) intermittently fails the MiniMax-Provider-Verifier
`test_15_04_extreme_agent_thinking` — the streamed response contains no thinking
block a few percent of the time.

## What the failure is

The verifier's only real assertion is `assert_thinking_present`, which passes if
the response has a `<think>` tag in `content` **or** a non-empty
`reasoning_content`/`reasoning`. The request sends `thinking: {"type":
"adaptive"}` (the model decides whether to reason). A failure is a clean HTTP
200 with `finish_reason="stop"` and **no thinking block of any kind**.

Note that the `<think>`-in-`content` branch is lenient: a raw `<think>` tag that
*leaks* into the visible answer still counts as "thinking present." That is the
upstream verifier's behavior, not ours — this kit mirrors
`assert_thinking_present` exactly so the guard matches what actually gates the
MiniMax-Provider-Verifier. Flagging a leaked tag as its own failure mode would
be a separate, stricter check.

MiniMax's own endpoint passes 100/100. On MAX it failed ~3–7% on one replica.

## Leading hypothesis (investigation still open)

**This is not a closed investigation.** Treat what follows as the strongest
hypothesis so far, not a confirmed root cause — don't take it as ground truth.
This kit exists to let anyone re-run the experiment and push the investigation
forward, not to assert the conclusion.

The failure correlated strongly with a **single replica** and behaved as if that
pod carried a **stale/bad prefix-cache (tiered-KV) state**, surfacing only under
concurrent batching. What we were able to rule out:

- Not sampling — forcing `top_p=0.95, top_k=40` did not change the rate.
- Not the code — a local build of the exact deployed commit did not reproduce.
- Not the GPUs — driver/VBIOS/ECC/NVLink identical across replicas, and a
  cross-node determinism test produced **bit-identical** bf16/fp8/reduction
  results on all GPUs.
- Not load — the failing and healthy replicas had near-identical real traffic.
- It reproduced only under concurrency (batch mixing), never sequentially
  (batch size 1).

What worked as a **mitigation**: `POST /reset_prefix_cache` on the failing pod
took it from ~6.4% to **0/1000** (same pod, same session, no restart). That
points at the prefix-cache / tiered-KV reuse path, but does not by itself prove
causation — the flush also perturbs batching and timing.

Open questions this kit has **not** answered:

- Why would a stale/bad prefix-cache state change only
  *whether the model emits a thinking block*, rather than corrupting the output
  more broadly?
- Was the cached state merely stale (should have been evicted/cleared) or
  actually corrupted? The two have different durable fixes.

## Reproduce

### 1. Point at a server

Port-forward a prod pod (no API key needed against the raw engine):

```bash
kubectl -n org-modular--prod-1-mammoth port-forward \
    pod/minimax-m3-mxfp8-engine-<hash>-<id> 8000:8000
```

Or serve MiniMax-M3 locally on 8×B200 (see the deployment recipe).

### 2. Run the standalone script (stdlib only, no pip installs)

```bash
# Reproduce (all-identical == prefix-cache-HIT path), 500 requests @ conc 50
python ceng781_thinking_repro.py --url http://localhost:8000 --total 500 --conc 50

# Cache-HIT vs cache-MISS A/B in one run (unique prefix forces a cache miss)
python ceng781_thinking_repro.py --url http://localhost:8000 --total 500 --prepend-ratio 0.5

# Confirm the fix: flush the prefix cache first, then re-run
python ceng781_thinking_repro.py --url http://localhost:8000 --flush --total 500
```

The script prints the "thinking absent" rate split by cache-HIT vs cache-MISS.
A healthy replica reports 0%; the failing replica reported ~6%.

Useful flags: `--think {adaptive,enabled,disabled}` (`enabled` forces a thinking
block and should always be 0% present-absent; `disabled` should be ~100%
absent, a sanity check on the detector), `--top-p` / `--top-k` (verified not to
help), `--flush` (`POST /reset_prefix_cache`).

### 3. Or run it through llm-fuzz (framework scenario)

The same check is packaged as the `adaptive_thinking_presence` scenario, which
runs both the cache-HIT and cache-MISS paths and fails above a threshold.
Because adaptive thinking is a MiniMax-M3 feature, the scenario is gated to the
`minimax-m3` model profile: a default `llm-fuzz` run only executes it under
`--model-profile minimax-m3`, but naming it explicitly with `--scenarios
adaptive_thinking_presence` (as below) runs it regardless of profile.

```bash
LLM_FUZZ="./bazelw run //max/tests/integration/accuracy:llm-fuzz --"
$LLM_FUZZ --url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \
    --scenarios adaptive_thinking_presence

# Tune it:
LLM_FUZZ_ADAPTIVE_THINKING_RUNS=200 \
LLM_FUZZ_ADAPTIVE_THINKING_CONC=50 \
LLM_FUZZ_ADAPTIVE_THINKING_MAXPCT=1.0 \
    $LLM_FUZZ --url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \
    --scenarios adaptive_thinking_presence
```

## Interpreting results

- **Both paths 0%** — healthy (or the prefix cache is currently clean; flush and
  concurrency are needed to expose a bad state).
- **cache-HIT fails, cache-MISS clean** — the regression rides on prefix-cache
  reuse (the CENG-781 signature). Flush the prefix cache to mitigate.
- **Both fail** — not prefix-cache-specific; look at the reasoning parser /
  sampling / decode path.
