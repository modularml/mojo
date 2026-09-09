# Accuracy smoke tests

End-to-end accuracy smoke tests: they start a model server (MAX Serve by
default), drive it with `lm-eval` chat-completion requests, and report accuracy
(higher is better; `1.0` is 100%). Each model argument is either a HuggingFace
path or a registered variant that maps to a recipe in `MODEL_RECIPES`
(`smoke_test.py`); vision models additionally run a vision task (`chartqa`).

## Running locally

```bash
# A plain HuggingFace model (320 questions by default):
./bazelw run //:smoke-test -- meta-llama/Llama-3.2-1B-Instruct

# A registered, recipe-backed variant, with fewer questions for a faster loop:
./bazelw run //:smoke-test -- google/gemma-4-31B-it__tuned \
  --num-questions 100 --max-concurrent 8
```

Common flags: `--num-questions N`, `--max-concurrent K`, `--print-responses`,
`--print-cot`, `--serve-extra-args "..."`, `--disable-timeouts`. Only models
with a chat template (typically `-instruct`, `-it`, `-chat`) are supported.

The first local run compiles the model graph, which can take several minutes for
a large model (subsequent runs reuse the compile cache).

## What CI runs: nightly vs manual

`smoke_test_github_matrix.py` holds two model sets:

- `MODELS` is every model CI knows how to smoke test, with its per-framework
  and per-GPU exclusions.
- `NIGHTLY_MODELS` is the subset the nightly suite runs (`--tier nightly`).

To run anything outside `NIGHTLY_MODELS`, dispatch the Serve Smoke Tests
workflow (`.github/workflows/serveSmokeTest.yaml`) with `model_tier: all`, the
default for a manual run, or name the models in `models_override`.

Membership in `NIGHTLY_MODELS` is opt-in: adding a model to `MODELS` does not
add it to the nightly.

## Task overrides

By default the harness runs `gsm8k_cot_llama` (plus `chartqa` for vision
models), which is what automated CI uses. To run an eval that isn't in that
automated CI set — say a long-context sweep — pass `--override-tasks <task>`,
which selects the tasks you name instead of the model-derived ones. Some tasks
are parameterized at runtime via `--lm-eval-metadata '<json>'`, which is
forwarded verbatim to lm-eval's `--metadata`; for example, babilong's context
length is set with `--lm-eval-metadata '{"max_seq_lengths": "16k"}'`.

`--override-tasks` accepts only a curated allowlist (enforced by
`click.Choice`); arbitrary lm-eval task names are rejected. We restrict it
because not every lm-eval task runs cleanly here — some need optional
dependencies absent from the bazel build, and some use output types the
chat-completions endpoint can't serve. The allowlisted tasks are the ones we've
confirmed work end-to-end and give a meaningful accuracy signal.

```bash
# babilong two-fact reasoning (qa2) at 16k context:
./bazelw run //:smoke-test -- <model> \
  --override-tasks babilong_qa2 \
  --lm-eval-metadata '{"max_seq_lengths": "16k"}'
```

## Gotchas when running locally

These bite when running smoke tests outside CI. None are model bugs.

### `HF_TOKEN` must be set, even for a public model

`validate_hf_token()` (`eval_runner.py`) hard-exits before the server starts if
`HF_TOKEN` is unset. The token is not used to authenticate a public
(non-gated) model, but the environment variable must be present:

```bash
export HF_TOKEN=hf_...   # any value works for a public model
```

### `--serve-extra-args` for local serve tuning

Pass extra flags through to MAX Serve when startup fails or dies during warmup.
Combine multiple flags in one quoted string.

**KV-head count must divide the device count.** Otherwise startup fails
immediately (for example, `4 KV heads not divisible by 8 devices`). Pin the
device set so it divides evenly:

```bash
# gemma-4-31B has 4 KV heads, so use 4 GPUs (8 fails):
--serve-extra-args "--devices gpu:0,1,2,3"
```

**Device-graph-capture warmup can run out of memory.** A model that fills most
of device memory can die during `Capturing device graph shapes`, often with no
traceback (the GPUs free and the process exits). Graph capture is a latency
optimization and does not affect accuracy. Try one of:

```bash
--serve-extra-args "--no-device-graph-capture"
# or lower reserved memory:
--serve-extra-args "--device-memory-utilization 0.85"
# or cap batch size so capture shapes need less KV headroom:
--serve-extra-args "--max-batch-size=16"
```

### KV-connector recipes force connector-only prefix hits

For any recipe with a `kv_connector` (`tiered` or `rust_tiered`), the harness
sets `MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE=1` (`smoke_test.py`). This
disables the device prefix cache so every prefix-cache hit is served through the
connector — intentional, so the test exercises the CPU/disk offload path it is
meant to cover. Per-step connector metrics (`D2H`/`H2D` blocks, disk
reads/writes, hit rate) are printed in the server log.

### Local accuracy severity can be milder than CI

For KV-cache-offload accuracy regressions, the local manifestation can be much
milder than CI or production, because severity scales with eviction pressure
(batch size, sequence mix, cache size relative to the working set). Treat a
local accuracy delta as directional, and confirm worst-case recovery on the
production configuration in CI.
