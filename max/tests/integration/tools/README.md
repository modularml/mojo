# MAX integration tools

This directory contains utilities for testing, debugging, and validating MAX
pipelines.

For a complete walkthrough of debugging model accuracy issues, see the
[accuracy debugging guide](/max/docs/accuracy-debugging.md).

## `download_models_for_testing.py`

Downloads one or more Hugging Face model repos into the local cache. This is
useful before running integration tests that resolve models through
`generate_local_model_path()`, especially on a fresh machine where the cache
has not been populated yet.

**Basic usage:**

```bash
# Download a repo at its default revision
bazel run //max/tests/integration/tools:download_models_for_testing -- \
    modularai/Llama-3.1-8B-Instruct-GGUF

# Download multiple repos in one run
bazel run //max/tests/integration/tools:download_models_for_testing -- \
    meta-llama/Llama-3.2-1B-Instruct \
    openai/whisper-large-v3

# Override the revision explicitly
bazel run //max/tests/integration/tools:download_models_for_testing -- \
    meta-llama/Llama-3.2-1B-Instruct@main
```

**Notes:**

- If `HF_TOKEN` is unset, public repos can still download, but gated/private
  repos will fail.
- Downloaded snapshots land in the standard Hugging Face cache used by tests.

## `debug_model.py`

The primary tool for inspecting intermediate tensors during model execution.
It automatically attaches print hooks to all layers and outputs their
inputs/outputs. Supports MAX, PyTorch, and vLLM frameworks.

**Basic usage:**

```bash
# Run with MAX framework (default uses 1 hidden layer)
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max \
    --pipeline google/gemma-3-1b-it

# Run with PyTorch framework
bazel run //max/tests/integration/tools:debug_model -- \
    --framework torch \
    --pipeline google/gemma-3-1b-it

# Export tensors to files for comparison
bazel run //max/tests/integration/tools:debug_model -- \
    --framework max \
    --pipeline google/gemma-3-1b-it \
    -o /tmp/max_tensors/
```

**Key options:**

| Option                         | Description                                                                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `--framework {max,torch,vllm}` | Framework to run the model with                                                                                                              |
| `--pipeline NAME`              | Hugging Face model path or pipeline oracle key                                                                                               |
| `--device DEVICE`              | Device type: `cpu`, `gpu`, `default`, or `gpu:0,1` for specific GPUs                                                                         |
| `-o, --output DIR`             | Save full tensors to directory (`.pt`/`.max` files for `compare_tensors`). Without this flag, prints abbreviated representations to console. |
| `--num-hidden-layers N`        | Number of hidden layers to use (default: 1, use `all` for full model)                                                                        |
| `--num-steps N`                | Number of inference steps to run (default: 1)                                                                                                |
| `--prompt TEXT`                | Custom prompt text (uses pipeline default if omitted)                                                                                        |
| `--image URL`                  | Image URL for multimodal models (can be repeated)                                                                                            |
| `--encoding NAME`              | Quantization encoding (such as `bfloat16`, `float32`)                                                                                        |
| `--hf-config-overrides JSON`   | JSON dict of Hugging Face config overrides                                                                                                   |

Use `--help` for the complete list of options.

## `compare_tensors.py`

Compares tensor files from MAX and PyTorch to quantify numerical differences.
Supports comparing individual files or auto-matching tensors by name across
directories.

File formats (for input):

- `.max` files for MAX tensors (generated via `debug_model -o` and
  `TensorValue.print()`)
- `.pt` files for PyTorch tensors

**Basic usage:**

When you name tensors consistently in both frameworks, the tool automatically
pairs them by name when you pass the path to all tensor files:

```bash
bazel run //max/tests/integration/tools:compare_tensors -- \
    --torch-tensor torch_tensors/ \
    --max-tensor max_tensors/
```

You can also specify individual tensors to compare:

```bash
bazel run //max/tests/integration/tools:compare_tensors -- \
    --torch-tensor torch_tensors/mlp_gate.pt \
    --max-tensor max_tensors/mlp_gate.max \
    --rtol 1e-5 --atol 1e-8
```

**Key options:**

| Option                | Description                                                          |
|-----------------------|----------------------------------------------------------------------|
| `--torch-tensor PATH` | Path to PyTorch tensor file (`.pt`) or directory                     |
| `--max-tensor PATH`   | Path to MAX tensor file (`.max`) or directory                        |
| `--rtol FLOAT`        | Relative tolerance for pass/fail check                               |
| `--atol FLOAT`        | Absolute tolerance for pass/fail check                               |
| `--allow-reshape`     | Allow comparing tensors with different shapes but same element count |

**Output includes:**

- Shape comparison
- Greatest absolute difference and its location
- Greatest relative difference and its location
- Pass/fail status when tolerances are specified
- Percentage of mismatched elements

## `generate_llm_logits.py`

Generates logit golden files for comparing model outputs across frameworks.
Used by the pipeline verification system to validate model accuracy.

**Basic usage:**

```bash
# Generate MAX logits
bazel run //max/tests/integration/tools:generate_llm_logits -- \
    --framework max \
    --pipeline google/gemma-3-1b-it \
    --output /tmp/max_logits.json

# Generate PyTorch reference logits
bazel run //max/tests/integration/tools:generate_llm_logits -- \
    --framework torch \
    --pipeline google/gemma-3-1b-it \
    --output /tmp/torch_logits.json
```

Use `--help` for more options including batch size, encoding, and reference
comparison.
