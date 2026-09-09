# Benchmark MAX

This directory contains tools to benchmark the performance of an LLM model
server—measuring throughput, latency, and resource utilization. You can use
these scripts to compare other serving backends, namely
[vLLM](https://github.com/vllm-project/vllm), against MAX.

The `benchmark_serving.py` script is adapted from
[vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks),
licensed under Apache 2.0. We forked this script to ensure consistency with
vLLM's measurement methodology and extended it with features we found helpful,
such as client-side GPU metric collection via `max.profiler`.

`benchmark_serving.py` supports:

- text generation
- text-to-image generation
- image-to-image generation via `/v1/responses`

For image-to-image benchmarks, use:

- `--dataset-name local-image --dataset-path /path/to/file.jsonl` for a generic
  local JSONL dataset with `prompt` and `image_path` rows
- `--dataset-name synthetic-pixel` for a synthetic image-edit workload backed
  by a generated local placeholder image

Example `local-image` JSONL:

```json
{"prompt": "Turn this into watercolor", "image_path": "images/sample.png"}
{"prompt": "Replace the sky with a sunset", "image_path": "/abs/path/to/photo.jpg"}
```

`synthetic-pixel` is analogous to the placeholder-image path used in diffusion
serving benchmarks such as vLLM-Omni: MAX generates a white PNG in the system
temp directory and reuses it for each request in the run.

For `benchmark_serving.py` usage instructions, see [Benchmarking a MAX
endpoint](/max/docs/max-benchmarking.md).

To inject arbitrary top-level request-body fields that lack a dedicated flag
(for example `stop`, `include_stop_str_in_output`, or `chat_template_kwargs`),
use `--extra-body` with an inline JSON string or a path to a YAML/JSON file:

```bash
--extra-body '{"stop": ["}"], "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

Fields are merged onto every text-generation request (last-writer-wins, so a key
that collides with a dedicated flag overrides it). See the [benchmark
configuration guide](benchmark_config.md) for the config-file form and details.

> [!NOTE]
> This benchmarking script is also available with the `max benchmark` command,
> which you can get by installing `modular` with pip, uv, conda, or pixi
> package managers. Try it now by following the detailed guide to [benchmark
> MAX on GPUs](https://max.modular.com/serve/benchmark/).
