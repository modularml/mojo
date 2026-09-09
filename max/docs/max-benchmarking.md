# Benchmarking a MAX endpoint

> [!CAUTION]
> This guide is for MAX framework developers (Git repo developers).
> If you're using MAX as an application developer (via the `modular` package),
> you should instead use the `max benchmark` command line tool—see the more
> detailed guide to [benchmark MAX on
> GPUs](https://max.modular.com/serve/benchmark/).

This guide describes how to benchmark the performance of a MAX model
server—measuring throughput, latency, and resource utilization—using
`benchmark_serving.py`. You can use this script to compare other serving
backends, namely [vLLM](https://github.com/vllm-project/vllm), against MAX.

Key features:

- Tests any OpenAI-compatible HTTP endpoint
- Supports both chat and completion APIs
- Measures detailed latency metrics
- Works with hosted services

> The `benchmark_serving.py` script is adapted from
> [vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks),
> licensed under Apache 2.0. We forked this script to ensure consistency with
> vLLM's measurement methodology and extended it with features we found helpful,
> such as client-side GPU metric collection via `max.profiler`.

## Basic usage

You can benchmark any HTTP endpoint that implements
OpenAI-compatible APIs as follows:

```bash
python benchmark_serving.py \
  --model google/gemma-3-27b-it \
  --backend modular \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --num-prompts 500 \
  --sonnet-input-len 512 \
  --output-lengths 256 \
  --sonnet-prefix-len 200
```

To see all the available options, run:

```sh
python benchmark_serving.py --help
```

For more information, see the [`max benchmark`
documentation](https://max.modular.com/cli/benchmark).

## Config files

When you want to save your benchmark configurations, you can define them in a
YAML file and pass it with the `--config-file` option.

The YAML file supports all the same properties as the `benchmark_serving.py`
arguments, except in the YAML file the properties **must use `snake_case`
names** instead of hyphenated names—for example, `--num-prompts` becomes
`num_prompts`. And all properties must be nested under a top level
`benchmark_config` key. For example:

```sh
benchmark_config:
  model: google/gemma-3-27b-it
  backend: modular
  endpoint: /v1/chat/completions
  dataset_name: sonnet
  ...
```

To help you reproduce our own benchmarks, we've made some of our config files
available in the [`configs`](../python/max/benchmark/configs/) directory. For
example, copy our
[`gemma-3-27b-sonnet-decode-heavy-prefix200.yaml`](https://github.com/modular/modular/tree/main/benchmark/configs)
file from GitHub, and you can benchmark Gemma3-27B with this command:

```sh
max benchmark --config-file gemma-3-27b-sonnet-decode-heavy-prefix200.yaml
```

## Output

Results are printed to the terminal but you can also save a JSON-formatted
file by providing a path with `--result-filename`. The path can include a
directory prefix, for example `--result-filename results/my-benchmark.json`.

The output in the terminal should look similar to the following:

```bash
============ Serving Benchmark Result ============
Successful requests:                     500
Failed requests:                         0
Benchmark duration (s):                  46.27
Total input tokens:                      100895
Total generated tokens:                  106511
Request throughput (req/s):              10.81
------------- Client Experience Metrics -------------
Total TPM (input+output, whole bench):   268435.20
Global Cached Token Rate:                12.50%
------------ Latency & Throughput Percentiles -------------
┌───────────────────────────┬──────────┬─────────┬──────────┬──────────┬──────────┬──────────┐
│ Metric                    │     Mean │     Std │      P50 │      P90 │      P95 │      P99 │
├───────────────────────────┼──────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
│ TTFT (ms)                 │ 15539.31 │ 4200.50 │ 15068.37 │ 28000.00 │ 31000.00 │ 33034.17 │
│ TPOT (ms)                 │    34.23 │   18.10 │    28.47 │    60.20 │    95.40 │   138.55 │
│ ITL (ms)                  │    26.76 │   22.40 │     5.42 │    48.90 │   120.30 │   228.45 │
│ Request Latency (ms)      │ 20345.10 │ 5120.30 │ 19980.40 │ 30120.50 │ 33450.10 │ 35880.90 │
│ Input throughput (tok/s)  │  2180.51 │  210.30 │  2200.10 │  1980.40 │  1900.20 │  1820.60 │
│ Output throughput (tok/s) │  2301.89 │  180.70 │  2320.50 │  2100.30 │  2010.80 │  1950.40 │
└───────────────────────────┴──────────┴─────────┴──────────┴──────────┴──────────┴──────────┘
  TPOT = (latency - TTFT) / (output_tokens - 1)
  Throughput P90/P95/P99 are reversed (bottom 10/5/1%; higher is better)
-------------------Token Stats--------------------
Max input tokens:                        933
Max output tokens:                       806
Max total tokens:                        1570
--------------------GPU Stats---------------------
GPU Utilization (%):                     94.74
Peak GPU Memory Used (MiB):              37228.12
GPU Memory Available (MiB):              3216.25
==================================================
```

### Key metrics explained

- **Request throughput**: Number of complete requests processed per second
- **Input token throughput**: Number of input tokens processed per second
- **Output token throughput**: Number of tokens generated per second
- **TTFT**: Time to first token (TTFT), the time from request start to first
token generation
- **TPOT**: Time per output token (TPOT), the average time taken to generate
each output token
- **ITL**: Inter-token latency (ITL), the average time between consecutive token
or token-chunk generations
- **GPU utilization**: Percentage of time during which at least one GPU kernel
is being executed
- **Peak GPU memory used**: Peak memory usage during benchmark run

## Troubleshooting

### Memory issues

- Reduce batch size
- Check GPU memory availability: `nvidia-smi` or `rocm-smi`

### Permission issues

- Verify `HF_TOKEN` is set correctly
- Ensure model access on Hugging Face
