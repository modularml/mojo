# Benchmark Configuration

This document describes the benchmark configuration classes in
`max.benchmark.benchmark_shared.config`, the optional YAML inheritance
mechanism (`depends_on`), and how sweep parameters work.

## Overview

Benchmark configurations are Pydantic models built on top of `ConfigFileModel`
(from `max.config`). They can be constructed directly in Python or loaded from
a YAML file via the `--config-file` CLI flag.

When loading from YAML, configs may inherit defaults from another YAML file
using the `depends_on` field. The merge logic is implemented in
`max/python/max/config/__init__.py` and resolves paths relative to the
inheriting file's directory. Today no benchmark configs in
`max/python/max/benchmark/configs/` use `depends_on`, but the mechanism
remains available for users who want to layer their own configs.

## Configuration Classes

| Class                    | File                                                  | Purpose                                                                                                                                     |
|--------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `BaseBenchmarkConfig`    | `max/python/max/benchmark/benchmark_shared/config.py` | Common parameters: model/tokenizer, dataset, workload size, control flags.                                                                  |
| `ServingBenchmarkConfig` | `max/python/max/benchmark/benchmark_shared/config.py` | Serving-specific parameters: backend, host/port, endpoint, concurrency, request rate, traffic control. Inherits from `BaseBenchmarkConfig`. |

### Import Examples

```python
from max.benchmark.benchmark_shared.config import (
    BaseBenchmarkConfig,
    ServingBenchmarkConfig,
)
```

## Sweep Parameters

`ServingBenchmarkConfig` has two fields that accept either a single value or
a comma-separated list of values, which `sweep_benchmark_serving.py` expands
into a matrix of runs:

- `max_concurrency` (`str | None`, parsed as `int | None` per entry)
- `request_rate` (`str`, parsed as `float` per entry)

The expansion happens in
`max.benchmark.benchmark_serving.main_with_parsed_args`:

```python
concurrency_range = parse_comma_separated(args.max_concurrency, int_or_none)
request_rate_range = parse_comma_separated(args.request_rate, float)
```

`parse_comma_separated` and `int_or_none` live in
`max/python/max/benchmark/benchmark_shared/utils.py`. Special values:

- `None` (case insensitive) parses to `None`.
- `inf` (case insensitive) is accepted by `float` for `request_rate`.
- Empty entries between commas parse to `None`.

To add another sweep dimension, add a new `str` field on the config (with a
description that documents the comma-separated form) and add a matching
`parse_comma_separated` call + sweep loop in `benchmark_serving.py`.

## Adding a New YAML Config

To add a new benchmark YAML config under
`max/python/max/benchmark/configs/`:

1. Create the file (e.g., `my_benchmark_config.yaml`).
2. Put benchmark fields under a top-level `benchmark_config:` section
   (this matches `BaseBenchmarkConfig`'s `section_name` default).
3. Optionally set `depends_on: "<parent_config>.yaml"` to inherit defaults
   from another YAML config in the same directory.
4. Bazel automatically picks the file up via `srcs = glob(["configs/*.yaml"])`
   in `max/python/max/benchmark/BUILD.bazel`; no BUILD edit is required.

Example:

```yaml
# my_benchmark_config.yaml
name: "My Custom Benchmark Configuration"
description: "Configuration for my custom benchmark"
version: "0.0.1"

benchmark_config:
  model: "modularai/Llama-3.1-8B-Instruct-GGUF"
  num_prompts: 100

  # Sweepable fields accept either a single value or a comma-separated list.
  max_concurrency: "1,2,4,8"
  request_rate: "1.0,2.0,4.0,inf"
```

## Passing arbitrary request fields with `extra_body`

`extra_body` injects arbitrary top-level fields onto every text-generation
request payload — for fields that lack a dedicated flag (`stop`,
`chat_template_kwargs`, vendor extensions, and so on). In a config file it is a
native mapping under the `benchmark_config` block:

```yaml
benchmark_config:
  model: "modularai/Llama-3.1-8B-Instruct-GGUF"
  extra_body:
    stop: ["}"]
    chat_template_kwargs:
      reasoning_effort: "low"
```

On the CLI, `--extra-body` also accepts an inline JSON object or a path to a
YAML/JSON file. Fields are merged after the dedicated flags (last-writer-wins);
see the `extra_body` field description for the full merge semantics.
