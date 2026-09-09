# Benchmark metrics invariants

This document describes the invariants that the benchmark metric types in
`max.benchmark.benchmark_shared.metrics` and
`max.benchmark.benchmark_shared.percentile_metrics` are expected to hold. It
exists so that anyone editing these classes, adding a new metric, or writing a
consumer that deserializes benchmark results can rely on a single, explicit
contract instead of rediscovering it from the producer code.

The most important takeaway: a percentile or throughput metric is `None` when
the iteration had no samples to derive it from. It is never a placeholder object
full of `NaN`. See [When a metric is `None`](#when-a-metric-is-none) for the
exact conditions.

## Where these types live

The following table lists the metric types and the file that defines each.

| Type                        | File                    | Purpose                                                    |
|-----------------------------|-------------------------|------------------------------------------------------------|
| `PercentileMetrics`         | `percentile_metrics.py` | Container for `mean`/`std`/`p50`/`p90`/`p95`/`p99`.        |
| `ConfidenceInfo`            | `percentile_metrics.py` | Optional 95% confidence-interval metadata for a metric.    |
| `StandardPercentileMetrics` | `metrics.py`            | Builds `PercentileMetrics` with ascending percentiles.     |
| `ThroughputMetrics`         | `metrics.py`            | Builds `PercentileMetrics` with reversed percentiles.      |
| `RatePercentileMetrics`     | `metrics.py`            | Bounded ratio in `[0, 1]` (or `[0, 100]` as a percentage). |
| `_CompletedRunBase`         | `metrics.py`            | Fields common to any completed iteration.                  |
| `TextGenAggregates`         | `metrics.py`            | Text-generation iteration aggregates.                      |
| `PixelGenAggregates`        | `metrics.py`            | Pixel-generation iteration aggregates.                     |
| `BenchmarkResult`           | `metrics.py`            | One iteration: text or pixel aggregates plus device stats. |

The engine benchmark path (`utils/benchmarking/engine/results.py`) reuses
`PercentileMetrics` directly through `StandardPercentileMetrics`, so the
`PercentileMetrics` invariants below apply there too.

## `PercentileMetrics` invariants

- The six numeric fields (`mean`, `std`, `p50`, `p90`, `p95`, `p99`) are always
  present together. They are computed in one pass and set as a group, so a
  `PercentileMetrics` object never carries a subset of them. If you have the
  object at all, you have all six.
- `unit` is optional metadata (`str` or `None`) and does not affect the numeric
  fields.
- `confidence_info` is `None` when there are fewer than two data points (a
  confidence interval is undefined for a single sample), or when the mean is not
  finite and positive.
- Percentile ordering depends on the builder:
  - `StandardPercentileMetrics` uses ascending percentiles, where a higher value
    is worse. Expect `p50 <= p90 <= p95 <= p99` (subject to floating-point
    noise).
  - `ThroughputMetrics` reverses the percentiles, because a smaller throughput
    is worse. Its `p90`, `p95`, and `p99` hold the bottom 10%, 5%, and 1%, so
    numerically `p90 >= p95 >= p99`.
- `RatePercentileMetrics` is a bounded ratio: values stay within `[0, 1]` when
  `as_percent=False` and within `[0, 100]` when `as_percent=True`. Its
  validation enforces the corresponding upper bound.
- The `p50` field carries the median. When flattening to the legacy column
  layout, `to_flat_dict()` emits the key `median_<name>` (not `p50_<name>`) to
  preserve the historical column names that the legacy upload path expects.

## Iteration aggregate invariants

`BenchmarkResult` holds one iteration, and its two workload-aggregate fields
follow a strict contract:

- At most one of `text_data` and `pixel_data` is ever set, and a set aggregate
  must match `task_type`: `text_data` requires `task_type == "text"` and
  `pixel_data` requires `task_type == "pixel"`. A `model_validator` on
  `BenchmarkResult` enforces this.
- The serving producer (`build_text_generation_result` and
  `build_pixel_generation_result`) always builds the aggregate matching the
  task, so a successful iteration has exactly one set. This holds even when
  every request failed: an all-failed text iteration still carries a `text_data`
  with `completed == 0` and every percentile or throughput field `None` (see
  [When a metric is `None`](#when-a-metric-is-none)). It does not collapse to a
  both-`None` result.
- Both fields `None` is a state the type permits (it represents an iteration
  with no workload aggregates), and callers may construct it directly. The
  serving sweep does not emit such rows, and dry runs produce no iterations at
  all.

`_CompletedRunBase` (the shared base of both aggregate types) always populates
its scalar counters: `duration`, `completed`, `failures`, and
`request_throughput`. `TextGenAggregates` additionally always populates its
token counters, such as `total_input`, `total_output`, and the `max_*` fields.
These scalars describe the run itself and are meaningful even when no
per-request latency samples exist.

The percentile and throughput fields on the aggregates are optional. A populated
metric implies at least one measured sample contributed to it. A `None` metric
means there were no samples for it this iteration (see below). The
always-present scalar counters, not the metric fields, are the reliable signal
for "did this iteration run."

## When a metric is `None`

Each metric is derived from a list of per-request samples collected over the
measured requests (the successful requests that remain after the skip windows).
A metric is `None` exactly when its sample list is empty. These are the concrete
cases.

### No measured samples: every metric is `None`

`latency_ms` and `ttft_ms` receive one sample per measured request, so they (and
every other metric) are `None` when there are no measured requests:

- Every request failed, so `completed` is 0.
- Every successful request was excluded by `skip_first_n_requests` and
  `skip_last_n_requests` together. The producer emits a warning in this case,
  and `excluded_successful` records how many successes the trim dropped. When
  `completed == 0` with `excluded_successful > 0`, `validate_metrics()` reports
  a single insufficient-data error instead of one error per degenerate metric.
  The server completed requests -- the run just yielded too few samples to
  measure. This distinguishes "not enough data" from "the server did nothing".

### No decode data: only the decode-phase metrics are `None`

When requests succeed but produce at most one output token, the decode-phase
metrics have no samples while `ttft_ms` and `latency_ms` stay populated:

- `tpot_ms` collects a sample only for requests with more than one output token.
- `itl_ms` is built from inter-token latencies, which do not exist for a
  single-token response.
- `step_tpot_ms` is populated only when the backend provides chunk-level text
  for re-tokenization.

Prefill-only and single-output-token workloads therefore leave `tpot_ms`,
`itl_ms`, and `step_tpot_ms` as `None`. The `validate_metrics()` check already
skips decode-phase metrics when `max_output <= 1`.

### Timing- or endpoint-specific `None` values

- `input_throughput` is `None` when no measured request reported a nonzero time
  to first token. A non-streaming endpoint that does not report a per-request
  TTFT produces this.
- `output_throughput` is `None` when no measured request had a positive decode
  window (latency greater than TTFT), which happens with degenerate or
  single-token timing.

## Serialization invariants

These invariants explain why the "no samples" case must be `None` rather than a
`NaN`-filled object:

- `model_dump_json()` renders non-finite floats (`NaN`, `inf`, `-inf`) as JSON
  `null` by default.
- BigQuery applies the same rewrite when ingesting a JSON value: `NaN` and
  `Infinity` become `null`.

So a metric built from `[float("nan")]` (the old "no data" placeholder)
serializes to an object whose six numeric fields are all `null`. Strict
consumers that deserialize result rows type these fields as required, non-null
floats, so they reject such an object field by field and report it as dropped
data, even though the iteration was produced correctly. Emitting `None` for a
metric with no samples avoids this: the field serializes to a single `null`, and
consumers whose schemas mark the metric optional accept it and render an empty
value.

For the same reason, an unbounded `request_rate` is serialized as `None` at the
results-publication boundary rather than as `inf`.

## Adding or changing a metric

Keep these rules in mind when you touch the metric types:

- If a metric can legitimately have zero samples, type its field as optional and
  emit `None` for the empty case. Do not build it from `[float("nan")]`.
- Guard every consumer of an optional metric: `to_result_dict()`,
  `validate_metrics()`, `confidence_warnings()`, the human-readable summary
  printer, and the legacy CSV or upload path. The legacy path may substitute a
  local `NaN` to preserve its historical column values.
- Keep the strict consumer schemas in sync with `PercentileMetrics`. A schema
  that mirrors these types must list the same fields, and any field that the
  producer can leave `None` must be optional in the mirror.
