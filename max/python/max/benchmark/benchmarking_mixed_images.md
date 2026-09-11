# Mix generated images into any benchmark workload

This guide describes how to mix generated images into any `benchmark_serving.py`
workload, so you can benchmark vision-capable models without a dataset that
natively carries images.

## Overview

Every dataset that `benchmark_serving.py` supports (`sonnet`, `sharegpt`,
`instruct-coder`, `arxiv-summarization`, and so on) can now have images mixed
into a fraction of its requests or chat turns. Image mixing runs as a
post-sampling step, so it works the same way regardless of which dataset
you choose.

This is separate from the `random` dataset's own `--random-image-count` and
`--random-image-size` flags, which generate a fixed number and size of images
for every request. Use the general flags described here when you want images
mixed into a fraction of requests, with configurable size and count
distributions, on any dataset. The two mechanisms are mutually exclusive—if
you set both, the benchmark raises an error at startup.

## Quick start

Mix images into 50% of a `sonnet` benchmark's requests:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --backend modular \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --num-prompts 100 \
  --image-fraction 0.5 \
  --image-count 1 \
  --image-long-side 512 \
  --image-aspect-ratio 1.0
```

This selects 50% of requests to carry one generated 512x512 image each. The
remaining requests are unchanged.

## Flag reference

| Flag                   | Description                                                                                                                                                                   | Default |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `--image-fraction`     | Fraction (`0.0`-`1.0`) of requests (single-turn) or chat sessions (multi-turn) that get at least one image.                                                                   | `0.0`   |
| `--image-count`        | Distribution for the number of images on a selected request or turn. Accepts a constant or a distribution string such as `DU(1,4)`.                                           | `1`     |
| `--image-long-side`    | Distribution for each image's longer side, in pixels. Accepts a constant or a distribution string such as `U(224,1024)`.                                                      | `512`   |
| `--image-aspect-ratio` | Distribution for each image's width divided by height. `1.0` produces square images; values above `1.0` produce landscape images; values below `1.0` produce portrait images. | `1.0`   |
| `--image-turn`         | Which user turn(s) in a multi-turn chat session get images: `first`, `last`, or `every`. Ignored for single-turn requests.                                                    | `first` |

`--image-fraction` defaults to `0.0`, so none of these flags change behavior
unless you set it above zero.

## Multi-turn sessions

For multi-turn datasets (for example `sharegpt` with `--num-turns` set),
`--image-turn` controls which user turn in each selected session carries the
image:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --backend modular-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --num-prompts 50 \
  --image-fraction 1.0 \
  --image-turn last
```

This attaches an image to the last user turn of every session. Use `every` to
attach an image to every user turn instead of just one.

## Matching a real image-size distribution

`--image-count`, `--image-long-side`, and `--image-aspect-ratio` each accept
any distribution string supported by this codebase's distribution
grammar—not just a constant. Beyond the usual parametric shapes (`N(mean,std)`,
`U(lower,upper)`, `DU(lower,upper)`, `NB(n,p)`, `G(shape,scale)`,
`LN(mean,std)`, `Burr12(c,d,scale)`), there's `Cat(v1:w1, v2:w2, ...)`: an
explicit, weighted set of values. Use it when you've measured a real image-size
distribution (for example, from production traffic) and want to reproduce it
exactly, rather than approximate it with a bell curve or a uniform range.

For example, given measured production stats showing 70% of images at 1024px,
20% at 512px, and 10% at 2048px on the long side:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --dataset-name sonnet \
  --num-prompts 100 \
  --image-fraction 0.1 \
  --image-long-side "Cat(1024:0.7, 512:0.2, 2048:0.1)" \
  --image-aspect-ratio 1.0
```

The `:weight` suffix is optional per entry—omit it for a uniform choice among
the listed values, so `Cat(1,2,3)` picks each of `1`, `2`, and `3` with equal
probability. Weights don't need to sum to `1`; they're normalized
automatically, so `Cat(1024:7, 512:2, 2048:1)` behaves identically to the
example above.

`Cat(...)` isn't image-specific—it works with any distribution-accepting flag
in this codebase, including `--random-input-len`, `--random-output-len`, and
`--random-num-turns`.

## Checking your workload before running live

Use `--dry-run` to sample the workload and print distribution statistics
without sending any requests to a server. When a workload has images, the
output includes an image-count distribution table, plus a decoded
image-long-side-pixel table when the images are data URIs:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --dataset-name sonnet \
  --num-prompts 100 \
  --image-fraction 0.5 \
  --image-count 1 \
  --image-long-side 512 \
  --image-aspect-ratio 1.0 \
  --dry-run
```

```output
================ Workload Statistics ================
  Total requests:                100
  ...

  Image count (per request):
    min       max       mean      std       p5        p25       p50       p75       p95       p99
    0.00      1.00      0.48      0.50      0.00      0.00      0.00      1.00      1.00      1.00

  Image long side (px):
    min       max       mean      std       p5        p25       p50       p75       p95       p99
    512.00    512.00    512.00    0.00      512.00    512.00    512.00    512.00    512.00    512.00
=======================================================
```

The image tables are only printed when the workload actually has images—a run
without `--image-fraction` (or with `--image-fraction 0.0`) shows no image
section. Use this to confirm your flags produce the mix you expect before
spending time against a live server.

> [!NOTE]
> The reported image-count and pixel distributions describe the generated
> workload, not vision-token cost for any specific model. Real token
> accounting for images is architecture-specific (for example, patch-budget
> resizing differs between model families), so treat these tables as a
> workload-shaping tool, not an exact token forecast.
