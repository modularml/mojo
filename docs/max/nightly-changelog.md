---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

## MAX framework

- Hardened decoding of client-supplied images. `Image.open` is now restricted
  to an explicit format allowlist (PNG, JPEG, WEBP, GIF, BMP, PPM, TIFF, TGA,
  and AVIF where the platform provides it), shrinking the native-decoder attack
  surface and keeping image bytes away from decoders such as EPS/Ghostscript.
- Media memory is now governed by a single knob, `MAX_SERVE_MAX_MEDIA_BYTES`
  (default 100 MiB). The fetched size of all resolved media (`http(s)://`,
  `data:`, and `file://` images and videos) for a request is bounded in
  aggregate by it, rather than each item being capped independently. No single
  image may *decode* to more than it either. That decoded size is estimated
  from the image header and rejected before the pixel buffer is allocated,
  which is what catches a decompression bomb (a small on-the-wire image that
  expands to hundreds of MB). This replaces both the former per-item
  `MAX_SERVE_MAX_BYTES` server cap and a separate decoded-pixel limit. It is
  deliberately separate from `MAX_SERVE_MAX_REQUEST_BYTES`, which bounds only
  the request body. A small body can name URLs the server then fetches, so
  raising one limit should not silently widen the other.
- Added dataset-agnostic image mixing to `max benchmark` (`--image-fraction`,
  `--image-count`, `--image-long-side`, `--image-aspect-ratio`, `--image-turn`),
  so any dataset (`sonnet`, `sharegpt`, `instruct-coder`, and so on) can have
  generated images mixed into a fraction of its requests or chat turns, not just
  the `random` dataset's existing fixed
  `--random-image-count`/`--random-image-size`. `--dry-run` reports an
  image-count distribution table (and a decoded image-long-side-pixel table)
  when a workload has images. See the
  [image mixing quick-start guide](https://github.com/modular/modular/blob/main/max/python/max/benchmark/benchmarking_mixed_images.md).
- Added a `Cat(v1:w1, v2:w2, ...)` categorical distribution for every
  `max benchmark` config field that accepts a distribution string (for
  example `--image-long-side`, `--image-count`, `--random-input-len`), so an
  explicit, empirically-measured distribution (such as image sizes measured
  from real production traffic) can be reproduced exactly instead of
  approximated with a parametric shape like `N`/`U`/`LN`. The `:weight`
  suffix is optional per entry (uniform when omitted), and weights don't need
  to sum to 1.

### Inference server

### Server metrics

### `max` CLI

### Python API

### C API

## Kernels and GPU programming

## Breaking changes

- `KVCacheMetrics` drops `nixl_read_blocks_local`, `nixl_read_blocks_remote`,
  and the `remote_read_ratio` property computed over them. No code path ever
  populated either field, so the ratio returned `0.0` for every caller. Six
  populated counters are added in their place: `dkv_peer_attaches`,
  `dkv_peer_attach_failures`, `dkv_peers_dropped`, `dkv_peer_loads`,
  `dkv_peer_load_failures`, and `dkv_hints_rejected`. They are zero unless the
  external KV-cache connector is in use.

## Fixes

## Mojo language
