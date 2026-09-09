---
title: MAX accelerator library
sidebar_label: Overview
sidebar_position: 1
description: "Mojo APIs for hardware-accelerated programming: GPU primitives, compute kernels, tensor layouts, and graph extension points."
---

<section class='mojo-docs'>

The MAX accelerator library is the Mojo API for hardware-accelerated
programming. It covers GPU primitives, the compute kernels that MAX graphs
execute, the tensor layouts those kernels are written against, and the
extension points for adding your own operations.

## API summary

- Program a GPU directly with [`max.gpu`](/api/mojo/max/gpu/): thread and block
  indexing, memory spaces, synchronization, and host-side device management.
- Describe how tensors sit in memory with [`layout`](/api/mojo/layout/):
  layouts, tiling, and the tensor types kernels are written against.
- Call ready-made compute kernels from [`linalg`](/api/mojo/linalg/) for linear
  algebra such as matrix multiplication, [`nn`](/api/mojo/nn/) for
  neural-network operators such as attention and convolution,
  [`quantization`](/api/mojo/quantization/) for quantized weight encodings, and
  [`kv_cache`](/api/mojo/kv_cache/) for transformer key-value caches.
- Add your own operation to a MAX graph by writing it against
  [`extensibility`](/api/mojo/extensibility/).
- Spread work across GPUs and nodes with [`comm`](/api/mojo/comm/) for multi-GPU
  collectives and [`shmem`](/api/mojo/shmem/) for multi-node communication.
- Time and profile a kernel with
  [`max.benchmark`](/api/mojo/max/benchmark/) for timing harnesses and
  [`profiling_range`](/api/mojo/profiling_range/) for annotating profiler
  traces.

## Package organization

`max` is a top-level package with subpackages, so its members carry the `max`
prefix:

```mojo
from max.gpu.compute import mma
from max.benchmark import Bench
```

The rest are top-level packages alongside `max`:

```mojo
from linalg.matmul import matmul
from layout import TileTensor, row_major
```

The split tracks where the code lives today. More of these packages will move
under `max` in later releases.

</section>
