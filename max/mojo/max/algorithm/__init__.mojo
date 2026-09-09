# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""High performance data operations: parallelization, reduction, memory.

The `algorithm` package provides high-performance primitives for data-parallel
operations that run on both CPUs and accelerators. It includes tools for
parallelization (distributing work across multiple cores), elementwise and
stencil traversal, and reductions (aggregating values). These building blocks
enable efficient computational kernels without manual SIMD intrinsics or thread
management. The `std.algorithm` package retains the non-accelerator algorithms,
such as vectorization and tiling.

Use this package for large datasets, numerical algorithms, or compute-intensive
code. For element-wise operations on small data, standard loops may be simpler.
"""

from .functional import (
    elementwise,
    parallelize,
    parallelize_over_rows,
    stencil,
    stencil_gpu,
    sync_parallelize,
)
from .memory import parallel_memcpy, unsafe_parallel_memcpy
from .reduction import (
    cumsum,
    map_reduce,
    max,
    mean,
    min,
    product,
    reduce,
    sum,
    variance,
)
