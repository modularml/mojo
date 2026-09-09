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
"""Kernel-geometry-derived scheduling constants.

The pipeline framework consumes four kernel-derived counts that live in
two different configs today:

  - `PipelineConfig.vm_per_load_a` / `vm_per_load_b`: vmcnt entries
    consumed by one channel-A or channel-B prefetch (drives `wait_vm`
    derivation).
  - `ScheduleConfig.lgkm_per_load_a` / `lgkm_per_load_b`: lgkmcnt
    entries consumed by one channel-A or channel-B fragment load
    (drives `wait_lgkm` derivation).

Both are kernel-geometry properties (they fall out of BM/BN/BK,
warp count, MMA shape, dtype size), but every kernel computes them
from its own warp-grid + MMA-pattern assumptions. `KernelGeometry`
captures the result once and exposes the derived counts as fields;
each kernel layout (4-wave, ping-pong, ...) provides its own factory
that does the layout-specific math and returns a `KernelGeometry`.

Existing kernels that pass the four values directly continue to work;
this struct is purely additive consolidation of the API surface.
"""


@fieldwise_init
struct KernelGeometry(Copyable, Movable):
    """Bundles kernel-shape inputs and the derived scheduling counts."""

    var BM: Int
    """Block shape M (rows per workgroup tile)."""
    var BN: Int
    """Block shape N (columns per workgroup tile)."""
    var BK: Int
    """Block shape K (reduction dimension per workgroup tile)."""
    var MMA_M: Int
    """MFMA op shape M."""
    var MMA_N: Int
    """MFMA op shape N."""
    var MMA_K: Int
    """MFMA op shape K."""
    var elem_bytes: Int
    """Element size in bytes (1 for FP8, 2 for BF16/FP16, 4 for FP32)."""
    var simd_width: Int
    """SIMD load width in elements (typically `simd_width_of[in_type]()`)."""
    var is_fp8: Bool
    """True iff `elem_bytes == 1` (FP8 dtypes)."""
    var vm_per_load_a: Int
    """Number of `vmcnt` entries consumed per channel-A prefetch."""
    var vm_per_load_b: Int
    """Number of `vmcnt` entries consumed per channel-B prefetch."""
    var lgkm_per_load_a: Int
    """Number of `lgkmcnt` entries consumed per channel-A fragment load."""
    var lgkm_per_load_b: Int
    """Number of `lgkmcnt` entries consumed per channel-B fragment load."""
