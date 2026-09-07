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
"""Multi-GPU allreduce implementation for efficient tensor reduction across GPUs.

This module provides an optimized implementation of allreduce operations across multiple GPUs,
supporting both peer-to-peer (P2P) and non-P2P communication patterns. The implementation
automatically selects between two approaches based on hardware capabilities:

1. P2P-based implementation (when P2P access is available):
   - Uses direct GPU-to-GPU memory access for better performance
   - Implements both single-stage and two-stage algorithms:
     - Single-stage for latency-bound transfers (small tensors)
     - Two-stage (reduce-scatter + all-gather) for bandwidth-bound transfers (large tensors)
   - Optimized for NVLink bandwidth utilization
   - Uses vectorized memory access and higher precision accumulation

2. Non-P2P fallback implementation:
   - Copies data through host memory when direct GPU access isn't possible
   - Simple but functional approach for systems without P2P support

The implementation is tuned for common GPU architectures (A100, H100) and includes
parameters that can be adjusted for different hardware configurations.

## Per-Device Architecture

The allreduce operation follows a per-device execution model:

1. **Single-Device Instances**: Each GPU runs its own instance of the allreduce
   operation.

2. **Parallel Execution**: The Python/Graph API layer is responsible for:
   - Creating one allreduce op instance per participating GPU.
   - Ensuring all instances execute in parallel.
   - Ensuring correctness by staging mo.fence.

3. **Device Affinity**: Each allreduce instance:
   - Executes on its assigned GPU (specified via device context).
   - Reads from all GPUs' input buffers (requires P2P access).
   - Writes only to its own output buffer.
   - Uses the same synchronization signals as other instances.

4. **Requirements**:
   - Peer-to-peer access must be enabled between all participating GPUs.
   - All instances must launch before any can complete (for synchronization).
   - The device context determines which GPU executes each instance.

Limitations:
- Maximum of 8 GPUs supported.
- Multimem mode still requires the element count to be a multiple of SIMD width.
- All input/output buffers must have identical shapes.

Non-multimem 1-stage P2P and naive epilogue accept arbitrary ``N``: when
``N`` is a multiple of device SIMD width, the 1-stage kernel uses the same
vectorized ``_load_reduce`` grid loop as before; otherwise it runs that loop on
the SIMD-aligned prefix and finishes the last ``< simd_width`` elements with a
grid-strided scalar reduce-store. The naive epilogue kernel uses the same
SIMD-prefix + scalar-tail pattern for ``accum → out``.

## Visual Overview

1) 1-Stage P2P (latency-bound)

   Each GPU r reads its portion from every peer buffer directly (via P2P),
   accumulates, then writes to its result using the epilogue:

       GPU r (result_r)
       src_tensors[0] ─┐
       src_tensors[1] ─┼──► Σ (high-precision accum) ──► output_lambda ──► result_r
       ...         ─┘

   Notes:
   - Non-multimem: SIMD-vector ``_load_reduce`` on the aligned prefix; optional
     scalar tail when ``N`` is not a multiple of SIMD width. Multimem: unchanged
     full-vector loads (``N`` must be SIMD-aligned).
   - Good for small/latency-bound tensors.

2) 2-Stage P2P (bandwidth-bound)

   Stage 1 (reduce-scatter): Each GPU r reduces its assigned partition and writes
   into its own signal payload (the bytes after the Signal header).

       src_tensors[*]  ──►  reduce(partition r)  ──►  rank_sigs[r].payload  (per-GPU)

   Stage 2 (all-gather): Each GPU r gathers all partitions from peers' payloads
   and writes them to its result using the epilogue.

       [payload_0], [payload_1], ..., [payload_{ngpus-1}]  ──►  result_r (via output_lambda)

For the naive allreduce (no P2P) per-device flow and staging details, see the
`_allreduce_naive_single` docstring in this file.
"""

from std.atomic import Atomic, Ordering, fence
from std.collections import Array
from std.math import ceildiv, clamp
from std.sys import align_of, is_amd_gpu, is_nvidia_gpu, simd_width_of, size_of

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import TensorLayout
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    global_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target

from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from std.collections.optional import Optional

from .reducescatter import (
    ReduceScatterConfig,
    _reduce_scatter_impl,
    _load_reduce,
    _target_address_space,
)
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    circular_add,
    is_p2p_enabled,
)
from .lamport import (
    Lamport,
    LamportGeneration,
    has_neg_zero,
    remove_neg_zero,
    set_neg_zero,
)
from .device_query import (
    dispatch_select_comm_config,
    CommTuningConfig,
    KB,
    MB,
    GB,
)
from internal_utils import Table

comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int
](Coord, SIMD[dtype, length=width]) capturing -> None

# Tuning table to get num_blocks for allreduce.


@fieldwise_init
struct AllReduceAlgorithm(TrivialRegisterPassable, Writable):
    """Selects which P2P allreduce kernel `_allreduce_p2p` launches.

    Replaces the former pair of `use_2stage` / `use_lamport` booleans with a
    single mutually-exclusive selector (the booleans could encode nonsensical
    combinations such as "2-stage and Lamport").
    """

    var _value: Int

    comptime ONE_STAGE = Self(0)
    """Latency-bound: every block reads all peers and reduces directly."""
    comptime TWO_STAGE = Self(1)
    """Bandwidth-bound: reduce-scatter into peer payloads, then all-gather."""
    comptime LAMPORT = Self(2)
    """Barrier-free negative-zero sentinel path (small messages only)."""

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    def write_to(self, mut writer: Some[Writer]):
        """Writes the human-readable algorithm name.

        Args:
            writer: The writer to write to.
        """
        if self == Self.ONE_STAGE:
            writer.write("1_stage")
        elif self == Self.TWO_STAGE:
            writer.write("2_stage")
        else:
            writer.write("lamport")


@fieldwise_init
struct AllReduceTuningConfig(CommTuningConfig, TrivialRegisterPassable):
    """
    Parameters:
        ngpus: Number of GPUs for running allreduce.
        num_bytes: Total number of input bytes supported by the config.
        sm_version: SM version (as string).
        num_blocks: Number of thread blocks for running allreduce.
    """

    var ngpus: Int
    var num_bytes: Int
    var sm_version: StaticString
    var num_blocks: Int
    var algorithm: AllReduceAlgorithm

    def get_num_blocks(self) -> Int:
        return self.num_blocks

    def get_num_bytes(self) -> Int:
        return self.num_bytes

    def get_sm_version(self) -> StaticString:
        return self.sm_version

    def get_ngpus(self) -> Int:
        return self.ngpus

    def get_algorithm(self) -> AllReduceAlgorithm:
        return self.algorithm

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tuning config as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            self.ngpus,
            self.num_bytes,
            self.sm_version,
            self.num_blocks,
            self.algorithm,
        )


# Arch-specific defaults use ngpus=-1, num_bytes=-1 with the arch's sm_version.
# The global default (sm_version="default") is the ultimate fallback for
# unknown architectures -- dispatch_select_comm_config prefers arch-specific
# defaults when available.
comptime allreduce_tuning_table = Table(
    [
        # default for sm90 (encoded with ngpus=-1, num_bytes=-1)
        AllReduceTuningConfig(
            ngpus=-1,
            num_bytes=-1,
            sm_version="sm_90a",
            num_blocks=216,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # 2xH100: 1-stage wins across all measured sizes (16KB to 128MB);
        # ratio 2/1 = 1.02 at 128MB, narrowing but no crossover.
        AllReduceTuningConfig(
            ngpus=2,
            num_bytes=(2 * GB),
            sm_version="sm_90a",
            num_blocks=216,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # 4xH100 / 8xH100 thresholds: 1-stage wins below 1MB, 2-stage wins
        # cleanly from 1MB upward (ratio 0.83 at 1MB / ngpus=4, 0.74 at
        # 1MB / ngpus=8).
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(512 * KB),  # 512KB: largest size where 1-stage wins.
            sm_version="sm_90a",
            num_blocks=216,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(128 * MB),
            sm_version="sm_90a",
            num_blocks=232,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(2 * GB),
            sm_version="sm_90a",
            num_blocks=216,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(512 * KB),
            sm_version="sm_90a",
            num_blocks=216,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(2 * GB),
            sm_version="sm_90a",
            num_blocks=216,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        # default for sm100 (encoded with ngpus=-1, num_bytes=-1).
        AllReduceTuningConfig(
            ngpus=-1,
            num_bytes=-1,
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # B200 Lamport crossover (1 KiB - 8 MiB forced sweep): Lamport beats
        # 1-stage by ~1.1-1.68x across the small-message range, reaching
        # break-even at ~1 MiB (1-stage pulls ahead above it).
        AllReduceTuningConfig(
            ngpus=2,
            num_bytes=(1 * MB),
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.LAMPORT,
        ),
        # 2xB200: 1-stage wins across all measured sizes (16KB to 256MB).
        # The 2-stage curve approaches but does not cross 1-stage in the
        # measured range (ratio 2/1 = 1.01 at 256MB).
        AllReduceTuningConfig(
            ngpus=2,
            num_bytes=(2 * GB),
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(1 * MB),
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.LAMPORT,
        ),
        # 8xB200 / 4xB200 thresholds: 1-stage wins for latency-bound sizes,
        # 2-stage wins where bandwidth dominates. The crossover is at a
        # different size for each ngpus, so the boundary entries differ.
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(4 * MB),  # 4MB: 1-stage wins by 9% at 4MB; 2-stage
            # wins by 25% at 8MB.
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(2 * GB),
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(1 * MB),
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.LAMPORT,
        ),
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(2 * MB),  # 2MB: 1-stage and 2-stage are within 3%
            # at 2MB; 2-stage wins by 37% at 4MB. 2MB stays in the 1-stage
            # bucket as a noise margin for sub-2MB workloads.
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(2 * GB),
            sm_version="sm_100a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        # default for sm103 (B300, encoded with ngpus=-1, num_bytes=-1)
        AllReduceTuningConfig(
            ngpus=-1,
            num_bytes=-1,
            sm_version="sm_103a",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # default for CDNA3 (MI300X, encoded with ngpus=-1, num_bytes=-1)
        AllReduceTuningConfig(
            ngpus=-1,
            num_bytes=-1,
            sm_version="CDNA3",
            num_blocks=32,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # default for CDNA4 (MI355X, encoded with ngpus=-1, num_bytes=-1)
        AllReduceTuningConfig(
            ngpus=-1,
            num_bytes=-1,
            sm_version="CDNA4",
            num_blocks=64,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # 2xMI355: 1-stage and 2-stage are within noise across all measured
        # sizes (busbw ~64 GB/s for both at 128MB -- XGMI bandwidth bound,
        # not algorithm bound). Keep 1-stage to match arch convention.
        AllReduceTuningConfig(
            ngpus=2,
            num_bytes=(2 * GB),
            sm_version="CDNA4",
            num_blocks=64,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # 4xMI355 / 8xMI355 thresholds: 256KB boundary. At 256KB, 1-stage
        # ties or wins narrowly; at 512KB 2-stage starts winning (~8% on
        # ngpus=4, tie on ngpus=8); from 1MB upward 2-stage wins decisively.
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(256 * KB),
            sm_version="CDNA4",
            num_blocks=64,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=4,
            num_bytes=(2 * GB),
            sm_version="CDNA4",
            num_blocks=64,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(256 * KB),
            sm_version="CDNA4",
            num_blocks=64,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
        # Recovered from pre-refactor tuning table: 44 blocks was tuned for
        # ngpus=8 / large sizes. Under the old hard-coded threshold ngpus=8 /
        # >256KB used 2-stage, so this tuning belongs to the 2-stage entry.
        AllReduceTuningConfig(
            ngpus=8,
            num_bytes=(2 * GB),
            sm_version="CDNA4",
            num_blocks=44,
            algorithm=AllReduceAlgorithm.TWO_STAGE,
        ),
        # global default for unknown architectures
        AllReduceTuningConfig(
            ngpus=-1,
            num_bytes=-1,
            sm_version="default",
            num_blocks=512,
            algorithm=AllReduceAlgorithm.ONE_STAGE,
        ),
    ],
    "allreduce_table",
)


@__name(t"naive_reduce_{dtype}")
def _naive_reduce_kernel[
    dtype: DType
](
    dst_buf: MutPointer[Scalar[dtype], MutAnyOrigin],
    src_buf: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int32,
):
    """
    A simple reduction kernel that adds source buffer values to destination buffer.

    Parameters:
        dtype: DType - The data type of the values being reduced.

    Args:
        dst_buf: Destination buffer to accumulate results.
        src_buf: Source buffer containing values to add.
        num_elements: Number of elements to process.

    Each thread handles multiple elements with striding for coalesced memory access.
    """
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x
    var _num_elements = Int(num_elements)

    # Each thread handles multiple elements with striding
    for i in range(tid, _num_elements, stride):
        dst_buf[i] += src_buf[i]


@__name(t"naive_reduce_with_lambda_{dtype}")
def _naive_reduce_kernel_with_lambda[
    dtype: DType,
    out_layout: TensorLayout,
    *,
    output_lambda: elementwise_epilogue_type,
](
    dst_buf: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_buf: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int32,
):
    """Apply ``output_lambda`` from ``src_buf`` into ``dst_buf`` (naive epilogue).

    Uses device SIMD width loads on the aligned prefix (same pattern as the
    pre-ragged vector epilogue), then grid-strided scalar loads for any tail when
    ``num_elements`` is not a multiple of SIMD width.
    """
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime simd_align = align_of[SIMD[dtype, simd_width]]()
    comptime scalar_align = align_of[SIMD[dtype, 1]]()
    var global_tid = global_idx.x
    var total_threads = grid_dim.x * Int(block_dim.x)
    var _num_elements = Int(num_elements)
    var num_simd_vectors = _num_elements // simd_width
    var simd_prefix_elems = num_simd_vectors * simd_width

    if num_simd_vectors > 0:
        for idx in range(global_tid, num_simd_vectors, total_threads):
            var elem_idx = idx * simd_width
            output_lambda[width=simd_width, alignment=simd_align](
                dst_buf.layout.idx2crd(elem_idx),
                src_buf.load[width=simd_width, alignment=simd_align](elem_idx),
            )

    if simd_prefix_elems < _num_elements:
        for elem_idx in range(
            simd_prefix_elems + global_tid, _num_elements, total_threads
        ):
            output_lambda[width=1, alignment=scalar_align](
                dst_buf.layout.idx2crd(elem_idx),
                src_buf.load[width=1, alignment=scalar_align](elem_idx),
            )


@always_inline
def _allreduce_naive_single[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: elementwise_epilogue_type,
    num_tensors: Int = ngpus,
](
    list_of_in_tensors: Array[
        TileTensor[dtype, in_layout, in_origin], num_tensors
    ],
    out_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    max_num_blocks: Int,
    ctx: DeviceContext,
) raises:
    """Naive per-device allreduce using a local temporary staging buffer.

    Overview
    - One op instance runs per GPU ("device r").
    - Each instance builds its local result by summing all inputs into a local
      accumulation buffer, then writes to its own output.
    - To stage remote inputs for accumulation (no P2P), it allocates a temporary
      buffer on the current device.

    Memory layout per device (r):

        tmp_r  (device-local buffer, length = N elements)

    Parameters:
        dtype: The data type of tensor elements.
        ngpus: Number of GPUs participating in allreduce.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        output_lambda: An elementwise output lambda function.
        num_tensors: Number of buffers to process (defaults to ngpus).

    Per-device flow (device r):

        in_r  ───────►  accumulate into A_r
        for each i != r:
          in_i  ──copy──►  S_r  ──accumulate──►  A_r
        A_r  ──output_lambda──► out_r

    ASCII for a 3-GPU example (naive path, no P2P):

        GPU0:  in0  →  A0 += in0
               in1  →  tmp0 → A0 += tmp0
               in2  →  tmp0 → A0 += tmp0
               A0   →  out0 (via output_lambda)

        GPU1:  in1  →  A1 += in1
               in0  →  tmp1 → A1 += tmp1
               in2  →  tmp1 → A1 += tmp1
               A1   →  out1 (via output_lambda)

        GPU2:  in2  →  A2 += in2
               in0  →  tmp2 → A2 += tmp2
               in1  →  tmp2 → A2 += tmp2
               A2   →  out2 (via output_lambda)

    Requirements
    - Inputs across GPUs must be identical shape and dtype.
    - Each op instance only writes to its own temporary buffer and its own
      output buffer (`out_r`).
    """
    comptime BLOCK_SIZE = 256
    var num_elements = list_of_in_tensors[0].num_elements()

    # Wrap ALL input buffers as DeviceBuffer with their respective device contexts.
    # rebind to MutAnyOrigin is safe: DeviceBuffer only reads via DMA copy.
    var dev_inputs = List[DeviceBuffer[dtype]](capacity=ngpus)
    for i in range(ngpus):
        var rctx = DeviceContext(device_id=i)
        dev_inputs.append(
            DeviceBuffer[dtype](
                rctx,
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    list_of_in_tensors[i]._storage
                ),
                num_elements,
                owning=False,
            )
        )

    # Accumulation buffer on this device.
    var accum = ctx.enqueue_create_buffer[dtype](num_elements)
    ctx.enqueue_memset(accum, 0)

    # Resolve this device's rank and allocate a temp staging buffer.
    var my_rank: Int = Int(ctx.id())
    var scratch = ctx.enqueue_create_buffer[dtype](num_elements)

    # Grid configuration for naive kernels.
    var grid_size = clamp(ceildiv(num_elements, BLOCK_SIZE), 1, max_num_blocks)
    comptime simd_width_epi = simd_width_of[dtype, target=get_gpu_target()]()
    var num_simd_vecs_epi = num_elements // simd_width_epi
    var tail_elems_epi = num_elements - num_simd_vecs_epi * simd_width_epi
    var grid_simd_epi = ceildiv(num_simd_vecs_epi, BLOCK_SIZE)
    var grid_tail_epi = ceildiv(tail_elems_epi, BLOCK_SIZE)
    var grid_epilogue = clamp(
        max(grid_simd_epi, grid_tail_epi), 1, max_num_blocks
    )

    # Reduce local buffer first.
    ctx.enqueue_function[_naive_reduce_kernel[dtype]](
        accum,
        dev_inputs[my_rank],
        Int32(num_elements),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )

    # Reduce contributions from peers via scratch.
    for i in range(ngpus):
        if i == my_rank:
            continue

        # Copy remote input into device-local scratch, then accumulate.
        ctx.enqueue_copy(scratch, dev_inputs[i])
        ctx.enqueue_function[_naive_reduce_kernel[dtype]](
            accum,
            scratch,
            Int32(num_elements),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

    # Apply elementwise epilogue to write into the output buffer.
    comptime naive_reduce_with_lambda_kernel = _naive_reduce_kernel_with_lambda[
        dtype,
        out_layout,
        output_lambda=output_lambda,
    ]
    ctx.enqueue_function[naive_reduce_with_lambda_kernel](
        rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
        accum,
        Int32(num_elements),
        grid_dim=grid_epilogue,
        block_dim=BLOCK_SIZE,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allreduce_2stage_{dtype}_{use_multimem}")
def _allreduce_2stage_kernel[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
    domain_id: Int = 0,
](
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_tensors: Array[
        TileTensor[dtype, in_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int32,
    my_rank: Int32,
):
    """2-stage allreduce algorithm for bandwidth-bound transfers.

    This kernel implements a reduce-scatter + all-gather algorithm that is
    bandwidth optimal.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        use_multimem: If True, use multi-memory space buffers for input.
        domain_id: Barrier counter bank to use (0 for full-world; a distinct
            nonzero value for grouped collectives). See `_multi_gpu_barrier`.

    Args:
        result: Output buffer for reduced values.
        src_tensors: Input buffers from all GPUs.
        rank_sigs: Signal pointers for synchronization.
            IMPORTANT: the Signal pointers have trailing buffers for
            communication, which must be at least `ngpus * size_of(payload)`.
            | -- size_of(Signal) -- | ------ a few MB ----- |
        num_elements: Number of elements to reduce.
        my_rank: Current GPU rank.
    """
    var _my_rank = Int(my_rank)
    var _num_elements = Int(num_elements)
    var my_sig = rank_sigs[_my_rank]

    # --- Thread Indexing ---
    var global_tid = global_idx.x
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = grid_dim.x * BLOCK_SIZE

    var rs_config = ReduceScatterConfig[dtype, ngpus](_num_elements, stride)

    comptime num_tensors = 1 if use_multimem else ngpus

    with PDL():
        # --- Define tmp buffers by offsetting for Signal struct ---
        # Round-robin access pattern to balance NVLink traffic across GPUs.
        # Skip Signal header.
        comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
        var tmps = Array[_, ngpus](
            fill_with_unrolled=lambda [i: Int]() -> PtrType: (
                rank_sigs[circular_add[ngpus](_my_rank, i)].address_space_cast[
                    .GENERIC
                ]()
                + 1
            ).bitcast[Scalar[dtype]]()
        )

        # Current rank's output buffer.
        var tmp_out = tmps[0]

        # --- Stage 1: Reduce-Scatter Phase ---
        # Uses two-phase synchronization protocol with release-acquire semantics:
        # 1. Initial barrier establishes happens-before relationship.
        # 2. Memory fence ensures visibility of partial reductions.
        _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )

        # TODO(KERN-2273): Remove this once temporary buffers removed
        # Output lambda for reduce-scatter: write to scratch buffer
        var tmp_buff = TileTensor[mut=True, dtype](
            tmp_out, row_major(rs_config.rank_part(_my_rank))
        )

        @always_inline
        @__parameter
        @__copy_capture(tmp_buff)
        def rs_output_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            tmp_buff.address_space_cast[_target_address_space]().store[
                width=_width, alignment=_alignment
            ](
                coords,
                val.cast[dtype](),
            )

        # Slice input tiles to this rank's partition for reduce-scatter.
        var elem_start = rs_config.rank_start(_my_rank)
        var n_elements = rs_config.rank_num_elements(_my_rank)
        comptime SlicedTile = TileTensor[dtype, SlicedLayout, ImmutAnyOrigin]
        comptime SlicedLayout = type_of(row_major(n_elements))
        # Round-robin access pattern to balance NVLink traffic across GPUs.
        var sliced_tiles = Array[_, num_tensors](
            fill_with_unrolled=lambda [i: Int]() -> SlicedTile: SlicedTile(
                src_tensors[
                    0 if num_tensors
                    == 1 else circular_add[num_tensors](_my_rank, i)
                ]._storage
                + elem_start,
                row_major(n_elements),
            )
        )

        _reduce_scatter_impl[
            ngpus, output_lambda=rs_output_lambda, use_multimem=use_multimem
        ](sliced_tiles, tmp_buff, n_elements, rs_config.stride)

        # Second barrier with memory ordering guarantees.
        _multi_gpu_barrier[
            ngpus, is_start=False, need_fence=True, domain_id=domain_id
        ](rank_sigs, my_sig, _my_rank)

        # --- Stage 2: All-Gather Phase ---
        # Maintains thread index consistency to satisfy memory model:
        # The same tid guarantees visibility of prior writes.
        # So if thread `idx` computes the sum of `start + idx` in the first stage,
        # then thread `idx` also gathers `start + idx` from all ranks.
        comptime simd_width = rs_config.simd_width
        comptime alignment = rs_config.alignment

        # Ragged handling:
        # GPU-0 is guaranteed to have largest partition
        # GPU-ngpus-1 has smallest partition (only 1 simd vector smaller)

        # Main loop - only process unragged elements (no bounds check)
        for idx in range(
            rs_config.thr_local_start(global_tid),
            rs_config.rank_part(ngpus - 1),
            rs_config.stride,
        ):
            comptime for gpu_idx in range(ngpus):
                var peer_rank = circular_add[ngpus](_my_rank, gpu_idx)

                var dst_idx = rs_config.rank_start(peer_rank) + idx
                output_lambda[width=simd_width, alignment=alignment](
                    result.layout.idx2crd(dst_idx),
                    tmps[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[width=simd_width, alignment=alignment](idx),
                )

        # Ragged tail - max 1 simd vector per gpu, spread work between threads
        if global_tid < ngpus:
            var peer_rank = circular_add[ngpus](_my_rank, global_tid)
            if peer_rank < rs_config.axis_remainder:
                var idx = (
                    rs_config.rank_part(0) - simd_width
                )  # last ragged simd_vector
                var dst_idx = rs_config.rank_start(peer_rank) + idx
                output_lambda[width=simd_width, alignment=alignment](
                    result.layout.idx2crd(dst_idx),
                    tmps[global_tid]
                    .address_space_cast[_target_address_space]()
                    .load[width=simd_width, alignment=alignment](idx),
                )


@always_inline
def _allreduce_1stage_reduce_store_one[
    dtype: DType,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    num_tensors: Int,
    *,
    accum_type: DType,
    output_lambda: elementwise_epilogue_type,
](
    elem_idx: Int,
    ptrs: Array[TileTensor[dtype, in_layout, ImmutAnyOrigin], num_tensors],
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
) -> None:
    """Load one element from every peer, reduce in ``accum_type``, epilogue store.
    """
    comptime scalar_align = align_of[SIMD[dtype, 1]]()
    var accum = (
        ptrs[0]
        .address_space_cast[_target_address_space]()
        .load[width=1, alignment=scalar_align, invariant=True](Coord(elem_idx))
        .cast[accum_type]()
    )
    comptime for gpu_idx in range(1, num_tensors):
        accum += (
            ptrs[gpu_idx]
            .address_space_cast[_target_address_space]()
            .load[width=1, alignment=scalar_align, invariant=True](
                Coord(elem_idx)
            )
            .cast[accum_type]()
        )
    var reduced = accum.cast[dtype]()
    output_lambda[width=1, alignment=scalar_align](
        result.layout.idx2crd(elem_idx),
        reduced,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allreduce_1stage_{dtype}_{use_multimem}")
def _allreduce_1stage_kernel[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
    domain_id: Int = 0,
](
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_tensors: Array[
        TileTensor[dtype, in_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int32,
    my_rank: Int32,
):
    """
    Kernel implementing allreduce using peer-to-peer access between GPUs.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        use_multimem: If True, use multi-memory space buffers for input.
        domain_id: Barrier counter bank to use (0 for full-world; a distinct
            nonzero value for grouped collectives). See `_multi_gpu_barrier`.

    Args:
        result: Output tensor for reduced values
        src_tensors: Input tensors from all GPUs
        rank_sigs: Signal pointers for synchronization
        num_elements: Number of elements to reduce
        my_rank: Current GPU rank

    Uses P2P access to directly read from other GPU buffers and perform reduction.
    Synchronizes using _multi_gpu_barrier before and after reduction.

    **Non-multimem path:** grid-strided loop over full SIMD vectors using
    ``_load_reduce`` (same as historical 1-stage performance). If
    ``num_elements`` is not divisible by ``simd_width``, the aligned prefix is
    still processed as SIMD vectors; the remaining ``< simd_width`` scalars use
    ``_allreduce_1stage_reduce_store_one``.

    **Multimem path:** unchanged vectorized SIMD loads over full
    ``simd_width`` vectors (input size must remain SIMD-aligned).
    """
    comptime accum_type = get_accum_type[dtype]()
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var total_threads = grid_dim.x * BLOCK_SIZE
    var _my_rank = Int(my_rank)
    var _num_elements = Int(num_elements)
    var my_sig = rank_sigs[_my_rank]

    # Route input pointers according to round-robin pattern.
    # For 8 GPUs: Rank 0 accesses 0→1→2→...→7, Rank 1 accesses 1→2→...→7→0, etc.
    comptime num_tensors = 1 if use_multimem else ngpus
    # It's safe to prefetch the input pointers
    comptime TileType = TileTensor[dtype, in_layout, ImmutAnyOrigin]
    var ptrs = Array[_, num_tensors](
        fill_with_unrolled=lambda [i: Int]() -> TileType: src_tensors[
            0 if num_tensors == 1 else circular_add[num_tensors](_my_rank, i)
        ]
    )

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )

        comptime if use_multimem:
            var num_simd_chunks = _num_elements // simd_width
            for idx in range(global_tid, num_simd_chunks, total_threads):
                var elem_idx = idx * simd_width
                var reduced_result = _load_reduce[
                    ngpus,
                    simd_width=simd_width,
                    alignment=alignment,
                    accum_type=accum_type,
                    use_multimem=use_multimem,
                ](elem_idx, ptrs)
                output_lambda[width=simd_width, alignment=alignment](
                    result.layout.idx2crd(elem_idx), reduced_result
                )
        else:
            var ptrs_ngpus = rebind[
                Array[TileTensor[dtype, in_layout, ImmutAnyOrigin], ngpus]
            ](ptrs).copy()
            var num_simd_vectors = _num_elements // simd_width
            var simd_prefix_elems = num_simd_vectors * simd_width
            if num_simd_vectors > 0:
                for idx in range(global_tid, num_simd_vectors, total_threads):
                    var elem_idx = idx * simd_width
                    var reduced_result = _load_reduce[
                        ngpus,
                        simd_width=simd_width,
                        alignment=alignment,
                        accum_type=accum_type,
                        use_multimem=False,
                    ](elem_idx, ptrs_ngpus)
                    output_lambda[width=simd_width, alignment=alignment](
                        result.layout.idx2crd(elem_idx), reduced_result
                    )
            if simd_prefix_elems < _num_elements:
                for elem_idx in range(
                    simd_prefix_elems + global_tid,
                    _num_elements,
                    total_threads,
                ):
                    _allreduce_1stage_reduce_store_one[
                        dtype,
                        in_layout,
                        out_layout,
                        ngpus,
                        accum_type=accum_type,
                        output_lambda=output_lambda,
                    ](elem_idx, ptrs_ngpus, result)

        _multi_gpu_barrier[ngpus, is_start=False, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )


@always_inline
def _lamport_supported() -> Bool:
    """Whether the current GPU target is cleared for the Lamport protocol.

    The barrier-free Lamport allreduce relies on a naturally-aligned 128-bit
    volatile load/store being a single, peer-visible transaction.
    """
    return is_nvidia_gpu() or is_amd_gpu()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allreduce_lamport_{dtype}_{use_fence}")
def _allreduce_lamport_kernel[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_fence: Bool = False,
](
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_tensors: Array[TileTensor[dtype, in_layout, ImmutAnyOrigin], ngpus],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int32,
    my_rank: Int32,
):
    """Barrier-free one-shot Lamport allreduce for small, latency-bound transfers.

    Replaces the two cross-GPU counter barriers of `_allreduce_1stage_kernel`
    with a push-then-poll protocol keyed on the negative-zero sentinel.

    Per thread, over a grid-strided range of 128-bit packs:

    1. load this rank's local input pack and `remove_neg_zero` (producer
       sanitize -- the correctness linchpin); the local term is kept in-register
       and seeds the sum, so this rank only touches the `ngpus - 1` REMOTE slots;
    2. push the sanitized pack into each peer's generation buffer at this rank's
       slot (skipping self), round-robin over peers (`circular_add`) for balance;
    3. `comptime if use_fence`: emit a system-scope release `atomic.fence` after
       the push (default off; slice 2 proved fenceless is correct on both
       targets for the single-location sentinel scheme);
    4. poll all `ngpus - 1` remote slots in this rank's own buffer, re-reading
       until none still hold the sentinel (observes parallel arrival rather than
       spinning on one peer at a time);
    5. reduce the in-register local seed plus the arrived peers in
       `get_accum_type[dtype]`, apply `output_lambda`, and write the result;
    6. fused into the same loop, clear this pack's remote slots in the generation
       reused two calls from now (`(flag+2)%3`) over the extent the previous
       writer wrote (`prev_num_elements`), priming it for reuse (a short tail
       loop covers any leftover when the previous call was larger);
    7. advance this rank's generation counter exactly once (grid-barrier
       epilogue), recording this call's size for the next call's clear.

    The generation counter (`flag`) and clear extent (`prev_num_elements`) are
    read from and advanced in this rank's device-resident `Signal.lamport_state`

    Parameters:
        dtype: Data dtype of tensor elements (bf16/fp16/fp32 transport).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        use_fence: If True, emit a system-scope release fence after each push and
            an acquire fence after each peer read. Default False (fenceless).

    Args:
        result: Output buffer for reduced values.
        src_tensors: Input buffers from all GPUs.
        rank_sigs: Signal pointers; the Lamport comm region and `lamport_state`
            follow the header.
        num_elements: Number of elements to reduce. Must be a multiple of the
            128-bit pack width (the dispatch falls back to the 1-stage kernel
            otherwise).
        my_rank: Current GPU rank.
    """
    comptime assert (
        _lamport_supported()
    ), "Lamport allreduce support not established on this hardware."
    comptime accum_type = get_accum_type[dtype]()
    # Pin the pack to the 128-bit single-copy-atomic transaction width the
    # sentinel protocol depends on, NOT the dtype's natural SIMD width (4xfp32 /
    # 8xbf16). The atomicity guarantee that couples "data present" with
    # "sentinel gone" holds only at exactly 16 bytes.
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    comptime assert (
        atomic_width * size_of[dtype]() == 16
    ), "Lamport pack must be exactly 16 bytes (the 128-bit atomic width)."
    comptime alignment = align_of[SIMD[dtype, atomic_width]]()

    var global_tid = global_idx.x
    var total_threads = grid_dim.x * BLOCK_SIZE
    var _my_rank = Int(my_rank)
    var _num_elements = Int(num_elements)

    # Whole 128-bit packs only; a scalar tail cannot carry the sentinel, so the
    # dispatch routes non-pack-aligned sizes to the 1-stage path.
    var num_packs = _num_elements // atomic_width

    # This rank's own comm region (where it polls for peer data).
    var my_region = rank_sigs[_my_rank][].lamport_region_ptr[dtype]()

    # Peer comm-region bases in round-robin order (balances NVLink/XGMI traffic
    # the same way the existing kernels do).
    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var peer_regions = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: rank_sigs[
            circular_add[ngpus](_my_rank, i)
        ][].lamport_region_ptr[dtype]()
    )

    var sentinel = set_neg_zero[dtype, atomic_width]()

    with PDL():
        var state = rank_sigs[_my_rank][].lamport_state_ptr()
        var flag = Int(state.load[width=1, volatile=True](Lamport.STATE_FLAG))
        var clear_size = Int(
            state.load[width=1, volatile=True](Lamport.STATE_PREV_ELEMS)
        )
        # Packs to reset in the generation cleared this call
        var clear_packs = clear_size // atomic_width

        # Generation geometry. The per-generation stride is FIXED at the reserved
        # capacity (`Lamport.MAX_PACKS` packs per rank slot), NOT this call's
        # `num_packs`, so generation `g` always occupies the same buffer region
        # regardless of message size and calls of different sizes never alias. Slot
        # s of generation g starts at pack index
        # `g * (ngpus * Lamport.MAX_PACKS) + s * Lamport.MAX_PACKS`.
        comptime gen_stride_packs = ngpus * Lamport.MAX_PACKS
        var data_gen = LamportGeneration.data_index(flag)
        var clear_gen = LamportGeneration.clear_index(flag)
        var data_gen_off = data_gen * gen_stride_packs
        var clear_gen_off = clear_gen * gen_stride_packs

        for pack in range(global_tid, num_packs, total_threads):
            var elem_idx = pack * atomic_width

            # (a) Load this rank's local input pack and producer-sanitize. The
            # local contribution stays in-register and seeds the sum (e); it is
            # never staged through the comm region, so this rank only pushes,
            # polls, and clears the `ngpus - 1` REMOTE slots.
            var local = remove_neg_zero[dtype, atomic_width](
                src_tensors[_my_rank]
                .address_space_cast[_target_address_space]()
                .load[width=atomic_width, alignment=alignment](Coord(elem_idx))
            )

            # (b) Push into each PEER's generation buffer at this rank's slot
            # (skip self -- i in 1..ngpus-1; peer_regions is round-robin order so
            # iterating i balances fabric traffic).
            var push_off = data_gen_off + _my_rank * Lamport.MAX_PACKS + pack
            comptime for i in range(1, ngpus):
                peer_regions[i].store[
                    width=atomic_width, alignment=alignment, volatile=True
                ](push_off * atomic_width, local)

            # (c) Optional release fence (default off; see slice-2 result).
            comptime if use_fence:
                fence[ordering=Ordering.RELEASE, scope=StaticString("")]()

            # (d) Poll all `ngpus - 1` remote slots in this rank's own buffer,
            # re-reading every slot until none still hold the sentinel. This
            # observes parallel arrival, unlike spinning on one peer at a time
            # (which serializes the wait behind the slowest-ordered peer).
            var peer_packs = Array[SIMD[dtype, atomic_width], ngpus](
                uninitialized=True
            )
            var done = False
            while not done:
                done = True
                comptime for i in range(1, ngpus):
                    var peer = circular_add[ngpus](_my_rank, i)
                    var slot_off = (
                        data_gen_off + peer * Lamport.MAX_PACKS + pack
                    )
                    var p = my_region.load[
                        width=atomic_width, alignment=alignment, volatile=True
                    ](slot_off * atomic_width)
                    peer_packs[i] = p
                    if has_neg_zero[dtype, atomic_width](p):
                        done = False
            comptime if use_fence:
                fence[ordering=Ordering.ACQUIRE, scope=StaticString("")]()

            # (e) Reduce (in-register local seed + the arrived peers) and write.
            var accum = local.cast[accum_type]()
            comptime for i in range(1, ngpus):
                accum += peer_packs[i].cast[accum_type]()
            output_lambda[width=atomic_width, alignment=alignment](
                result.layout.idx2crd(elem_idx), accum.cast[dtype]()
            )

            # (f) Fused clear: reset this pack's slots in the generation reused
            # two calls from now (the `ngpus - 1` remote slots we poll) back to
            # the sentinel, while still within the previous writer's extent.
            # Fusing into the main loop avoids a separate full pass over the
            # region (the dominant per-call overhead vs the 1-stage path).
            if pack < clear_packs:
                comptime for i in range(1, ngpus):
                    var peer = circular_add[ngpus](_my_rank, i)
                    var clr_off = (
                        clear_gen_off + peer * Lamport.MAX_PACKS + pack
                    )
                    my_region.store[
                        width=atomic_width, alignment=alignment, volatile=True
                    ](clr_off * atomic_width, sentinel)

        # (f-tail) If the previous call was LARGER than this one, the fused clear
        # only covered packs [0, num_packs); clear the leftover
        # [num_packs, clear_packs) of the reused generation here. Empty (no
        # second pass) in the common same-size case where clear_packs ==
        # num_packs.
        for cp in range(num_packs + global_tid, clear_packs, total_threads):
            comptime for i in range(1, ngpus):
                var peer = circular_add[ngpus](_my_rank, i)
                var clr_off = clear_gen_off + peer * Lamport.MAX_PACKS + cp
                my_region.store[
                    width=atomic_width, alignment=alignment, volatile=True
                ](clr_off * atomic_width, sentinel)

        # (g) Advance this rank's generation flag exactly once per call via an
        # intra-GPU block-arrival counter.
        barrier()
        if thread_idx.x == 0:
            var arrived = Atomic.fetch_add(
                state + Lamport.STATE_ARRIVAL, UInt32(1)
            )
            if Int(arrived) == Int(grid_dim.x) - 1:
                state.store[volatile=True](Lamport.STATE_FLAG, UInt32(flag + 1))
                state.store[volatile=True](
                    Lamport.STATE_PREV_ELEMS, UInt32(_num_elements)
                )
                state.store[volatile=True](Lamport.STATE_ARRIVAL, UInt32(0))


@always_inline
def _allreduce_lamport_p2p[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel,
    use_fence: Bool = False,
](
    list_of_in_tensors: Array[TileTensor[dtype, in_layout, in_origin], ngpus],
    out_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    dispatch_config: AllReduceTuningConfig,
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Launches the barrier-free Lamport allreduce on this GPU.

    Requires whole 128-bit packs (`num_elements % atomic_width == 0`) and a
    per-rank message no larger than `Lamport.MAX_SMALL_MESSAGE_BYTES`; callers
    that cannot meet these must route to `_allreduce_p2p` (the 1-stage fallback)
    instead. The generation counter and clear extent are device-resident in each
    rank's `Signal.lamport_state` (advanced in-kernel); the signal buffers must
    have been sentinel-initialized once (e.g. via `Signals.buffers()` /
    `init_signal_buffer`) before the first call.
    """
    # Pin to the 128-bit atomic width the kernel uses (not the natural SIMD
    # width); see the kernel docstring.
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    var num_elements = list_of_in_tensors[0].num_elements()
    if num_elements == 0:
        return

    var num_bytes = num_elements * size_of[dtype]()
    if num_elements % atomic_width != 0:
        raise Error(
            "Lamport allreduce requires the element count to be a multiple of"
            " the 128-bit pack width (whole 128-bit packs)"
        )
    if num_bytes > Lamport.MAX_SMALL_MESSAGE_BYTES:
        raise Error(
            "Lamport allreduce message exceeds reserved workspace ("
            + String(num_bytes)
            + " > "
            + String(Lamport.MAX_SMALL_MESSAGE_BYTES)
            + " bytes); route to the 1-stage path"
        )

    comptime FlatLayout = type_of(row_major(num_elements))
    comptime FlatIn = TileTensor[dtype, FlatLayout, ImmutAnyOrigin]
    var flat_inputs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> FlatIn: FlatIn(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                list_of_in_tensors[i]._storage
            ),
            row_major(num_elements),
        )
    )

    comptime sm_version = ctx.default_device_info.version
    comptime BLOCK_SIZE = 512 if sm_version == "CDNA4" else 256
    var num_packs = num_elements // atomic_width
    var grid_size = clamp(
        ceildiv(num_packs, BLOCK_SIZE), 1, dispatch_config.num_blocks
    )

    comptime lamport_kernel = _allreduce_lamport_kernel[
        dtype,
        ngpus,
        FlatLayout,
        out_layout,
        BLOCK_SIZE=BLOCK_SIZE,
        output_lambda=output_lambda,
        use_fence=use_fence,
    ]
    ctx.enqueue_function[lamport_kernel](
        rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
        flat_inputs,
        rank_sigs,
        Int32(num_elements),
        Int32(my_rank),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def _allreduce_p2p[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel,
    use_multimem: Bool = False,
    domain_id: Int = 0,
](
    list_of_in_tensors: Array[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    out_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    dispatch_config: AllReduceTuningConfig,
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """
    Performs allreduce using peer-to-peer access for a single GPU.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        output_lambda: An output elementwise lambda.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: If True, use multi-memory space buffers for input.
        domain_id: Barrier counter bank to use (0 for full-world; a distinct
            nonzero value for grouped collectives). See `_multi_gpu_barrier`.

    Args:
        list_of_in_tensors: Input buffers from ALL GPUs (peer access required)
        out_tensor: Output buffer for THIS GPU
        rank_sigs: Signal pointers for synchronization
        dispatch_config: Dispatch configuration defining block count, algorithm selection
        ctx: Device context for THIS GPU
        my_rank: Rank of THIS GPU within the allreduce group.

    Launches P2P reduction kernel on the current GPU to perform direct reduction.
    """
    comptime num_tensors = 1 if use_multimem else ngpus
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    var num_elements = list_of_in_tensors[0].num_elements()

    # Do nothing if there are no elements to reduce.
    if num_elements == 0:
        return

    if use_multimem and num_elements % simd_width != 0:
        raise Error(
            "multimem allreduce requires the element count to be a multiple of"
            " SIMD width"
        )

    # Flatten inputs to 1D - allreduce does not need dimension info
    comptime FlatLayout = type_of(row_major(num_elements))
    comptime FlatIn = TileTensor[dtype, FlatLayout, ImmutAnyOrigin]
    var flat_inputs = Array[_, num_tensors](
        fill_with=lambda (i: Int) -> FlatIn: FlatIn(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                list_of_in_tensors[i]._storage
            ),
            row_major(num_elements),
        )
    )

    var max_num_blocks = dispatch_config.num_blocks

    # Barrier-free Lamport path. The tuning table selects Lamport by byte
    # bucket, but the kernel additionally requires whole 128-bit packs and a
    # per-rank message within the reserved small-message workspace, and it is
    # incompatible with multimem. Grouped collectives (nonzero domain) are
    # excluded: the per-device Lamport generation state is shared with
    # full-world collectives, and mixing group sizes would skew generations
    # across devices. Grouped allreduces take the barrier-based kernels,
    # whose domain banks isolate them.
    comptime lamport_atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    comptime if not use_multimem and domain_id == 0:
        if (
            dispatch_config.algorithm == AllReduceAlgorithm.LAMPORT
            and num_elements % lamport_atomic_width == 0
            and num_elements * size_of[dtype]()
            <= Lamport.MAX_SMALL_MESSAGE_BYTES
        ):
            return _allreduce_lamport_p2p[
                dtype,
                ngpus,
                in_layout,
                in_origin,
                out_layout,
                output_lambda=output_lambda,
                pdl_level=pdl_level,
            ](
                rebind[Array[TileTensor[dtype, in_layout, in_origin], ngpus]](
                    list_of_in_tensors
                ),
                out_tensor,
                rank_sigs,
                dispatch_config,
                ctx,
                my_rank,
            )

    # 1-stage unless the config explicitly selects 2-stage (a non-2-stage
    # algorithm that reached here -- including a multimem Lamport fall-through --
    # runs 1-stage).
    var use_1stage = dispatch_config.algorithm != AllReduceAlgorithm.TWO_STAGE
    # TODO(KERN-2632): Incorporate this into dispatch table
    comptime sm_version = ctx.default_device_info.version
    comptime BLOCK_SIZE = 512 if sm_version == "CDNA4" else 256

    # The 2-stage path partitions by full SIMD vectors only; use 1-stage when a
    # scalar tail is present (unless multimem, which is rejected above).
    use_1stage = use_1stage or (num_elements % simd_width != 0)

    if use_1stage:
        var grid_size: Int
        comptime if use_multimem:
            var simd_chunks = num_elements // simd_width
            var tail_elems_mm = num_elements - simd_chunks * simd_width
            grid_size = clamp(
                ceildiv(max(simd_chunks, tail_elems_mm), BLOCK_SIZE),
                1,
                max_num_blocks,
            )
        else:
            var num_simd_vecs = num_elements // simd_width
            var tail_elems = num_elements - num_simd_vecs * simd_width
            var grid_simd = ceildiv(num_simd_vecs, BLOCK_SIZE)
            var grid_tail = ceildiv(tail_elems, BLOCK_SIZE)
            grid_size = clamp(max(grid_simd, grid_tail), 1, max_num_blocks)

        # Use the 1-stage allreduce when transfer is latency bound.
        comptime allreduce_1stage_kernel = _allreduce_1stage_kernel[
            dtype,
            ngpus,
            FlatLayout,
            out_layout,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=output_lambda,
            use_multimem=use_multimem,
            domain_id=domain_id,
        ]
        ctx.enqueue_function[allreduce_1stage_kernel](
            rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
            flat_inputs,
            rank_sigs,
            Int32(num_elements),
            Int32(my_rank),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # Define grid size for 2-stage, which processes 1/ngpus of the
        # number of elements.
        var grid_size = clamp(
            ceildiv(num_elements // (simd_width * ngpus), BLOCK_SIZE),
            1,
            max_num_blocks,
        )

        # Otherwise, use 2-stage allreduce for the bandwidth bound regime.
        comptime kernel = _allreduce_2stage_kernel[
            dtype,
            ngpus,
            FlatLayout,
            out_layout,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=output_lambda,
            use_multimem=use_multimem,
            domain_id=domain_id,
        ]
        ctx.enqueue_function[kernel](
            rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
            flat_inputs,
            rank_sigs,
            Int32(num_elements),
            Int32(my_rank),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )


@__parameter
def allreduce[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    use_multimem: Bool = False,
    domain_id: Int = 0,
](
    input_tensors: Array[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    output_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
    local_rank: Optional[Int] = None,
) raises:
    """Per-device allreduce: one instance per GPU builds its own output.

    High-level model
    - Each GPU runs one instance of this function in parallel with the others.
    - Every instance reads all inputs but writes only its own output buffer.
    - A Python-level fence is inserted across the outputs to prevent reordering.

    Two execution paths
    1) P2P fast path (when peer access is available)
       - 1-stage kernel (latency-bound): each thread vector-loads from all GPUs,
         accumulates in higher precision, and writes directly to the result.
       - 2-stage kernel (bandwidth-bound): reduce-scatter then all-gather.
         Uses each GPU's `rank_sigs[*]` payload as a staging area for partitions.

         Diagram (per GPU r, 2-stage):
           - Stage 1: write reduced partition r into payload of `rank_sigs[r]`.
           - Stage 2: gather partitions from all peers' payloads into `out_r`.

    2) Naive fallback (no P2P)
       - For GPU r: create local accumulator A_r, allocate a temporary buffer S_r,
         copy each peer input into S_r and accumulate into A_r, then apply the epilogue
         into `out_r`.

         Diagram (per GPU r, naive):
           in_r -> A_r += in_r; for i!=r: in_i -> tmp_r -> A_r += tmp_r; A_r -> out_r

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating in the allreduce.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        output_lambda: Elementwise epilogue applied on the device result.
        pdl_level: Controls PDL behavior for P2P kernels.
        use_multimem: Whether to use multimem mode for improved performance.
        domain_id: Barrier counter bank to use (0 for full-world; a distinct
            nonzero value for grouped collectives sharing the same Signal
            buffers with full-world ones). See `_multi_gpu_barrier`.

    Args:
        input_tensors: Inputs from ALL GPUs as TileTensors.
        output_tensor: Output for THIS GPU as a TileTensor.
        rank_sigs: Per-GPU Signal pointers.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional grid limit.
        local_rank: Optional rank of THIS GPU within the allreduce group.
            Defaults to the device id, which is only correct for a full-world
            collective over devices 0..ngpus-1.

    Notes:
      - Inputs must have identical shape/dtype across GPUs.
      - Signal buffers must be sized at least `size_of(Signal) + payload_bytes`
        for the P2P 2-stage path, where `payload_bytes` equals the input
        tensor bytecount.
      - The naive path is automatically selected if P2P cannot be enabled.
      - The `use_multimem` parameter requires P2P access between GPUs.
    """
    comptime assert ngpus >= 2, "allreduce requires at least 2 GPUs"
    comptime num_tensors = 1 if use_multimem else ngpus

    # Return early, if the input buffer is empty
    var num_elements = input_tensors[0].num_elements()
    if num_elements == 0:
        return

    @always_inline
    @__parameter
    @__copy_capture(output_tensor)
    def default_output_lambda[
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        output_tensor.store[width=_width, alignment=_alignment](
            coords, val.cast[dtype]()
        )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    # TODO: check all devices have the same GPU sm_version
    comptime sm_version = ctx.default_device_info.version
    var num_bytes = num_elements * size_of[dtype]()
    var dispatch_config = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](num_bytes)

    if _max_num_blocks:
        dispatch_config.num_blocks = _max_num_blocks.value()

    if dispatch_config.num_blocks > MAX_NUM_BLOCKS_UPPER_BOUND:
        raise Error(
            "expected allreduce num_blocks less than upper bound: "
            + String(MAX_NUM_BLOCKS_UPPER_BOUND)
            + " but got: "
            + String(dispatch_config.num_blocks)
        )

    var my_rank = local_rank.value() if local_rank else Int(ctx.id())

    # Check P2P availability.
    if not is_p2p_enabled():
        comptime if use_multimem:
            raise Error(
                "Allreduce with multimem requires P2P access between GPUs"
            )
        comptime if domain_id != 0:
            # The naive fallback assumes the participating devices are
            # 0..ngpus-1; a grouped collective's devices are an arbitrary
            # contiguous run, so it would copy from the wrong peers.
            raise Error(
                "grouped allreduce (nonzero domain_id) requires P2P access"
                " between GPUs"
            )
        return _allreduce_naive_single[
            ngpus=ngpus,
            output_lambda=actual_output_lambda,
            num_tensors=1 if use_multimem else ngpus,
        ](input_tensors, output_tensor, dispatch_config.num_blocks, ctx)

    # P2P path.
    return _allreduce_p2p[
        ngpus=ngpus,
        output_lambda=actual_output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
        domain_id=domain_id,
    ](input_tensors, output_tensor, rank_sigs, dispatch_config, ctx, my_rank)
