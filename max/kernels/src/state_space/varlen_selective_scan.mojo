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

"""Variable-length selective scan kernels for Mamba SSM architecture."""

from max.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from layout import DefaultEngine, TensorLayout, TensorEngine, TileTensor
from std.utils.index import IndexList
from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
import std.math
from std.math import exp2
from nn.activations import silu
from state_space.selective_scan import softplus

# LOG2E constant for converting exp to exp2 (faster on GPU)
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 256  # Larger for Mamba-2 models

# Stride types for passing tensor strides to kernels
comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


def varlen_selective_state_update_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    state_LT: TensorLayout,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    z_LT: TensorLayout,
    output_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    state_batch_indices_LT: TensorLayout,
    Engine: TensorEngine = DefaultEngine[element_width=1],
](
    # Grid dimensions
    total_threads: Int32,  # batch * nheads * dim / BLOCK_SIZE_M
    batch: Int32,
    nheads: Int32,
    dim: Int32,
    nheads_ngroups_ratio: Int32,
    pad_slot_id: Int32,
    dt_softplus: Int8,
    has_state_batch_indices: Int8,
    # Tensors
    state: TileTensor[
        kernel_dtype, state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    x: TileTensor[kernel_dtype, x_LT, MutUntrackedOrigin, Engine=Engine],
    dt: TileTensor[kernel_dtype, dt_LT, MutUntrackedOrigin, Engine=Engine],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin, Engine=Engine],
    B: TileTensor[kernel_dtype, B_LT, MutUntrackedOrigin, Engine=Engine],
    C: TileTensor[kernel_dtype, C_LT, MutUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin, Engine=Engine],
    z: TileTensor[kernel_dtype, z_LT, MutUntrackedOrigin, Engine=Engine],
    output: TileTensor[
        kernel_dtype, output_LT, MutUntrackedOrigin, Engine=Engine
    ],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, MutUntrackedOrigin, Engine=Engine
    ],
    state_batch_indices: TileTensor[
        .int32, state_batch_indices_LT, MutUntrackedOrigin, Engine=Engine
    ],
    state_strides: Strides4D,  # (batch, nheads, dim, dstate)
    x_strides: Strides3D,  # (batch, nheads, dim)
    dt_strides: Strides3D,  # (batch, nheads, dim)
    dt_bias_strides: Strides2D,  # (nheads, dim)
    A_strides: Strides3D,  # (nheads, dim, dstate)
    B_strides: Strides3D,  # (batch, ngroups, dstate)
    C_strides: Strides3D,  # (batch, ngroups, dstate)
    D_strides: Strides2D,  # (nheads, dim)
    z_strides: Strides3D,  # (batch, nheads, dim)
    out_strides: Strides3D,  # (batch, nheads, dim)
):
    """GPU kernel for selective state update with multi-head support."""
    var _total_threads = Int(total_threads)
    var _batch = Int(batch)
    var _nheads = Int(nheads)
    var _dim = Int(dim)
    var _nheads_ngroups_ratio = Int(nheads_ngroups_ratio)
    comptime BLOCK_SIZE_M = 4  # Process 4 dims per thread

    var pid_m = block_idx.x  # Dim block index
    var pid_b = block_idx.y  # Batch index
    var pid_h = block_idx.z  # Head index

    if pid_b >= _batch or pid_h >= _nheads:
        return

    # Determine state _batch index
    var state_batch_idx = Int32(pid_b)
    if Bool(Int(has_state_batch_indices) != 0):
        state_batch_idx = state_batch_indices.raw_load(pid_b)
        # Check for padding
        if state_batch_idx == pad_slot_id:
            return

    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_D = Int(D.dim[0]()) > 0
    var has_z = Int(z.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = pid_h // _nheads_ngroups_ratio

    # Process BLOCK_SIZE_M dims per thread
    comptime for local_m in range(BLOCK_SIZE_M):
        var m = pid_m * BLOCK_SIZE_M + local_m
        if m >= _dim:
            continue

        # Load x value
        var x_offset = UInt32(
            pid_b * x_strides[0] + pid_h * x_strides[1] + m * x_strides[2]
        )
        var x_val = Scalar[kernel_dtype](x.raw_load(x_offset)).cast[
            DType.float32
        ]()

        # Load dt value
        var dt_offset = UInt32(
            pid_b * dt_strides[0] + pid_h * dt_strides[1] + m * dt_strides[2]
        )
        var dt_val = Scalar[kernel_dtype](dt.raw_load(dt_offset)).cast[
            DType.float32
        ]()

        # Apply dt_bias if present
        if has_dt_bias:
            var dt_bias_offset = UInt32(
                pid_h * dt_bias_strides[0] + m * dt_bias_strides[1]
            )
            var bias_val = Scalar[kernel_dtype](
                dt_bias.raw_load(dt_bias_offset)
            ).cast[.float32]()
            dt_val += bias_val

        # Apply softplus if requested
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var out_val = Float32(0.0)

        # Process each dstate element
        comptime for n in range(DSTATE):
            # Load A value
            var A_offset = UInt32(
                pid_h * A_strides[0] + m * A_strides[1] + n * A_strides[2]
            )
            var A_val = Scalar[kernel_dtype](A.raw_load(A_offset)).cast[
                DType.float32
            ]()

            # Compute dA = exp(A * dt) using exp2 for faster GPU execution
            var dA = exp2(A_val * LOG2E * dt_val)

            # Load B value
            var B_offset = UInt32(
                pid_b * B_strides[0]
                + group_id * B_strides[1]
                + n * B_strides[2]
            )
            var B_val = Scalar[kernel_dtype](B.raw_load(B_offset)).cast[
                DType.float32
            ]()

            # Compute dB = B * dt
            var dB = B_val * dt_val

            # Load current state
            var state_offset = UInt32(
                Int(state_batch_idx) * state_strides[0]
                + pid_h * state_strides[1]
                + m * state_strides[2]
                + n * state_strides[3]
            )
            var state_val = Scalar[kernel_dtype](
                state.raw_load(state_offset)
            ).cast[.float32]()

            # Update state: state = state * dA + dB * x
            state_val = state_val * dA + dB * x_val

            # Store updated state
            state.raw_store(
                state_offset,
                Scalar[kernel_dtype](state_val.cast[kernel_dtype]()),
            )

            # Load C value
            var C_offset = UInt32(
                pid_b * C_strides[0]
                + group_id * C_strides[1]
                + n * C_strides[2]
            )
            var C_val = Scalar[kernel_dtype](C.raw_load(C_offset)).cast[
                DType.float32
            ]()

            # Accumulate output
            out_val += state_val * C_val

        # Add skip connection if D is present
        if has_D:
            var D_offset = UInt32(pid_h * D_strides[0] + m * D_strides[1])
            var D_val = Scalar[kernel_dtype](D.raw_load(D_offset)).cast[
                DType.float32
            ]()
            out_val += x_val * D_val

        # Apply gating if z is present, using optimized silu
        if has_z:
            var z_offset = UInt32(
                pid_b * z_strides[0] + pid_h * z_strides[1] + m * z_strides[2]
            )
            var z_val = Scalar[kernel_dtype](z.raw_load(z_offset)).cast[
                DType.float32
            ]()
            out_val *= silu(z_val)

        # Store output
        var out_offset = UInt32(
            pid_b * out_strides[0] + pid_h * out_strides[1] + m * out_strides[2]
        )
        output.raw_store(
            out_offset, Scalar[kernel_dtype](out_val.cast[kernel_dtype]())
        )


def varlen_selective_scan_fwd_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    u_LT: TensorLayout,
    delta_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    z_LT: TensorLayout,
    delta_bias_LT: TensorLayout,
    ssm_states_LT: TensorLayout,
    output_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    Engine: TensorEngine = DefaultEngine[element_width=1],
](
    dim: Int32,
    ngroups: Int32,
    batch: Int32,
    pad_slot_id: Int32,
    delta_softplus: Int8,
    # Tensors - varlen format: (dim, total_length) for u, delta, z, out
    u: TileTensor[kernel_dtype, u_LT, MutUntrackedOrigin, Engine=Engine],
    delta: TileTensor[
        kernel_dtype, delta_LT, MutUntrackedOrigin, Engine=Engine
    ],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin, Engine=Engine],
    B: TileTensor[
        kernel_dtype, B_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (ngroups, dstate, total_length)
    C: TileTensor[
        kernel_dtype, C_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (ngroups, dstate, total_length)
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin, Engine=Engine],
    z: TileTensor[kernel_dtype, z_LT, MutUntrackedOrigin, Engine=Engine],
    delta_bias: TileTensor[
        kernel_dtype, delta_bias_LT, MutUntrackedOrigin, Engine=Engine
    ],
    ssm_states: TileTensor[
        kernel_dtype, ssm_states_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch, dim, dstate)
    output: TileTensor[
        kernel_dtype, output_LT, MutUntrackedOrigin, Engine=Engine
    ],  # Output written here (or to z if z is present)
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch + 1,)
    cache_indices: TileTensor[
        .int32, cache_indices_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch,)
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch,)
    u_strides: Strides2D,  # (dim, total_length)
    delta_strides: Strides2D,  # (dim, total_length)
    A_strides: Strides2D,  # (dim, dstate)
    B_strides: Strides3D,  # (ngroups, dstate, total_length)
    C_strides: Strides3D,  # (ngroups, dstate, total_length)
    D_strides: Strides1D,  # (dim,)
    z_strides: Strides2D,  # (dim, total_length)
    delta_bias_strides: Strides1D,  # (dim,)
    ssm_states_strides: Strides3D,  # (batch, dim, dstate)
    out_strides: Strides2D,  # (dim, total_length)
):
    """GPU kernel for variable-length selective scan."""
    var _dim = Int(dim)
    var _ngroups = Int(ngroups)
    var _batch = Int(batch)
    # 2D grid: block_idx.x for _dim, block_idx.y for _batch
    var d = block_dim.x * block_idx.x + thread_idx.x
    var b = block_idx.y

    if d >= _dim or b >= _batch:
        return

    var has_D = Int(D.dim[0]()) > 0
    var has_z = Int(z.dim[0]()) > 0
    var has_delta_bias = Int(delta_bias.dim[0]()) > 0
    var has_cache_indices = Int(cache_indices.dim[0]()) > 0
    var has_initial_state_tensor = Int(has_initial_state.dim[0]()) > 0
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)

    # Get sequence start and length
    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    # Get cache index for this sequence
    var cache_idx = b
    if has_cache_indices:
        cache_idx = Int(cache_indices.raw_load(b))
        if cache_idx == Int(pad_slot_id):
            return

    # Pre-load D and delta_bias for this _dim
    var D_val = Float32(0.0)
    if has_D:
        var D_offset = UInt32(d * D_strides[0])
        D_val = Scalar[kernel_dtype](D.raw_load(D_offset)).cast[.float32]()

    var delta_bias_val = Float32(0.0)
    if has_delta_bias:
        var bias_offset = UInt32(d * delta_bias_strides[0])
        delta_bias_val = Scalar[kernel_dtype](
            delta_bias.raw_load(bias_offset)
        ).cast[.float32]()

    # Pre-load A values for this _dim and pre-multiply by LOG2E for faster exp2
    var A_vals = SIMD[.float32, MAX_DSTATE](0.0)

    comptime for n in range(DSTATE):
        var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
        A_vals[n] = (
            Scalar[kernel_dtype](A.raw_load(A_offset)).cast[.float32]() * LOG2E
        )

    # Determine group for this _dim
    var group_size = _dim // _ngroups
    var group_id = d // group_size

    # Initialize state - either from cache or zeros
    var state = SIMD[.float32, MAX_DSTATE](0.0)

    # Load initial state if requested
    var use_initial_state = False
    if has_initial_state_tensor:
        var init_state_val = has_initial_state.raw_load(b)
        use_initial_state = Bool(init_state_val)

    if use_initial_state:
        comptime for n in range(DSTATE):
            var state_offset = UInt32(
                cache_idx * ssm_states_strides[0]
                + d * ssm_states_strides[1]
                + n * ssm_states_strides[2]
            )
            state[n] = Scalar[kernel_dtype](
                ssm_states.raw_load(state_offset)
            ).cast[.float32]()

    # Process sequence
    for t in range(seq_len):
        var global_t = seq_start + t

        # Load u value
        var u_offset = UInt32(d * u_strides[0] + global_t * u_strides[1])
        var u_val = Scalar[kernel_dtype](u.raw_load(u_offset)).cast[
            DType.float32
        ]()

        # Load delta value
        var delta_offset = UInt32(
            d * delta_strides[0] + global_t * delta_strides[1]
        )
        var delta_val = Scalar[kernel_dtype](delta.raw_load(delta_offset)).cast[
            DType.float32
        ]()

        # Apply delta_bias
        if has_delta_bias:
            delta_val += delta_bias_val

        # Apply softplus
        if delta_softplus_bool:
            delta_val = softplus(delta_val)

        var delta_u = delta_val * u_val

        # Load B and C values for this timestep
        var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var B_offset = UInt32(
                group_id * B_strides[0]
                + n * B_strides[1]
                + global_t * B_strides[2]
            )
            var C_offset = UInt32(
                group_id * C_strides[0]
                + n * C_strides[1]
                + global_t * C_strides[2]
            )

            B_vals[n] = Scalar[kernel_dtype](B.raw_load(B_offset)).cast[
                DType.float32
            ]()
            C_vals[n] = Scalar[kernel_dtype](C.raw_load(C_offset)).cast[
                DType.float32
            ]()

        # SSM step: state = state * exp2(A * LOG2E * delta) + B * delta * u
        var a_t = exp2(A_vals * delta_val)
        var b_t = B_vals * delta_u
        state = state * a_t + b_t

        # Compute output: y = sum(state * C) - use SIMD reduce
        var output_val = (state * C_vals).reduce_add()

        # Add D * u if D is present
        if has_D:
            output_val += D_val * u_val

        # Apply gating with z if present, using optimized silu
        if has_z:
            var z_offset = UInt32(d * z_strides[0] + global_t * z_strides[1])
            var z_val = Scalar[kernel_dtype](z.raw_load(z_offset)).cast[
                DType.float32
            ]()
            output_val *= silu(z_val)

            # Write to z if z is present (vLLM convention: output written to z)
            z.raw_store(
                z_offset, Scalar[kernel_dtype](output_val.cast[kernel_dtype]())
            )
        else:
            # Write to output (or delta in vLLM convention)
            var out_offset = UInt32(
                d * out_strides[0] + global_t * out_strides[1]
            )
            output.raw_store(
                out_offset,
                Scalar[kernel_dtype](output_val.cast[kernel_dtype]()),
            )

    # Store final state to cache
    comptime for n in range(DSTATE):
        var state_offset = UInt32(
            cache_idx * ssm_states_strides[0]
            + d * ssm_states_strides[1]
            + n * ssm_states_strides[2]
        )
        ssm_states.raw_store(
            state_offset, Scalar[kernel_dtype](state[n].cast[kernel_dtype]())
        )


def varlen_selective_state_update_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
](
    batch: Int,
    nheads: Int,
    dim: Int,
    nheads_ngroups_ratio: Int,
    pad_slot_id: Int32,
    dt_softplus: Int8,
    has_state_batch_indices: Int8,
    # Tensors
    state: TileTensor[mut=True, kernel_dtype, ...],
    x: TileTensor[mut=False, kernel_dtype, ...],
    dt: TileTensor[mut=False, kernel_dtype, ...],
    A: TileTensor[mut=False, kernel_dtype, ...],
    B: TileTensor[mut=False, kernel_dtype, ...],
    C: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    z: TileTensor[mut=False, kernel_dtype, ...],
    output: TileTensor[mut=True, kernel_dtype, ...],
    dt_bias: TileTensor[mut=False, kernel_dtype, ...],
    state_batch_indices: TileTensor[mut=False, .int32, ...],
    # All strides (same as GPU version)
    state_strides: Strides4D,
    x_strides: Strides3D,
    dt_strides: Strides3D,
    dt_bias_strides: Strides2D,
    A_strides: Strides3D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides2D,
    z_strides: Strides3D,
    out_strides: Strides3D,
    ctx: Optional[DeviceContext] = None,
):
    """CPU kernel for varlen selective state update."""
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_D = Int(D.dim[0]()) > 0
    var has_z = Int(z.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)
    var has_state_batch_indices_bool = Bool(Int(has_state_batch_indices) != 0)

    def worker(idx: Int) {imm}:
        var b, remaining = divmod(idx, nheads * dim)
        var h, m = divmod(remaining, dim)

        # Determine state batch index
        var state_batch_idx = Int32(b)
        if has_state_batch_indices_bool:
            state_batch_idx = state_batch_indices.raw_load(b)
            if state_batch_idx == pad_slot_id:
                return

        var group_id = h // nheads_ngroups_ratio

        # Load x value
        var x_offset = UInt32(
            b * x_strides[0] + h * x_strides[1] + m * x_strides[2]
        )
        var x_val = Scalar[kernel_dtype](x.raw_load(x_offset)).cast[
            DType.float32
        ]()

        # Load dt value
        var dt_offset = UInt32(
            b * dt_strides[0] + h * dt_strides[1] + m * dt_strides[2]
        )
        var dt_val = Scalar[kernel_dtype](dt.raw_load(dt_offset)).cast[
            DType.float32
        ]()

        # Apply dt_bias if present
        if has_dt_bias:
            var dt_bias_offset = UInt32(
                h * dt_bias_strides[0] + m * dt_bias_strides[1]
            )
            var bias_val = Scalar[kernel_dtype](
                dt_bias.raw_load(dt_bias_offset)
            ).cast[.float32]()
            dt_val += bias_val

        # Apply softplus if requested
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var out_val = Float32(0.0)

        # Process each dstate element
        comptime for n in range(DSTATE):
            # Load A value
            var A_offset = UInt32(
                h * A_strides[0] + m * A_strides[1] + n * A_strides[2]
            )
            var A_val = Scalar[kernel_dtype](A.raw_load(A_offset)).cast[
                DType.float32
            ]()

            # Compute dA = exp(A * dt) using exp2 for consistency
            var dA = exp2(A_val * LOG2E * dt_val)

            # Load B value
            var B_offset = UInt32(
                b * B_strides[0] + group_id * B_strides[1] + n * B_strides[2]
            )
            var B_val = Scalar[kernel_dtype](B.raw_load(B_offset)).cast[
                DType.float32
            ]()

            # Compute dB = B * dt
            var dB = B_val * dt_val

            # Load current state
            var state_offset = UInt32(
                Int(state_batch_idx) * state_strides[0]
                + h * state_strides[1]
                + m * state_strides[2]
                + n * state_strides[3]
            )
            var state_val = Scalar[kernel_dtype](
                state.raw_load(state_offset)
            ).cast[.float32]()

            # Update state
            state_val = state_val * dA + dB * x_val

            # Store updated state
            state.raw_store(
                state_offset,
                Scalar[kernel_dtype](state_val.cast[kernel_dtype]()),
            )

            # Load C value
            var C_offset = UInt32(
                b * C_strides[0] + group_id * C_strides[1] + n * C_strides[2]
            )
            var C_val = Scalar[kernel_dtype](C.raw_load(C_offset)).cast[
                DType.float32
            ]()

            # Accumulate output
            out_val += state_val * C_val

        # Add skip connection if D is present
        if has_D:
            var D_offset = UInt32(h * D_strides[0] + m * D_strides[1])
            var D_val = Scalar[kernel_dtype](D.raw_load(D_offset)).cast[
                DType.float32
            ]()
            out_val += x_val * D_val

        # Apply gating if z is present, using optimized silu
        if has_z:
            var z_offset = UInt32(
                b * z_strides[0] + h * z_strides[1] + m * z_strides[2]
            )
            var z_val = Scalar[kernel_dtype](z.raw_load(z_offset)).cast[
                DType.float32
            ]()
            out_val *= silu(z_val)

        # Store output
        var out_offset = UInt32(
            b * out_strides[0] + h * out_strides[1] + m * out_strides[2]
        )
        output.raw_store(
            out_offset, Scalar[kernel_dtype](out_val.cast[kernel_dtype]())
        )

    sync_parallelize(worker, batch * nheads * dim, ctx)


def varlen_selective_scan_fwd_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
](
    dim: Int,
    ngroups: Int,
    batch: Int,
    pad_slot_id: Int32,
    delta_softplus: Int8,
    # Tensors
    u: TileTensor[mut=False, kernel_dtype, ...],
    delta: TileTensor[mut=False, kernel_dtype, ...],
    A: TileTensor[mut=False, kernel_dtype, ...],
    B: TileTensor[mut=False, kernel_dtype, ...],
    C: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    z: TileTensor[mut=True, kernel_dtype, ...],
    delta_bias: TileTensor[mut=False, kernel_dtype, ...],
    ssm_states: TileTensor[mut=True, kernel_dtype, ...],
    output: TileTensor[mut=True, kernel_dtype, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    cache_indices: TileTensor[mut=False, .int32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    # Strides (same as GPU version)
    u_strides: Strides2D,
    delta_strides: Strides2D,
    A_strides: Strides2D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    z_strides: Strides2D,
    delta_bias_strides: Strides1D,
    ssm_states_strides: Strides3D,
    out_strides: Strides2D,
    ctx: Optional[DeviceContext] = None,
):
    """CPU kernel for variable-length selective scan."""
    var has_D = Int(D.dim[0]()) > 0
    var has_z = Int(z.dim[0]()) > 0
    var has_delta_bias = Int(delta_bias.dim[0]()) > 0
    var has_cache_indices = Int(cache_indices.dim[0]()) > 0
    var has_initial_state_tensor = Int(has_initial_state.dim[0]()) > 0
    var delta_softplus_bool = Bool(Int(delta_softplus) != 0)
    var group_size = dim // ngroups

    def worker(d: Int) {imm}:
        # Pre-load D and delta_bias for this dim
        var D_val = Float32(0.0)
        if has_D:
            var D_offset = UInt32(d * D_strides[0])
            D_val = Scalar[kernel_dtype](D.raw_load(D_offset)).cast[
                DType.float32
            ]()

        var delta_bias_val = Float32(0.0)
        if has_delta_bias:
            var bias_offset = UInt32(d * delta_bias_strides[0])
            delta_bias_val = Scalar[kernel_dtype](
                delta_bias.raw_load(bias_offset)
            ).cast[.float32]()

        # Pre-load A values for this dim and pre-multiply by LOG2E for faster exp2
        var A_vals = SIMD[.float32, MAX_DSTATE](0.0)

        comptime for n in range(DSTATE):
            var A_offset = UInt32(d * A_strides[0] + n * A_strides[1])
            A_vals[n] = (
                Scalar[kernel_dtype](A.raw_load(A_offset)).cast[.float32]()
                * LOG2E
            )

        var group_id = d // group_size

        # Process each sequence
        for b in range(batch):
            var seq_start = Int(query_start_loc.raw_load(b))
            var seq_end = Int(query_start_loc.raw_load(b + 1))
            var seq_len = seq_end - seq_start

            if seq_len <= 0:
                continue

            var cache_idx = b
            if has_cache_indices:
                cache_idx = Int(cache_indices.raw_load(b))
                if cache_idx == Int(pad_slot_id):
                    continue

            # Initialize state
            var state = SIMD[.float32, MAX_DSTATE](0.0)

            var use_initial_state = False
            if has_initial_state_tensor:
                var init_state_val = has_initial_state.raw_load(b)
                use_initial_state = Bool(init_state_val)

            if use_initial_state:
                comptime for n in range(DSTATE):
                    var state_offset = UInt32(
                        cache_idx * ssm_states_strides[0]
                        + d * ssm_states_strides[1]
                        + n * ssm_states_strides[2]
                    )
                    state[n] = Scalar[kernel_dtype](
                        ssm_states.raw_load(state_offset)
                    ).cast[.float32]()

            # Process sequence
            for t in range(seq_len):
                var global_t = seq_start + t

                var u_offset = UInt32(
                    d * u_strides[0] + global_t * u_strides[1]
                )
                var u_val = Scalar[kernel_dtype](u.raw_load(u_offset)).cast[
                    DType.float32
                ]()

                var delta_offset = UInt32(
                    d * delta_strides[0] + global_t * delta_strides[1]
                )
                var out_offset = UInt32(
                    d * out_strides[0] + global_t * out_strides[1]
                )
                var delta_val = Scalar[kernel_dtype](
                    delta.raw_load(delta_offset)
                ).cast[.float32]()

                if has_delta_bias:
                    delta_val += delta_bias_val

                if delta_softplus_bool:
                    delta_val = softplus(delta_val)

                var delta_u = delta_val * u_val

                var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[.float32, MAX_DSTATE](0.0)

                comptime for n in range(DSTATE):
                    var B_offset = UInt32(
                        group_id * B_strides[0]
                        + n * B_strides[1]
                        + global_t * B_strides[2]
                    )
                    var C_offset = UInt32(
                        group_id * C_strides[0]
                        + n * C_strides[1]
                        + global_t * C_strides[2]
                    )

                    B_vals[n] = Scalar[kernel_dtype](B.raw_load(B_offset)).cast[
                        DType.float32
                    ]()
                    C_vals[n] = Scalar[kernel_dtype](C.raw_load(C_offset)).cast[
                        DType.float32
                    ]()

                # SSM step using SIMD exp2 with pre-multiplied LOG2E
                var a_t = exp2(A_vals * delta_val)
                var b_t = B_vals * delta_u
                state = state * a_t + b_t

                # Compute output using SIMD reduce
                var output_val = (state * C_vals).reduce_add()

                if has_D:
                    output_val += D_val * u_val

                if has_z:
                    var z_offset = UInt32(
                        d * z_strides[0] + global_t * z_strides[1]
                    )
                    var z_val = Scalar[kernel_dtype](z.raw_load(z_offset)).cast[
                        DType.float32
                    ]()
                    output_val *= silu(z_val)
                    z.raw_store(
                        z_offset,
                        Scalar[kernel_dtype](output_val.cast[kernel_dtype]()),
                    )
                else:
                    output.raw_store(
                        out_offset,
                        Scalar[kernel_dtype](output_val.cast[kernel_dtype]()),
                    )

            # Store final state
            comptime for n in range(DSTATE):
                var state_offset = UInt32(
                    cache_idx * ssm_states_strides[0]
                    + d * ssm_states_strides[1]
                    + n * ssm_states_strides[2]
                )
                ssm_states.raw_store(
                    state_offset,
                    Scalar[kernel_dtype](state[n].cast[kernel_dtype]()),
                )

    sync_parallelize(worker, dim, ctx)
