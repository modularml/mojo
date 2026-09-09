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
"""Implements KV-cache kernels for ragged (variable-length) sequences used in continuous batching."""

from std.sys.info import (
    _current_target,
    has_amd_gpu_accelerator,
    simd_width_of,
)
from std.math import ceildiv
from std.math.uutils import udivmod
from std.memory import ThinAllocation, dealloc
from std.memory.alloc import Layout as AllocLayout

from std.algorithm.functional import unswitch

from max.algorithm.functional import elementwise
from max.gpu import global_idx
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu.primitives.grid_controls import PDLLevel
from std.collections import Optional, OptionalReg
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    Layout,
    LayoutTensor,
    RowMajorLayout,
    RuntimeLayout,
    TileTensor,
    TensorLayout,
    UNKNOWN_VALUE,
    coord_to_index_list,
    lt_to_tt,
    row_major,
)
from linalg.matmul import elementwise_epilogue_type, matmul
from linalg.fp8_quantization import blockwise_scaled_fp8_with_epilogue
from linalg.block_scaled_quantization import block_scaled_matmul
from linalg.matmul.gpu.amd import block_scaled_matmul_amd
from internal_utils.fp8_utils import cast_saturating
from nn._ragged_utils import get_batch_from_row_offsets
from nn.attention.cpu.mha import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.attention.gpu.mha import (
    MHADecodeDispatchMetadata,
    flash_attention as gpu_flash_attention,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_utils import (
    MHAConfig,
    as_dynamic_row_major_1d,
    dispatch_mask,
    dispatch_relative_logits_mask,
)
from nn.attention.gpu.mla import (
    _k_cache_to_buffer,
    flare_mla_decoding,
    flare_mla_prefill,
    mla_prefill_plan,
)
from quantization.qmatmul import matmul_qint4
from quantization.qmatmul_gpu import matmul_gpu_qint4_impl
from quantization.qmatmul_k import matmul_Q4_K, matmul_Q6_K

from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# Fused QKV matmul (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged[
    dtype: DType,
    weight_dtype: DType,
    target: StaticString = "cpu",
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Parameters:
        dtype: Element type of the `hidden_state` input and `output` tensors.
        weight_dtype: Element type of the `weight` tensor.
        target: Compilation target string used to dispatch GPU versus CPU
            paths (defaults to "cpu").
        group_size: Block size for GPTQ-style quantization of `weight`; when
            set, `weight` must be `uint8` (defaults to `None` for no
            quantization).
        has_zp: Whether the weight quantization uses a zero point; currently
            must be falsy when `group_size` is set (defaults to `None`).

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged[
            kv_collection.CacheType,
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
    dtype: DType,
    weight_dtype: DType,
    target: StaticString = "cpu",
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    bias: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Parameters:
        dtype: Element type of the `hidden_state` input and `output` tensors.
        weight_dtype: Element type of the `weight` tensor.
        target: Compilation target string used to dispatch GPU versus CPU
            paths (defaults to "cpu").
        group_size: Block size for GPTQ-style quantization of `weight`; when
            set, `weight` must be `uint8` (defaults to `None` for no
            quantization).
        has_zp: Whether the weight quantization uses a zero point; currently
            must be falsy when `group_size` is set (defaults to `None`).

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        bias: Bias to be added to the QKV Tensor. Tensor is concatenated q + k + v. Rank 1.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.bias.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_bias[
            kv_collection.CacheType,
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    scales_granularity_mnk: IndexList[3],
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    bias: OptionalReg[
        LayoutTensor[
            output_dtype,
            Layout.row_major(UNKNOWN_VALUE),
            ImmutAnyOrigin,
            address_space=.GENERIC,
        ]
    ] = None,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Parameters:
        dtype: Element type of the `hidden_state` input tensor.
        weight_dtype: Element type of the `weight` tensor.
        output_dtype: Element type of the `output` tensor and of the Q
            projection written to it.
        scale_dtype: Element type of the `input_scale` and `weight_scale`
            tensors.
        scales_granularity_mnk: Block sizes along the M, N, and K matmul
            dimensions used to tile the scale application; `-1` selects
            per-tensor scaling, `1` selects per-channel scaling, and any
            other value selects blockwise scaling.
        target: Compilation target string used to dispatch GPU versus CPU
            paths (defaults to "cpu").

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from
            kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
        bias: Optional bias vector concatenated as [q, k, v].
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "input_scale", input_scale.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "weight_scale", weight_scale.runtime_layout.shape.value
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.scale.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(
        kv_collection.kv_params.head_size
    ) + ".m_scale_granularity_" + String(
        scales_granularity_mnk[0]
    ) + ".n_scale_granularity_" + String(
        scales_granularity_mnk[1]
    ) + ".k_scale_granularity_" + String(
        scales_granularity_mnk[2]
    )
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_scale[
            kv_collection.CacheType,
            scales_granularity_mnk=scales_granularity_mnk,
            target=target,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            kv_collection,
            layer_idx,
            output,
            ctx,
            bias,
        )


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, b_layout, MutAnyOrigin],
    input_scale: LayoutTensor[mut=False, scale_dtype, sfa_layout, MutAnyOrigin],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, sfb_layout, MutAnyOrigin
    ],
    tensor_sf: Float32,
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Parameters:
        dtype: Element type of the `hidden_state` input tensor.
        weight_dtype: Element type of the `weight` tensor.
        output_dtype: Element type of the `output` tensor and of the Q
            projection written to it.
        scale_dtype: Element type of the `input_scale` and `weight_scale`
            tensors.
        a_layout: Memory layout of the `hidden_state` input tensor.
        b_layout: Memory layout of the `weight` tensor.
        sfa_layout: Memory layout of the `input_scale` tensor.
        sfb_layout: Memory layout of the `weight_scale` tensor.
        SF_VECTOR_SIZE: Number of scale elements packed per scaling-factor
            vector; `32` for MXFP8 E8M0 scaling and `16` for NVFP4 scaling.
        target: Compilation target string used to dispatch GPU versus CPU
            paths (defaults to "cpu").

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size // 2).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size // 2).
        input_scale: 5D blockwise scale tensor to be multiplied to the input Tensor.
        weight_scale: 5D blockwise scale tensor to the weight Tensor.
        tensor_sf: Per-tensor scaling factor.
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from
            kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "input_scale", input_scale.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "weight_scale", weight_scale.runtime_layout.shape.value
                    ),
                    "tensor_sf=" + String(tensor_sf),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.scale.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_scale_float4[
            kv_collection.CacheType,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            tensor_sf,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged[
    dtype: DType,
    weight_dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = None
    comptime kv_params = collection_t.kv_params

    comptime if not kv_params.is_mla:
        v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime if is_gpu[target]():
        cuda_ctx = context

    return _fused_qkv_matmul_kv_cache_ragged_impl[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
    )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_bias[
    dtype: DType,
    weight_dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    bias: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        bias: Bias to be added to the QKV Tensor. Tensor is concatenated q + k + v. Rank 1.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime if is_gpu[target]():
        cuda_ctx = context

    return _fused_qkv_matmul_kv_cache_ragged_impl_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        output,
        bias,
        cuda_ctx,
    )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    scales_granularity_mnk: IndexList[3],
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    bias: OptionalReg[
        LayoutTensor[
            output_dtype,
            Layout.row_major(UNKNOWN_VALUE),
            ImmutAnyOrigin,
            address_space=.GENERIC,
        ]
    ] = None,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads *
            head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object
            from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
        bias: Optional bias vector concatenated as [q, k, v].
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = None
    comptime kv_params = collection_t.kv_params

    comptime if not kv_params.is_mla:
        v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime if is_gpu[target]():
        cuda_ctx = context

    return _fused_qkv_matmul_kv_cache_ragged_impl_scale[
        scales_granularity_mnk=scales_granularity_mnk, target=target
    ](
        hidden_state,
        input_row_offsets,
        weight,
        input_scale,
        weight_scale,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
        bias,
    )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    SF_VECTOR_SIZE: Int,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, b_layout, MutAnyOrigin],
    input_scale: LayoutTensor[mut=False, scale_dtype, sfa_layout, MutAnyOrigin],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, sfb_layout, MutAnyOrigin
    ],
    tensor_sf: Float32,
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads *
            head_size // 2).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size // 2).
        input_scale: 5D blockwise scale tensor to be multiplied to the input Tensor.
        weight_scale: 5D blockwise scale tensor to the weight Tensor.
        tensor_sf: Per-tensor scaling factor.
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object
            from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = None
    comptime kv_params = collection_t.kv_params

    comptime if not kv_params.is_mla:
        v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime if is_gpu[target]():
        cuda_ctx = context

    return _fused_qkv_matmul_kv_cache_ragged_impl_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, target=target
    ](
        hidden_state,
        input_row_offsets,
        weight,
        input_scale,
        weight_scale,
        tensor_sf,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
    )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_impl[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 * num_kv_heads) * head_size).
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    comptime assert (
        kv_type == dtype or kv_type == DType.float8_e4m3fn
    ), "KV cache must be the Q dtype or e4m3"

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @__parameter
    @__copy_capture(q_dim, qk_offset, batch_size)
    @always_inline
    def write_to_cache[
        _dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                rebind[SIMD[dtype, width]](val),
            )
            return

        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx: Int
        var hd_idx: Int
        var cache: cache_t
        var output_val = val

        comptime if kv_params.is_mla:
            cache = k_cache
            h_idx = 0  # in MLA mode we only have one head
            hd_idx = idx[1] - q_dim

        else:
            if idx[1] < qk_offset:
                cache = k_cache
                h_idx, hd_idx = udivmod(idx[1] - q_dim, kv_params.head_size)
            else:
                cache = v_cache.value()
                h_idx, hd_idx = udivmod(idx[1] - qk_offset, kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](cast_saturating[kv_type](output_val)),
        )

    comptime if group_size:
        comptime assert (
            not has_zp.value()
        ), "Zero point is not supported for quantization."
        comptime assert (
            weight_dtype == .uint8
        ), "Expect GPTQ weights in an uint8 tensor."

        _qmatmul_common[
            group_size=group_size.value(),
            target=target,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, weight.bitcast[.uint8](), context)

    else:
        comptime assert (
            weight_dtype == dtype
        ), "Mismatch in dtype between weight and QKV tensors"

        _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
            hidden_state, weight.bitcast[dtype](), context
        )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_impl_bias[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    k_cache: cache_t,
    v_cache: cache_t,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    bias: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 * num_kv_heads) * head_size).
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        bias: Bias to be added to the QKV Tensor. Tensor is concatenated q + k + v. Rank 1.
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    comptime assert (
        kv_type == dtype or kv_type == DType.float8_e4m3fn
    ), "KV cache must be the Q dtype or e4m3"

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @__parameter
    @__copy_capture(q_dim, qk_offset, batch_size)
    @always_inline
    def write_to_cache[
        _dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        var output_val = val + rebind[SIMD[_dtype, width]](
            bias.load[width=width](IndexList[1](idx[1]))
        )
        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                rebind[SIMD[dtype, width]](output_val),
            )
            return

        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx: Int
        var hd_idx: Int
        var cache: cache_t
        if idx[1] < qk_offset:
            cache = k_cache
            h_idx, hd_idx = udivmod(idx[1] - q_dim, kv_params.head_size)
        else:
            cache = v_cache
            h_idx, hd_idx = udivmod(idx[1] - qk_offset, kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](cast_saturating[kv_type](output_val)),
        )

    comptime if group_size:
        comptime assert (
            not has_zp.value()
        ), "Zero point is not supported for quantization."
        comptime assert (
            weight_dtype == .uint8
        ), "Expect GPTQ weights to be a 'uint8' tensor."

        _qmatmul_common[
            group_size=group_size.value(),
            target=target,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, weight.bitcast[.uint8](), context)

    else:
        comptime assert (
            weight_dtype == dtype
        ), "Mismatch in dtype between weight and QKV tensors"

        _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
            hidden_state, weight.bitcast[dtype](), context
        )


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_impl_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    cache_t: KVCacheT,
    //,
    scales_granularity_mnk: IndexList[3],
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
    bias: OptionalReg[
        LayoutTensor[
            mut=False,
            output_dtype,
            Layout.row_major(UNKNOWN_VALUE),
            MutAnyOrigin,
            address_space=.GENERIC,
        ]
    ] = None,
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 *
            num_kv_heads) * head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if is_cpu[target]().
        bias: Optional bias vector concatenated as [q, k, v].
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    # Here we decide the quantization scheme for the QKV Tensor.
    comptime use_per_tensor = (
        scales_granularity_mnk[0] == -1
        and scales_granularity_mnk[1] == -1
        and scales_granularity_mnk[2] == -1
    )
    comptime use_per_channel = (
        scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == 1
        and scales_granularity_mnk[2] == -1
    )
    comptime use_block_wise = not (use_per_tensor or use_per_channel)

    @__parameter
    @__copy_capture(
        input_scale, weight_scale, q_dim, qk_offset, batch_size, bias
    )
    @always_inline
    def write_to_cache[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        var output_val: SIMD[dtype, width]

        comptime if use_per_tensor:
            var scale_a = input_scale[0, 0][0].cast[dtype]()
            var scale_b = weight_scale[0, 0][0].cast[dtype]()
            output_val = val * (scale_a * scale_b)
        elif use_per_channel:
            var scale_a = input_scale.load[width=1](0, idx[0]).cast[dtype]()
            var scale_b = weight_scale.load[width=width](idx[1], 0).cast[
                dtype
            ]()
            output_val = val * (scale_a * scale_b)
        else:
            # blockwise quantization, we need to use the blockwise_scaled_fp8_with_epilogue kernel
            output_val = val

        var output_val_out: SIMD[output_dtype, width] = rebind[
            SIMD[output_dtype, width]
        ](output_val.cast[output_dtype]())

        if bias:
            output_val_out += bias.value().load[width=width](
                IndexList[1](idx[1])
            )

        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                output_val_out,
            )
            return

        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx: Int
        var hd_idx: Int
        var cache: cache_t

        comptime if kv_params.is_mla:
            cache = k_cache
            h_idx = 0  # in MLA mode we only have one head
            hd_idx = idx[1] - q_dim

        else:
            if idx[1] < qk_offset:
                cache = k_cache
                h_idx, hd_idx = udivmod(idx[1] - q_dim, kv_params.head_size)
            else:
                cache = v_cache.value()
                h_idx, hd_idx = udivmod(idx[1] - qk_offset, kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](output_val_out.cast[kv_type]()),
        )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"

    comptime if use_block_wise:
        comptime assert is_gpu[
            target
        ](), "Blockwise scaled fp8 matmul only works on GPU."

        _matmul_blockwise_scaled_fp8_common[
            output_dtype=cache_t.dtype,
            target=target,
            scales_granularity_mnk=scales_granularity_mnk,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, weight, input_scale, weight_scale, context.value())
    else:
        _matmul_common[
            target=target,
            elementwise_lambda_fn=write_to_cache,
            output_dtype=output_dtype,
        ](hidden_state, weight.bitcast[dtype](), context)


@always_inline
def _fused_qkv_matmul_kv_cache_ragged_impl_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    cache_t: KVCacheT,
    //,
    SF_VECTOR_SIZE: Int,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[dtype, a_layout, ImmutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[weight_dtype, b_layout, ImmutAnyOrigin],
    input_scale: LayoutTensor[scale_dtype, sfa_layout, ImmutAnyOrigin],
    weight_scale: LayoutTensor[scale_dtype, sfb_layout, ImmutAnyOrigin],
    tensor_sf: Float32,
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads *
            head_size // 2).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size // 2).
        input_scale: 5D blockwise scale tensor to be multiplied to the input Tensor.
        weight_scale: 5D blockwise scale tensor to the weight Tensor.
        tensor_sf: Per-tensor scaling factor.
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @__parameter
    @__copy_capture(input_scale, weight_scale, q_dim, qk_offset, batch_size)
    @always_inline
    def write_to_cache[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        # Blockwise-scaled matmul epilogue: scatter Q to `output` and K/V into
        # the paged cache. Runs as the `elementwise_lambda_fn` of the
        # block-scaled matmul (Mojo kernel when covered, vendor otherwise).
        var output_val_out: SIMD[output_dtype, width] = rebind[
            SIMD[output_dtype, width]
        ](val.cast[output_dtype]())

        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                output_val_out,
            )
            return

        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx: Int
        var hd_idx: Int
        var cache: cache_t

        comptime if kv_params.is_mla:
            cache = k_cache
            h_idx = 0  # in MLA mode we only have one head
            hd_idx = idx[1] - q_dim

        else:
            if idx[1] < qk_offset:
                cache = k_cache
                h_idx, hd_idx = udivmod(idx[1] - q_dim, kv_params.head_size)
            else:
                cache = v_cache.value()
                h_idx, hd_idx = udivmod(idx[1] - qk_offset, kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](cast_saturating[kv_type](val)),
        )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"

    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp4 matmul only works on GPU."

    _matmul_blockwise_scaled_fp4_common[
        output_dtype=output_dtype,
        target=target,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        elementwise_lambda_fn=write_to_cache,
    ](
        hidden_state,
        weight,
        input_scale,
        weight_scale,
        tensor_sf,
        context.value(),
    )


# ===-----------------------------------------------------------------------===#
# Dual-cache fused QKV + index projection (ragged, paged, MXFP8/NVFP4)
# ===-----------------------------------------------------------------------===#
#
# SM100 (B200) / generic-GPU MXFP8 (E8M0, SF_VECTOR_SIZE=32) or NVFP4
# (float8_e4m3fn scales, SF_VECTOR_SIZE=16) block-scaled GEMM that fuses
# MiniMax-M3's five projections (Q, K, V, IndexQ, IndexK) into ONE matmul over
# the concatenated weight W = [Wq | Wk | Wv | Wiq | Wik] (concat along N), then
# routes the matmul output columns in a single elementwise epilogue to:
#   - two separate output buffers: `q_output` [M, q_dim] (Q) and `iq_output`
#     [M, iq_dim] (IndexQ), and
#   - TWO paged KV cache collections: the MAIN cache (K + V) and the INDEX
#     cache (IndexK only, single shared KV head).
#
# This is the dual-cache generalization of
# `generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4`. The single-cache
# entry routes columns to {output | main-K | main-V}; this one adds two more
# column bands {IndexQ | IndexK} and a second collection. The fusion is
# bit-exact to running the QKV matmul and the IndexQK matmul separately ONLY
# when every band boundary lands on a quantization SF-atom row group
# (SF_VECTOR_SIZE divides each band width); for M3 all boundaries are multiples
# of 128, which holds. See `_can_fuse_index_qk` in
# minimax_m3/layers/sparse_indexer.py.
#
# Column routing over col in [0, N_total) with
#   N_total = q_dim + 2*kv_dim + iq_dim + ik_dim:
#   [0, q_dim)                       -> Q       : q_output[:, col]
#   [q_dim, q_dim+kv_dim)            -> K       : main k_cache (head/dim from col)
#   [q_dim+kv_dim, q_dim+2*kv_dim)   -> V       : main v_cache
#   [q_dim+2*kv_dim, +iq_dim)        -> IndexQ  : iq_output[:, col-base]
#   [+iq_dim, +ik_dim)               -> IndexK  : index k_cache (head=0, dim)
#
# The MAIN cache is GQA/MHA (non-MLA). The INDEX cache is MLA (single latent
# head, K only): M3 builds it with `is_mla=True`, `n_kv_heads=1`
# (model_config.py). The IndexK scatter therefore always uses head 0 and
# `hd_idx = col - <index band start>` — the MLA pattern, identical to the
# single-cache `write_to_cache` MLA branch. `iq_dim` (the IndexQ band width =
# num_index_heads * idx_head_dim) is passed explicitly because for an MLA index
# cache `index_kv_params.num_heads` (== 1) does NOT equal num_index_heads; only
# `ik_dim == index_kv_params.head_size` is derivable from the cache params.
def generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, b_layout, MutAnyOrigin],
    input_scale: LayoutTensor[mut=False, scale_dtype, sfa_layout, MutAnyOrigin],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, sfb_layout, MutAnyOrigin
    ],
    tensor_sf: Float32,
    kv_collection: PagedKVCacheCollection,
    index_kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    iq_dim: Int,
    q_output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    iq_output: LayoutTensor[
        mut=True, output_dtype, address_space=.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Performs a fused QKV + index-QK matmul. Q is written to `q_output` and
    IndexQ to `iq_output`; K/V are scattered into the MAIN `kv_collection` and
    IndexK into the INDEX `index_kv_collection`.

    Parameters:
        dtype: Element type of the `hidden_state` input tensor.
        weight_dtype: Element type of the `weight` tensor.
        output_dtype: Element type of the `output` tensor.
        scale_dtype: Element type of the `input_scale` and `weight_scale`
            tensors.
        a_layout: Memory layout of the `hidden_state` (matmul A operand) tensor.
        b_layout: Memory layout of the `weight` (matmul B operand) tensor.
        sfa_layout: Memory layout of the `input_scale` (scale-factor for A)
            tensor.
        sfb_layout: Memory layout of the `weight_scale` (scale-factor for B)
            tensor.
        SF_VECTOR_SIZE: Scale-factor vector size of the block-scaled format:
            32 for MXFP8 or 16 for NVFP4.
        target: Compilation target string used to dispatch GPU versus CPU
            paths (defaults to "cpu").

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), hidden).
        input_row_offsets: Tensor with shape (batch_size + 1,). The value at
            each index is the start_idx of the corresponding batch in
            hidden_state.
        weight: Concatenated weight W = [Wq | Wk | Wv | Wiq | Wik], shape
            (N_total, hidden) where N_total = q_dim + 2*kv_dim + iq_dim + ik_dim.
        input_scale: 5D blockwise scale tensor for the input.
        weight_scale: 5D blockwise scale tensor for the weight.
        tensor_sf: Per-tensor scaling factor.
        kv_collection: The MAIN KVCache collection (K, V) for this layer.
        index_kv_collection: The INDEX KVCache collection (IndexK only,
            single shared KV head) for this layer.
        layer_idx: The current layer, used to retrieve the KVCache objects.
        iq_dim: Width of the IndexQ output band (num_index_heads *
            idx_head_dim). Passed explicitly because for an MLA index cache it
            is not recoverable from the index cache's `num_heads`.
        q_output: The pre-allocated output buffer for the Q projection.
            Shape: (sum(seq_lens), q_dim).
        iq_output: The pre-allocated output buffer for the IndexQ projection.
            Shape: (sum(seq_lens), iq_dim).
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q_output", q_output.runtime_layout.shape.value),
                    trace_arg(
                        "iq_output", iq_output.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "tensor_sf=" + String(tensor_sf),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "idx_num_heads="
                    + String(index_kv_collection.kv_params.num_heads),
                    "idx_head_size="
                    + String(index_kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_index_matmul.ragged.paged.scale.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(
        kv_collection.kv_params.head_size
    )
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _fused_qkv_index_matmul_kv_cache_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            tensor_sf,
            kv_collection,
            index_kv_collection,
            layer_idx,
            iq_dim,
            q_output,
            iq_output,
            ctx,
        )


@always_inline
def _fused_qkv_index_matmul_kv_cache_ragged_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    collection_t: KVCollectionT,
    index_collection_t: KVCollectionT,
    //,
    SF_VECTOR_SIZE: Int,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, b_layout, MutAnyOrigin],
    input_scale: LayoutTensor[mut=False, scale_dtype, sfa_layout, MutAnyOrigin],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, sfb_layout, MutAnyOrigin
    ],
    tensor_sf: Float32,
    kv_collection: collection_t,
    index_kv_collection: index_collection_t,
    layer_idx: UInt32,
    iq_dim: Int,
    q_output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    iq_output: LayoutTensor[
        mut=True, output_dtype, address_space=.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    """Resolves both KVCache objects from their collections and dispatches the
    dual-cache impl. K/V come from the MAIN (non-MLA) collection, IndexK from
    the INDEX (MLA, single latent head) collection."""
    comptime kv_params = collection_t.kv_params

    # The MAIN cache is GQA/MHA — it has a real K and V side scattered per
    # head. The INDEX cache is MLA (K-only, single latent head); its scatter
    # uses head 0, so MLA is supported there (the assert below only constrains
    # the main cache). `get_value_cache` is therefore only taken on the main
    # collection, which is always non-MLA.
    comptime assert (
        not kv_params.is_mla
    ), "Dual-cache fused QKV+index: the MAIN (K/V) cache must be non-MLA."

    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)
    var index_k_cache = index_kv_collection.get_key_cache(layer_idx_cast)

    return _fused_qkv_index_matmul_kv_cache_ragged_impl_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, target=target
    ](
        hidden_state,
        input_row_offsets,
        weight,
        input_scale,
        weight_scale,
        tensor_sf,
        k_cache,
        v_cache,
        index_k_cache,
        iq_dim,
        q_output,
        iq_output,
        context,
    )


@always_inline
def _fused_qkv_index_matmul_kv_cache_ragged_impl_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    cache_t: KVCacheT,
    index_cache_t: KVCacheT,
    //,
    SF_VECTOR_SIZE: Int,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[dtype, a_layout, ImmutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[weight_dtype, b_layout, ImmutAnyOrigin],
    input_scale: LayoutTensor[scale_dtype, sfa_layout, ImmutAnyOrigin],
    weight_scale: LayoutTensor[scale_dtype, sfb_layout, ImmutAnyOrigin],
    tensor_sf: Float32,
    k_cache: cache_t,
    v_cache: cache_t,
    index_k_cache: index_cache_t,
    iq_dim: Int,
    q_output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    iq_output: LayoutTensor[
        mut=True, output_dtype, address_space=.GENERIC, ...
    ],
    context: Optional[DeviceContext],
) raises:
    """Dual-cache fused QKV + index matmul on ragged tensors.

    Q is written to `q_output` and IndexQ to `iq_output`; K/V are scattered
    into the main `k_cache`/`v_cache`; IndexK is scattered into the MLA
    `index_k_cache` (single latent head, head 0).
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params
    comptime index_kv_type = index_cache_t.dtype
    comptime index_kv_params = index_cache_t.kv_params

    # IndexK may be e4m3; saturating-cast from the GEMM output like main K/V.
    comptime assert (
        index_kv_type == output_dtype or index_kv_type == DType.float8_e4m3fn
    ), "the index-K cache must be the GEMM output dtype or e4m3"

    # Boundaries over the concatenated N dimension. Q lands in `q_output`
    # ([M, q_dim]) and IndexQ in `iq_output` ([M, iq_dim]):
    #   kv_dim = main num_heads * head_size   (K band == V band width)
    #   iq_dim = num_index_heads * idx_head_dim  (IndexQ band; PASSED IN, since
    #            for an MLA index cache index_kv_params.num_heads == 1 != it)
    #   ik_dim = index head_size              (IndexK band == single MLA latent
    #            head width; derivable from the index cache params)
    #   q_dim  = q_output width               (Q band)
    var kv_dim = kv_params.head_size * kv_params.num_heads
    var ik_dim = index_kv_params.head_size
    var q_dim = q_output.dim[1]()

    # Column band boundaries (exclusive upper bounds).
    var q_end = q_dim  # Q       : [0, q_end)
    var k_end = q_end + kv_dim  # K       : [q_end, k_end)
    var v_end = k_end + kv_dim  # V       : [k_end, v_end)
    var iq_end = v_end + iq_dim  # IndexQ  : [v_end, iq_end)
    var ik_end = iq_end + ik_dim  # IndexK  : [iq_end, ik_end)

    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    # The fusion is bit-exact to separate matmuls only when each band boundary
    # lands on the MXFP8/NVFP4 scale-block row group (SF_VECTOR_SIZE divides the
    # N-band widths). All band widths and offsets must be multiples of the SF
    # row group for the per-column scale lookup in the matmul to be identical to
    # the unfused matmuls. For M3 these are all multiples of 128.
    comptime SF_MN_GROUP = SF_VECTOR_SIZE
    if (
        q_dim % SF_MN_GROUP != 0
        or kv_dim % SF_MN_GROUP != 0
        or iq_dim % SF_MN_GROUP != 0
        or ik_dim % SF_MN_GROUP != 0
    ):
        raise Error(
            "Dual-cache fused QKV+index requires every output band width to be"
            " a multiple of the scale-block group size for bit-exact fusion."
        )

    @__parameter
    @__copy_capture(
        q_end,
        k_end,
        v_end,
        iq_end,
    )
    @always_inline
    def write_to_caches[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        # The block-scaled matmul already applied the scales; the epilogue just
        # routes and casts. `idx[1]` is the column in the concatenated N space.
        var output_val_out: SIMD[output_dtype, width] = rebind[
            SIMD[output_dtype, width]
        ](val.cast[output_dtype]())

        var col = idx[1]

        # Q band -> q_output[:, col].
        if col < q_end:
            q_output.store[width=width](idx, output_val_out)
            return

        # IndexQ band -> iq_output[:, col - v_end]. Checked before the cache
        # scatters so the branch order matches the column layout
        # (Q | K | V | IndexQ | IndexK).
        if col >= v_end and col < iq_end:
            iq_output.store[width=width](
                IndexList[2](idx[0], col - v_end),
                output_val_out,
            )
            return

        var global_token_idx = idx[0]
        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        # K / V bands -> main cache. IndexK band -> index cache.
        if col < v_end:
            var h_idx: Int
            var hd_idx: Int
            var cache: cache_t
            if col < k_end:
                cache = k_cache
                h_idx, hd_idx = udivmod(col - q_end, kv_params.head_size)
            else:
                cache = v_cache
                h_idx, hd_idx = udivmod(col - k_end, kv_params.head_size)
            var cache_length = cache.cache_length(batch_idx)
            var cache_token_idx = token_idx + cache_length
            cache.store(
                batch_idx,
                h_idx,
                cache_token_idx,
                hd_idx,
                rebind[SIMD[kv_type, width]](cast_saturating[kv_type](val)),
            )
        else:
            # IndexK band -> index cache, single shared head (head == 0).
            var hd_idx = col - iq_end
            var cache_length = index_k_cache.cache_length(batch_idx)
            var cache_token_idx = token_idx + cache_length
            index_k_cache.store(
                batch_idx,
                0,
                cache_token_idx,
                hd_idx,
                rebind[SIMD[index_kv_type, width]](
                    cast_saturating[index_kv_type](val)
                ),
            )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"

    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp4 matmul only works on GPU."

    _matmul_blockwise_scaled_fp4_common[
        output_dtype=output_dtype,
        target=target,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        elementwise_lambda_fn=write_to_caches,
    ](
        hidden_state,
        weight,
        input_scale,
        weight_scale,
        tensor_sf,
        context.value(),
    )


# BF16 (non-scaled) analog of
# `generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4`. Fuses the
# main QKV projection with the sparse-indexer QKV projection into a single GEMM
# over the stacked weight [Wq | Wk | Wv | Wiq | Wik]: Q and IndexQ go to the
# combined `output`, K/V scatter into the MAIN paged cache and IndexK into the
# INDEX paged cache. Hardware-agnostic (plain BF16 matmul; runs on AMD
# CDNA4/MI355 and NVIDIA). Attention in M3 is BF16 — quantization applies only
# to the MoE experts, not these projections, so there are no scale-factor
# operands here. The column-band boundaries and the `write_to_caches` scatter
# epilogue are identical to the `_scale_float4` variant above (see its comment
# block for the detailed routing rationale); this variant only drops the
# block-scaling machinery and swaps in the plain `_matmul_common` primitive used
# by the single-cache `mo.fused_qkv_matmul.ragged.paged` kernel.
#
# Column routing over col in [0, N_total) with
#   N_total = q_dim + 2*kv_dim + iq_dim + ik_dim:
#   [0, q_dim)                       -> Q       : output[:, col]
#   [q_dim, q_dim+kv_dim)            -> K       : main k_cache (head/dim from col)
#   [q_dim+kv_dim, q_dim+2*kv_dim)   -> V       : main v_cache
#   [q_dim+2*kv_dim, +iq_dim)        -> IndexQ  : output[:, q_dim + (col-base)]
#   [+iq_dim, +ik_dim)               -> IndexK  : index k_cache (head=0, dim)
def generic_fused_qkv_index_matmul_kv_cache_paged_ragged[
    dtype: DType,
    weight_dtype: DType,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    index_kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    iq_dim: Int,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Performs a fused QKV + index-QK matmul (BF16, non-scaled). Q and IndexQ
    are written to the combined `output` buffer; K/V are scattered into the MAIN
    `kv_collection` and IndexK into the INDEX `index_kv_collection`.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), hidden).
        input_row_offsets: Tensor with shape (batch_size + 1,). The value at
            each index is the start_idx of the corresponding batch in
            hidden_state.
        weight: Concatenated weight W = [Wq | Wk | Wv | Wiq | Wik], shape
            (N_total, hidden) where N_total = q_dim + 2*kv_dim + iq_dim + ik_dim.
        kv_collection: The MAIN KVCache collection (K, V) for this layer.
        index_kv_collection: The INDEX KVCache collection (IndexK only,
            single shared KV head) for this layer.
        layer_idx: The current layer, used to retrieve the KVCache objects.
        iq_dim: Width of the IndexQ output band (num_index_heads *
            idx_head_dim). Passed explicitly because for an MLA index cache it
            is not recoverable from the index cache's `num_heads`.
        output: The pre-allocated combined output buffer for Q and IndexQ
            projections. Shape: (sum(seq_lens), q_dim + iq_dim).
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "idx_num_heads="
                    + String(index_kv_collection.kv_params.num_heads),
                    "idx_head_size="
                    + String(index_kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_index_matmul.ragged.paged.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _fused_qkv_index_matmul_kv_cache_ragged[target=target,](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            index_kv_collection,
            layer_idx,
            iq_dim,
            output,
            ctx,
        )


@always_inline
def _fused_qkv_index_matmul_kv_cache_ragged[
    dtype: DType,
    weight_dtype: DType,
    collection_t: KVCollectionT,
    index_collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    index_kv_collection: index_collection_t,
    layer_idx: UInt32,
    iq_dim: Int,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Resolves both KVCache objects from their collections and dispatches the
    dual-cache impl. K/V come from the MAIN (non-MLA) collection, IndexK from
    the INDEX (MLA, single latent head) collection."""
    comptime kv_params = collection_t.kv_params

    # The MAIN cache is GQA/MHA — it has a real K and V side scattered per
    # head. The INDEX cache is MLA (K-only, single latent head); its scatter
    # uses head 0, so MLA is supported there (the assert below only constrains
    # the main cache). `get_value_cache` is therefore only taken on the main
    # collection, which is always non-MLA.
    comptime assert (
        not kv_params.is_mla
    ), "Dual-cache fused QKV+index: the MAIN (K/V) cache must be non-MLA."

    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)
    var index_k_cache = index_kv_collection.get_key_cache(layer_idx_cast)

    var cuda_ctx: Optional[DeviceContext] = None
    comptime if is_gpu[target]():
        cuda_ctx = context

    return _fused_qkv_index_matmul_kv_cache_ragged_impl[target=target](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        index_k_cache,
        iq_dim,
        output,
        cuda_ctx,
    )


@always_inline
def _fused_qkv_index_matmul_kv_cache_ragged_impl[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT,
    index_cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    k_cache: cache_t,
    v_cache: cache_t,
    index_k_cache: index_cache_t,
    iq_dim: Int,
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    """Dual-cache fused QKV + index matmul on ragged tensors (BF16, non-scaled).

    Q and IndexQ are written to the combined `output` buffer (Q in
    `[0, q_dim)`, IndexQ in `[q_dim, q_dim + iq_dim)`); K/V are scattered into
    the main `k_cache`/`v_cache`; IndexK is scattered into the MLA
    `index_k_cache` (single latent head, head 0).
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params
    comptime index_kv_type = index_cache_t.dtype

    # Q/IndexQ stay `dtype`. Cache dtypes may be e4m3; epilogue saturating-casts.
    comptime assert (
        kv_type == dtype or kv_type == DType.float8_e4m3fn
    ), "the main KV cache must be bf16 or e4m3"
    comptime assert (
        index_kv_type == dtype or index_kv_type == DType.float8_e4m3fn
    ), "the index-K cache must be bf16 or e4m3"

    # Boundaries over the concatenated N dimension. `output` is the combined
    # [M, q_dim + iq_dim] buffer (Q then IndexQ):
    #   kv_dim = main num_heads * head_size      (K band == V band width)
    #   iq_dim = num_index_heads * idx_head_dim  (IndexQ band; PASSED IN, since
    #            for an MLA index cache the index cache's num_heads == 1 != it)
    #   ik_dim = index head_size                 (IndexK band == single MLA
    #            latent head width; not materialized as a var — nothing branches
    #            on it, the scatter uses `col - iq_end`)
    #   q_dim  = output_width - iq_dim           (Q band)
    var kv_dim = kv_params.head_size * kv_params.num_heads
    var output_width = output.dim[1]()
    var q_dim = output_width - iq_dim

    # Fail fast if the stacked weight width doesn't match the routed bands
    # (q_dim + 2*kv_dim + iq_dim + ik_dim). A mismatched wqkv / iq_dim / output
    # would otherwise silently mis-route or read out of bounds in the epilogue.
    comptime ik_dim = index_cache_t.kv_params.head_size
    var n_total = weight.dim[0]()
    if n_total != q_dim + 2 * kv_dim + iq_dim + ik_dim:
        raise Error(
            "fused_qkv_index: stacked weight N (",
            n_total,
            ") != q_dim (",
            q_dim,
            ") + 2*kv_dim (",
            kv_dim,
            ") + iq_dim (",
            iq_dim,
            ") + ik_dim (",
            ik_dim,
            "); check the [Wq|Wk|Wv|Wiq|Wik] weight and iq_dim.",
        )

    # Column band boundaries (exclusive upper bounds).
    var q_end = q_dim  # Q       : [0, q_end)
    var k_end = q_end + kv_dim  # K       : [q_end, k_end)
    var v_end = k_end + kv_dim  # V       : [k_end, v_end)
    var iq_end = v_end + iq_dim  # IndexQ  : [v_end, iq_end)
    # IndexK band is [iq_end, iq_end + ik_dim); no `ik_end` var is needed
    # because nothing branches on the final upper bound (the matmul only emits
    # columns up to it, so the IndexK else-branch is reached only within range).

    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @__parameter
    @__copy_capture(
        q_end,
        k_end,
        v_end,
        iq_end,
        q_dim,
    )
    @always_inline
    def write_to_caches[
        _dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        # The matmul produced the projections; the epilogue just routes and
        # casts. `idx[1]` is the column in the concatenated N space.
        var output_val_out: SIMD[dtype, width] = rebind[SIMD[dtype, width]](
            val.cast[dtype]()
        )

        var col = idx[1]

        # Q band -> output[:, col].
        if col < q_end:
            output.store[width=width](idx, output_val_out)
            return

        # IndexQ band -> output[:, q_dim + (col - v_end)]. Packed right after Q
        # in the combined output. Checked before the cache scatters so the
        # branch order matches the column layout (Q | K | V | IndexQ | IndexK).
        if col >= v_end and col < iq_end:
            output.store[width=width](
                IndexList[2](idx[0], q_dim + (col - v_end)),
                output_val_out,
            )
            return

        var global_token_idx = idx[0]
        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        # K / V bands -> main cache. IndexK band -> index cache.
        if col < v_end:
            var h_idx: Int
            var hd_idx: Int
            var cache: cache_t
            if col < k_end:
                cache = k_cache
                h_idx, hd_idx = udivmod(col - q_end, kv_params.head_size)
            else:
                cache = v_cache
                h_idx, hd_idx = udivmod(col - k_end, kv_params.head_size)
            var cache_length = cache.cache_length(batch_idx)
            var cache_token_idx = token_idx + cache_length
            cache.store(
                batch_idx,
                h_idx,
                cache_token_idx,
                hd_idx,
                rebind[SIMD[kv_type, width]](cast_saturating[kv_type](val)),
            )
        else:
            # IndexK band -> index cache, single shared head (head == 0).
            var hd_idx = col - iq_end
            var cache_length = index_k_cache.cache_length(batch_idx)
            var cache_token_idx = token_idx + cache_length
            index_k_cache.store(
                batch_idx,
                0,
                cache_token_idx,
                hd_idx,
                rebind[SIMD[index_kv_type, width]](
                    cast_saturating[index_kv_type](val)
                ),
            )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"

    _matmul_common[target=target, elementwise_lambda_fn=write_to_caches](
        hidden_state, weight.bitcast[dtype](), context
    )


@always_inline
def _matmul_common[
    dtype: DType,
    //,
    *,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    output_dtype: DType = dtype,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])

    var c_alloc_layout = AllocLayout[Scalar[output_dtype]](
        count=TOTAL_SEQ_LEN * N
    )
    var c_nd: LayoutTensor[
        output_dtype, Layout.row_major(UNKNOWN_VALUE, N), MutUntrackedOrigin
    ]

    comptime if is_cpu[target]():
        # The CPU matmul codepath uses the C buffer as a workspace
        # even if an epilogue is provided, here we just allocate
        # something to ensure we don't segfault.
        var c_ptr = alloc(c_alloc_layout).unsafe_leak()

        c_nd = {
            c_ptr,
            RuntimeLayout[c_nd.layout].row_major(
                IndexList[2](TOTAL_SEQ_LEN, N)
            ),
        }
    else:
        c_nd = {
            None,
            RuntimeLayout[c_nd.layout].row_major(
                IndexList[2](TOTAL_SEQ_LEN, N)
            ),
        }

    matmul[
        target=target,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](lt_to_tt(c_nd), lt_to_tt(hidden_state), lt_to_tt(weight), context)

    comptime if is_cpu[target]():
        dealloc(
            ThinAllocation(unsafe_owned_ptr=c_nd.ptr).unsafe_with_layout(
                c_alloc_layout
            )
        )


@always_inline
def _qmatmul_common[
    dtype: DType,
    //,
    *,
    group_size: Int,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    comptime assert is_gpu[target](), "GPTQ quantization only works on GPU."

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE, N), MutAnyOrigin
    ]

    c_nd = {
        None,
        RuntimeLayout[c_nd.layout].row_major(IndexList[2](TOTAL_SEQ_LEN, N)),
    }

    matmul_gpu_qint4_impl[
        target=target,
        group_size=group_size,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](
        c_nd,
        hidden_state,
        weight,
        context,
    )


@always_inline
def _matmul_blockwise_scaled_fp8_common[
    output_dtype: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[mut=False, a_type, address_space=.GENERIC, ...],
    weight: LayoutTensor[mut=False, b_type, address_space=.GENERIC, ...],
    input_scale: LayoutTensor[
        mut=False, a_scales_type, address_space=.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, b_scales_type, address_space=.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp8 matmul only works on GPU."

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])

    # Helper to convert 2D LayoutTensor to TileTensor. Needed because
    # this function still accepts LayoutTensor parameters. Will be
    # removed when kv_cache_ragged.mojo is fully migrated to TileTensor.
    @always_inline
    def _lt_to_tt[
        dtype: DType,
    ](lt: LayoutTensor[dtype, _, ...]) -> TileTensor[
        dtype, RowMajorLayout[*Coord[Int64, Int64].element_types], lt.origin
    ]:
        var layout = row_major(
            (
                Int64(lt.dim(0)),
                Int64(lt.dim(1)),
            )
        )
        return TileTensor[
            dtype, RowMajorLayout[*Coord[Int64, Int64].element_types], lt.origin
        ](
            ptr=UnsafePointer[Scalar[dtype], lt.origin](
                unsafe_from_address=Int(lt.ptr)
            ),
            layout=layout,
        )

    # Allocate an output-typed scratch buffer for the matmul result; the
    # epilogue lambda reads from it and writes the final values to the KV
    # cache.
    var scratch_buffer = context.enqueue_create_buffer[output_dtype](
        TOTAL_SEQ_LEN * N
    )
    var c_tt = TileTensor(
        ptr=scratch_buffer.unsafe_ptr(),
        layout=row_major((Int64(TOTAL_SEQ_LEN), Int64(N))),
    )

    blockwise_scaled_fp8_with_epilogue[
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
        scales_granularity_mnk=scales_granularity_mnk,
    ](
        c_tt,
        _lt_to_tt(hidden_state),
        _lt_to_tt(weight),
        _lt_to_tt(input_scale),
        _lt_to_tt(weight_scale),
        context,
    )


@always_inline
def _matmul_blockwise_scaled_fp4_common[
    output_dtype: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    # c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    //,
    *,
    target: StaticString,
    SF_VECTOR_SIZE: Int = 16,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    weight: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    input_scale: LayoutTensor[scales_dtype, sfa_layout, ImmutAnyOrigin],
    weight_scale: LayoutTensor[scales_dtype, sfb_layout, ImmutAnyOrigin],
    tensor_sf: Float32,
    context: DeviceContext,
) raises:
    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp4 matmul only works on GPU."

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])

    # Scratch C for the matmul result. With the fused (in-kernel) epilogue the
    # SM100 Mojo kernel redirects its stores through `elementwise_lambda_fn` and
    # never writes C; the buffer only backs the vendor DISPATCH_MISS fallback.
    #
    # C is built with a STATIC N dim (dynamic M): `block_scaled_matmul` picks its
    # SM100 config from `c.static_shape[1]`. A fully-dynamic layout leaves
    # static_N unknown, which makes `choose_block_scaled_config` select MMA_N=0
    # for the cta_group=2 (prefill) regime -- a config that
    # `build_block_scaled_configs` never enumerates -- so every prefill QKV GEMM
    # would DISPATCH_MISS to vendor cuBLASLt instead of MAX's own kernel. (The
    # decode cta_group=1 regime does not use N to pick MMA_N, so it reached the
    # Mojo kernel regardless.)
    var scratch_buffer = context.enqueue_create_buffer[output_dtype](
        TOTAL_SEQ_LEN * N
    )
    # `Idx[N]` keeps the N dim STATIC (M stays dynamic).
    var c_tt = TileTensor(
        scratch_buffer.unsafe_ptr(),
        row_major(TOTAL_SEQ_LEN, Idx[N]),
    )

    var a_scales_tt = lt_to_tt(input_scale)
    var b_scales_tt = lt_to_tt(weight_scale)

    # Try MAX's own SM100 Mojo block-scaled kernel first; vendor cuBLASLt only
    # on DISPATCH_MISS. The K/V-scatter epilogue rides `elementwise_lambda_fn`.
    #
    # This path runs a single K partition: a B200 microbench of the served
    # decode shapes (M<=16, N in {2304,2560}, K=6144) showed split-K within
    # noise of the single-launch kernel (the GEMM is latency-bound, not
    # occupancy-bound), so split-K carries no benefit here; the `k_group_size=4`
    # dispatch tuning carries the real win.
    #
    # PDL (programmatic dependent launch) overlaps this GEMM's prologue with the
    # tail of the upstream grid (RMSNorm+MXFP8-quantize) and releases the
    # downstream grid (attention) only after the fused scatter epilogue
    # completes -- the small-BN/main kernels fire `launch_dependent_grids()`
    # after `tmem_dealloc_mbar.wait()`, so the KV/index stores are visible
    # before any dependent grid runs. The upstream RMSNorm-block-scaled and
    # downstream attention kernels already default `PDLLevel.ON`; keeping this
    # GEMM OFF forced a full serialization bubble on both sides.
    # AMD reaches the same band-routing epilogue through its own block-scaled
    # kernel: `block_scaled_matmul` is SM10x/SM12x only, and the AMD kernel
    # consumes rank-2 E8M0 scales and raw byte operands rather than the SM100
    # SF-atom layout. The per-block scales already carry the whole scaling, so
    # there is no `tensor_sf` to apply.
    #
    # The callee bottoms out in `cdna4_block_scaled_mfma`, so this needs CDNA4
    # (gfx950) in practice; the gate is any-AMD to match the predicate used
    # across the rest of the AMD kernels.
    comptime if has_amd_gpu_accelerator():
        if tensor_sf != 1.0:
            raise Error(
                "CDNA4 block-scaled fused QKV+index expects a unit tensor"
                " scale; MXFP8 carries all scaling in the E8M0 blocks."
            )
        comptime assert (
            scales_dtype == .float8_e8m0fnu
        ), "CDNA4 block-scaled fused QKV+index requires E8M0 scales"
        return block_scaled_matmul_amd[
            lane_bytes=32, elementwise_lambda_fn=elementwise_lambda_fn
        ](
            c_tt,
            lt_to_tt(hidden_state).bitcast[.uint8](),
            lt_to_tt(weight).bitcast[.uint8](),
            a_scales_tt.bitcast[.float8_e8m0fnu](),
            b_scales_tt.bitcast[.float8_e8m0fnu](),
            context,
        )

    block_scaled_matmul[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
        target=target,
        pdl_level=PDLLevel.ON,
    ](
        c_tt,
        lt_to_tt(hidden_state),
        lt_to_tt(weight),
        a_scales_tt,
        b_scales_tt,
        tensor_sf,
        context,
    )


# ===-----------------------------------------------------------------------===#
# Unfused KV cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


def kv_matmul_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    //,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
        ...,
    ],
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """Performs a matmul, writing the output into a mutable ContinuousBatchingKVCacheCollection object.

    Parameters:
        dtype: Element type of the `hidden_state` and `weight` tensors and of
            the KV cache entries (inferred).
        params: Static shape parameters of the paged KV cache, including the
            attention-head count, per-head size, and MLA flag (inferred).
        page_size: Number of tokens stored per cache page (inferred).
        target: Compilation target string used to dispatch GPU versus CPU
            paths.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.kv_matmul.ragged.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _matmul_kv_cache_ragged[target=target](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            ctx,
        )


@always_inline
def _matmul_kv_cache_ragged[
    dtype: DType, //, *, target: StaticString
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    context: DeviceContext,
) raises:
    """Helper for performing matmul with custom ContinuousBatchingKVCacheCollection dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, 2 * num_kv_heads * head_size)
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime if is_gpu[target]():
        cuda_ctx = context

    _matmul_kv_cache_ragged_impl[target=target](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        cuda_ctx,
    )


# HACK: `k_cache` and `v_cache` are the key/value halves (kv_idx 0 vs 1) of the
# same `blocks` buffer, so they share the collection's mutable `blocks_origin`
# (and `scales_origin`). They are only ever stored to at disjoint offsets, but
# the exclusivity checker cannot prove that and rejects passing both as
# separately-writable arguments to this call. Disabling the nested-origin
# exclusivity check is a stopgap workaround; the proper fix is to give the k/v
# views provably-disjoint origins instead of sharing the collection's.
@__unsafe_nested_origins_read_only
@always_inline
def _matmul_kv_cache_ragged_impl[
    dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    k_cache: cache_t,
    v_cache: cache_t,
    ctx: Optional[DeviceContext],
) raises:
    """Helper for performing matmul with custom KVCacheT dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, 2 * num_kv_heads * head_size)
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: Pointer containing the runtime context for the target device.
    """
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    comptime kv_params = cache_t.kv_params

    var batch_size = input_row_offsets.dim[0]() - 1

    # Set the matmul_common output lambda to write to K cache for the first N
    # elements and V cache for the next N.
    var k_offset = kv_params.head_size * kv_params.num_heads

    @__parameter
    @__copy_capture(input_row_offsets, k_offset, batch_size)
    @always_inline
    def write_to_cache_common[
        dtype: DType, cache_t: KVCacheT, width: SIMDLength
    ](
        k_cache: cache_t,
        v_cache: cache_t,
        idx: IndexList[2],
        val: SIMD[dtype, width],
    ):
        comptime kv_type = cache_t.dtype

        comptime assert (
            kv_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        var global_token_idx = idx[0]

        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var cache: cache_t
        var h_idx: Int
        var hd_idx: Int
        if idx[1] < k_offset:
            # Write this element to the K cache.
            cache = k_cache
            h_idx, hd_idx = udivmod(idx[1], kv_params.head_size)
        else:
            # Otherwise, write this element to the V cache.
            cache = v_cache
            h_idx, hd_idx = udivmod(idx[1] - k_offset, kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](val),
        )

    # Cast to a register passable dtype so the function closure works on GPU.
    var k_cache_reg = rebind[cache_t](k_cache)
    var v_cache_reg = rebind[cache_t](v_cache)

    @__parameter
    @__copy_capture(k_cache_reg, v_cache_reg)
    @always_inline
    def write_to_cache_continuous[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        write_to_cache_common(k_cache_reg, v_cache_reg, idx, val)

    _matmul_common[
        target=target, elementwise_lambda_fn=write_to_cache_continuous
    ](hidden_state, weight, ctx)


# ===-----------------------------------------------------------------------===#
# Unfused K cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


def k_matmul_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    //,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
        ...,
    ],
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """Performs a matmul, writing the output into a mutable PagedKVCacheCollection object.

    Parameters:
        dtype: Element type of the `hidden_state` input, the `weight` tensor,
            and the KV cache entries (inferred).
        params: Static `KVCacheStaticParams` describing the cache layout, such
            as head count and head dimension (inferred).
        page_size: Number of tokens stored per cache page in the paged KV
            cache (inferred).
        target: Compilation target string used to dispatch GPU versus CPU
            paths.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.k_matmul.ragged.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        return _matmul_k_cache_ragged[target=target](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            ctx,
        )


@always_inline
def _matmul_k_cache_ragged[
    dtype: DType,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    context: DeviceContext,
) raises:
    """Helper for performing matmul with custom PagedKVCacheCollection dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)

    comptime if is_gpu[target]():
        cuda_ctx = context

    _matmul_k_cache_ragged_impl[target=target](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        cuda_ctx,
    )


@always_inline
def _matmul_k_cache_ragged_impl[
    dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    k_cache: cache_t,
    ctx: Optional[DeviceContext],
) raises:
    """Helper for performing matmul with custom KVCacheT dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: Pointer containing the runtime context for the target device.
    """
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    comptime kv_params = cache_t.kv_params

    var batch_size = input_row_offsets.dim[0]() - 1

    @__parameter
    @__copy_capture(batch_size)
    @always_inline
    def write_to_cache[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width],):
        comptime kv_type = cache_t.dtype

        comptime assert (
            kv_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        var global_token_idx = idx[0]

        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx, hd_idx = udivmod(idx[1], kv_params.head_size)

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        k_cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](val),
        )

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, ctx
    )


def k_matmul_ragged_paged_scale[
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """Performs a matmul, writing the output into a mutable
    PagedKVCacheCollection object.

    Parameters:
        dtype: Element type of the `hidden_state` input tensor and of the
            key entries written to the KV cache.
        weight_dtype: Element type of the `weight` tensor; must match
            `dtype`.
        scale_dtype: Element type of the `input_scale` and `weight_scale`
            tensors.
        target: Compilation target string; must be a GPU target.
        scales_granularity_mnk: Block sizes along the M, N, and K matmul
            dimensions used to tile the scale application; `-1` selects
            per-tensor scaling, `1` selects per-channel scaling, and any
            other value selects blockwise scaling.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "input_scale", input_scale.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "weight_scale", weight_scale.runtime_layout.shape.value
                    ),
                    "layer_idx=" + String(layer_idx),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.k_matmul.ragged.paged.scale.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        comptime assert is_gpu[
            target
        ](), "Blockwise scaled fp8 matmul only works on GPU."
        var layer_idx_cast = Int(layer_idx)
        var k_cache = kv_collection.get_key_cache(layer_idx_cast)

        return _matmul_k_cache_ragged_scale_impl[
            target=target,
            scales_granularity_mnk=scales_granularity_mnk,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            k_cache,
            ctx,
        )


@always_inline
def _matmul_k_cache_ragged_scale_impl[
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
](
    hidden_state: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, address_space=.GENERIC, ...],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, address_space=.GENERIC, ...
    ],
    k_cache: cache_t,
    ctx: DeviceContext,
) raises:
    """Helper for performing matmul with custom KVCacheT dtypes.

    Currently assumes block size scaling.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: Pointer containing the runtime context for the target device.
    """
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    comptime kv_params = cache_t.kv_params

    var batch_size = input_row_offsets.dim[0]() - 1

    @__parameter
    @__copy_capture(input_scale, weight_scale, batch_size)
    @always_inline
    def write_to_cache[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width],):
        comptime kv_type = cache_t.dtype

        comptime assert (
            kv_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        var global_token_idx = idx[0]

        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx, hd_idx = udivmod(idx[1], kv_params.head_size)

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        k_cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](val),
        )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"
    _matmul_blockwise_scaled_fp8_common[
        output_dtype=cache_t.dtype,
        target=target,
        elementwise_lambda_fn=write_to_cache,
        scales_granularity_mnk=scales_granularity_mnk,
    ](hidden_state, weight.bitcast[dtype](), input_scale, weight_scale, ctx)


# ===-----------------------------------------------------------------------===#
# Unfused gguf quantized QKV cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


def unfused_qkv_matmul_ragged_paged_gguf_quantized[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    //,
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, .float32, address_space=.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    q_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    k_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    v_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
        ...,
    ],
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Performs a quantized matmul, writing the output into a mutable PagedKVCacheCollection object.

    Unlike the un-quantized version (kv_matmul_ragged_continuous_batching), this
    implementation does not concat the q, k, and v weights together. Instead, it
    performs three matmuls. This allows the q, k, and v weights to have different
    quantization encodings.

    This is only supported on CPU.

    Parameters:
        dtype: Element type of the KV cache collection entries (inferred).
        params: Static shape parameters of the paged KV cache, including the
            attention-head count, per-head size, and MLA flag (inferred).
        page_size: Number of tokens stored per cache page (inferred).
        quantization_encoding_q: GGUF quantization encoding name applied to
            the `q_weight` tensor.
        quantization_encoding_k: GGUF quantization encoding name applied to
            the `k_weight` tensor.
        quantization_encoding_v: GGUF quantization encoding name applied to
            the `v_weight` tensor.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        q_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        v_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The Collection object storing KVCache entries.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: Tensor with shape (sum(seq_lens), num_kv_heads * head_size).
            This is the output buffer for the Q matmul.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q_weight", q_weight.runtime_layout.shape.value),
                    trace_arg("k_weight", k_weight.runtime_layout.shape.value),
                    trace_arg("v_weight", v_weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "quantization_encoding_q=" + quantization_encoding_q,
                    "quantization_encoding_k=" + quantization_encoding_k,
                    "quantization_encoding_v=" + quantization_encoding_v,
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
        "mo.kv_matmul.ragged.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size)
        + ".quantization_encoding_q="
        + quantization_encoding_q
        + ".quantization_encoding_k="
        + quantization_encoding_k
        + ".quantization_encoding_v="
        + quantization_encoding_v,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
    ):
        return _unfused_qkv_matmul_ragged_paged_gguf_quantized_impl[
            quantization_encoding_q,
            quantization_encoding_k,
            quantization_encoding_v,
        ](
            hidden_state,
            input_row_offsets,
            q_weight,
            k_weight,
            v_weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
def _unfused_qkv_matmul_ragged_paged_gguf_quantized_impl[
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, .float32, address_space=.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    q_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    k_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    v_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime cache_t = PagedKVCache[
        DType.float32,
        kv_collection.kv_params,
        kv_collection.page_size,
        kv_collection.blocks_origin,
        kv_collection.cache_lengths_origin,
        kv_collection.lookup_table_origin,
        kv_collection.scales_origin,
    ]
    var k_cache_reg = rebind[cache_t](k_cache)
    var v_cache_reg = rebind[cache_t](v_cache)

    _matmul_kv_cache_ragged_gguf_quantized_impl[
        cache_t,
        quantization_encoding_q,
        quantization_encoding_k,
        quantization_encoding_v,
    ](
        hidden_state,
        input_row_offsets,
        q_weight,
        k_weight,
        v_weight,
        k_cache_reg,
        v_cache_reg,
        output,
    )


# HACK: `k_cache` and `v_cache` are the key/value halves (kv_idx 0 vs 1) of the
# same `blocks` buffer, so they share the collection's mutable `blocks_origin`
# (and `scales_origin`). They are only ever stored to at disjoint offsets, but
# the exclusivity checker cannot prove that and rejects passing both as
# separately-writable arguments to this call. Disabling the nested-origin
# exclusivity check is a stopgap workaround; the proper fix is to give the k/v
# views provably-disjoint origins instead of sharing the collection's.
@__unsafe_nested_origins_read_only
@always_inline
def _matmul_kv_cache_ragged_gguf_quantized_impl[
    cache_t: KVCacheT,
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, .float32, address_space=.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    q_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    k_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    v_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    k_cache: cache_t,
    v_cache: cache_t,
    output: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
) raises:
    """Helper for performing quantized matmul with custom KVCacheT dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_kv_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        q_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        k_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        v_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        k_cache: The Collection object storing KVCache K entries.
        v_cache: The Collection object storing KVCache V entries.
        output: Tensor with shape (sum(seq_lens), num_kv_heads * head_size).
            This is the output buffer for the Q matmul.
    """
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    # K matmul with epilogue
    _qmatmul_k_or_v_cache_ragged_gguf_quantized_impl[
        cache_t, quantization_encoding_k
    ](hidden_state, input_row_offsets, k_weight, k_cache)

    # V matmul with epilogue
    _qmatmul_k_or_v_cache_ragged_gguf_quantized_impl[
        cache_t, quantization_encoding_v
    ](hidden_state, input_row_offsets, v_weight, v_cache)

    # Q matmul without epilogue which writes to output buffer
    _qmatmul_gguf_quantized_common[quantization_encoding_q](
        hidden_state, q_weight, output
    )


@always_inline
def _qmatmul_k_or_v_cache_ragged_gguf_quantized_impl[
    cache_t: KVCacheT,
    quantization_encoding: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, .float32, address_space=.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    k_or_v_weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    k_or_v_cache: cache_t,
) raises:
    comptime kv_params = cache_t.kv_params

    var batch_size = input_row_offsets.dim[0]() - 1

    @__parameter
    @__copy_capture(input_row_offsets, batch_size)
    @always_inline
    def write_to_cache_common[
        dtype: DType, cache_t: KVCacheT, width: SIMDLength
    ](k_or_v_cache: cache_t, idx: IndexList[2], val: SIMD[dtype, width],):
        comptime k_or_v_type = cache_t.dtype

        comptime assert (
            k_or_v_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        var global_token_idx = idx[0]

        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        # Write this element to the K or V cache.
        var cache = k_or_v_cache
        var h_idx, hd_idx = udivmod(idx[1], kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[k_or_v_type, width]](val),
        )

    @__parameter
    @__copy_capture(k_or_v_cache)
    def write_to_k_or_v_cache_continuous[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        write_to_cache_common(k_or_v_cache, idx, val)

    _qmatmul_gguf_quantized_alloc_output[
        quantization_encoding,
        elementwise_lambda_fn=write_to_k_or_v_cache_continuous,
    ](hidden_state, k_or_v_weight)


@always_inline
def _qmatmul_gguf_quantized_alloc_output[
    quantization_encoding: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        mut=False, .float32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        .float32, Layout.row_major(UNKNOWN_VALUE, N), MutUntrackedOrigin
    ]

    # The CPU matmul codepath uses the C buffer as a workspace
    # even if an epilogue is provided, here we just allocate
    # something to ensure we don't segfault.
    var c_ptr = alloc(
        AllocLayout[Float32](count=TOTAL_SEQ_LEN * N)
    ).unsafe_leak()

    c_nd = {
        c_ptr,
        RuntimeLayout[c_nd.layout].row_major(IndexList[2](TOTAL_SEQ_LEN, N)),
    }

    _qmatmul_gguf_quantized_common[
        quantization_encoding, elementwise_lambda_fn
    ](hidden_state, weight, c_nd)

    dealloc(
        ThinAllocation(unsafe_owned_ptr=c_ptr).unsafe_with_layout(
            AllocLayout[Float32](count=TOTAL_SEQ_LEN * N)
        )
    )


@always_inline
def _qmatmul_gguf_quantized_common[
    quantization_encoding: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        mut=False, .float32, address_space=.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, .uint8, address_space=.GENERIC, ...],
    output: LayoutTensor[mut=True, .float32, address_space=.GENERIC, ...],
) raises:
    comptime if quantization_encoding == "q4_0":
        matmul_qint4[32, elementwise_lambda_fn=elementwise_lambda_fn](
            lt_to_tt(hidden_state),
            lt_to_tt(weight),
            lt_to_tt(output),
        )
    elif quantization_encoding == "q4_k":
        matmul_Q4_K[elementwise_lambda_fn=elementwise_lambda_fn](
            lt_to_tt(hidden_state),
            lt_to_tt(weight),
            lt_to_tt(output),
        )
    elif quantization_encoding == "q6_k":
        matmul_Q6_K[elementwise_lambda_fn=elementwise_lambda_fn](
            lt_to_tt(hidden_state),
            lt_to_tt(weight),
            lt_to_tt(output),
        )
    else:
        raise Error(
            "Unsupported quantization encoding: ", quantization_encoding
        )


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qk_rope_bshd_paged_ragged[
    dtype: DType,
    freq_dtype: DType,
    //,
    *,
    interleaved: Bool,
    has_position_ids: Bool,
    target: StaticString,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
](
    q_proj: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    freqs_cis: TileTensor[mut=False, freq_dtype, address_space=.GENERIC, ...],
    position_ids: TileTensor[.uint32, address_space=.GENERIC, ...],
    layer_idx: UInt32,
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque dtypes in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque dtype as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only executed after QKV completes.

    Parameters:
        dtype: Data type of the `q_proj` and `output` tensors (inferred).
        freq_dtype: Data type of the `freqs_cis` RoPE frequency table
            (inferred).
        interleaved: Whether RoPE applies interleaved (GPT-NeoX style) rotation
            to adjacent element pairs.
        has_position_ids: Whether per-token `position_ids` are provided; when
            `False`, `position_ids` is unused.
        target: Target device string for kernel dispatch.
        mrope_types: TypeList of coordinate element types constraining
            `mrope_section` for multimodal RoPE.
        mrope_section: Optional section sizes splitting the head dimension into
            temporal, height, and width spans for multimodal RoPE (defaults to
            `None`).

    Args:
        q_proj: Query projection tile tensor with shape (sum(seq_lens),
            num_heads, head_size).
        input_row_offsets: Tile tensor with shape (batch_size + 1,) denoting
            the start of each sequence along the ragged sequence dimension.
        kv_collection: The paged KV cache collection storing the K cache for
            this layer, retrieved via `layer_idx`.
        freqs_cis: Precomputed RoPE frequency table applied to Q and K.
        position_ids: Per-token position indices used to index into
            `freqs_cis`; ignored when `has_position_ids` is `False`.
        layer_idx: The index of the layer being executed, used to retrieve the
            K cache from `kv_collection`.
        output: The pre-allocated output tile tensor receiving the rotated Q
            projection.
        context: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_proj",
                        coord_to_index_list(q_proj.layout.shape_coord()),
                    ),
                    trace_arg(
                        "freqs_cis",
                        coord_to_index_list(freqs_cis.layout.shape_coord()),
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "interleaved=" + String(interleaved),
                ]
            )
        )

    comptime name = "mo.fused_qk_rope.ragged.paged.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        comptime if has_position_ids:
            fused_qk_rope_ragged[
                kv_collection.CacheType,
                interleaved=interleaved,
                target=target,
                PositionIdsLayoutType=position_ids.LayoutType,
                mrope_types=mrope_types,
                mrope_section=mrope_section,
            ](
                q_proj,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                TileTensor(
                    position_ids.ptr.unsafe_mut_cast[
                        True
                    ]().as_unsafe_any_origin(),
                    position_ids.layout,
                ).as_immut(),
                layer_idx,
                output,
                context,
            )
        else:
            fused_qk_rope_ragged[
                kv_collection.CacheType, interleaved=interleaved, target=target
            ](
                q_proj,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                None,
                layer_idx,
                output,
                context,
            )


# ===-----------------------------------------------------------------------===#
# MHA (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_flash_attention_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = dtype,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    decode_dispatch_metadata: MHADecodeDispatchMetadata,
) raises:
    """Dispatches flash attention over a ragged batch against a paged KV cache.

    Parameters:
        collection_t: The KV cache collection type storing the K and V caches
            for this layer (inferred).
        dtype: Data type of the query tensor (inferred).
        target: Target device string for kernel dispatch.
        mask_str: Attention mask name selecting the masking strategy, such as
            "causal", "null", or "sliding_window_causal".
        local_window_size: Sliding-window size in tokens for windowed masks;
            -1 for masks that ignore it (defaults to -1).
        output_dtype: Data type of the `output` tensor (defaults to `dtype`).

    Args:
        q: Query tensor with shape (sum(seq_lens), num_heads, head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each sequence along the ragged sequence dimension.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        scale: The scaling factor in scaled dot-product attention, usually
            rsqrt(head_size).
        output: The pre-allocated output buffer to write results to, with shape
            (sum(seq_lens), num_heads, head_size).
        context: The call context pointer, passed by the graph compiler.
        decode_dispatch_metadata: Precomputed dispatch metadata used to select
            decode kernels for the GPU target.
    """

    @always_inline
    def description_fn() {imm} -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q.runtime_layout.shape.value))
        desc_parts.append("scale=" + String(scale))
        desc_parts.append("layer_idx=" + String(layer_idx))
        desc_parts.append(
            "num_heads=" + String(collection_t.kv_params.num_heads)
        )
        desc_parts.append(
            "head_size=" + String(collection_t.kv_params.head_size)
        )
        desc_parts.append("local_window_size=" + String(local_window_size))
        desc_parts.append("sink=False")
        return String(";").join(desc_parts)

    comptime name = "mo.mha.ragged." + collection_t.name_str + "." + mask_str + ".nhead_" + String(
        collection_t.kv_params.num_heads
    ) + ".hdim_" + String(
        collection_t.kv_params.head_size
    )

    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return _flash_attention_dispatch[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            decode_dispatch_metadata,
        )


@always_inline
def _launch_flash_attention_with_mask[
    dtype: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    //,
    *,
    target: StaticString,
    output_dtype: DType = dtype,
    sink: Bool = False,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    decode_dispatch_metadata: MHADecodeDispatchMetadata,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    if q.dim[0]() == 0:
        return

    comptime if is_cpu[target]():
        comptime assert output_dtype == dtype, (
            "CPU flash attention requires output dtype == q dtype;"
            " the distinct-output-dtype (fp8->bf16) path is GPU-only."
        )
        return flash_attention_kv_cache_cpu(
            q,
            input_row_offsets,
            input_row_offsets,
            k,
            v,
            mask,
            scale,
            output.bitcast[dtype](),
            sink_weights,
        )
    else:
        gpu_flash_attention[ragged=True, sink=sink](
            output,
            q,
            k,
            v,
            mask,
            input_row_offsets,
            scale,
            context,
            sink_weights=sink_weights,
            decode_dispatch_metadata=OptionalReg[MHADecodeDispatchMetadata](
                decode_dispatch_metadata
            ),
        )


def _flash_attention_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = dtype,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    decode_dispatch_metadata: MHADecodeDispatchMetadata,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    def _dispatch_flash_attention[
        mask_t: MHAMask
    ](mask: mask_t) raises {var k, var v, imm}:
        def call_flash_attention[sink: Bool]() raises {imm}:
            return _launch_flash_attention_with_mask[
                target=target,
                output_dtype=output_dtype,
                sink=sink,
            ](
                q,
                input_row_offsets,
                k,
                v,
                mask,
                scale,
                output,
                context,
                decode_dispatch_metadata,
                sink_weights,
            )

        unswitch(Bool(sink_weights), call_flash_attention)

    return dispatch_mask[
        mask_str,
        local_window_size,
    ](_dispatch_flash_attention)


@always_inline
def generic_flash_attention_kv_cache_ragged_rel_logits[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = dtype,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    bias: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    cache_lengths: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    decode_dispatch_metadata: MHADecodeDispatchMetadata,
) raises:
    """Flash attention over a ragged batch with a relative-position bias.

    `bias` is gathered by `rel_dist = q_pos - k_pos` inside the kernel via
    `RelativeLogitsMask`; `local_window_size` picks the visibility mask
    (`-1` -> global causal). `cache_lengths` is the same tensor used to build
    `kv_collection`, passed again so the mask can recover each query's
    ragged-flat row into `bias` (the collection only exposes a per-sequence
    scalar accessor, not the raw tensor).
    """

    @always_inline
    def description_fn() {imm} -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q.runtime_layout.shape.value))
        desc_parts.append("scale=" + String(scale))
        desc_parts.append("layer_idx=" + String(layer_idx))
        desc_parts.append(
            "num_heads=" + String(collection_t.kv_params.num_heads)
        )
        desc_parts.append(
            "head_size=" + String(collection_t.kv_params.head_size)
        )
        desc_parts.append("local_window_size=" + String(local_window_size))
        desc_parts.append("relative_position_bias=True")
        return String(";").join(desc_parts)

    comptime name = "mo.mha.ragged." + collection_t.name_str + ".rel_logits.nhead_" + String(
        collection_t.kv_params.num_heads
    ) + ".hdim_" + String(
        collection_t.kv_params.head_size
    )

    var k = kv_collection.get_key_cache(Int(layer_idx))
    var v = kv_collection.get_value_cache(Int(layer_idx))

    def _dispatch_flash_attention[
        mask_t: MHAMask
    ](mask: mask_t) raises {var k, var v, imm}:
        return _launch_flash_attention_with_mask[
            target=target,
            output_dtype=output_dtype,
        ](
            q,
            input_row_offsets,
            k,
            v,
            mask,
            scale,
            output,
            context,
            decode_dispatch_metadata,
        )

    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return dispatch_relative_logits_mask[local_window_size,](
            LayoutTensor[bias.dtype, bias.layout, bias.origin](
                bias.ptr,
                RuntimeLayout[bias.layout].row_major(
                    bias.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_lengths,
            input_row_offsets,
            _dispatch_flash_attention,
        )


@always_inline
def generic_flash_attention_kv_cache_ragged_sink[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = dtype,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    sink_weights: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    decode_dispatch_metadata: MHADecodeDispatchMetadata,
) raises:
    """Dispatches flash attention over a ragged batch with attention sink weights.

    Parameters:
        collection_t: The KV cache collection type storing the K and V caches
            for this layer (inferred).
        dtype: Data type of the query tensor (inferred).
        target: Target device string for kernel dispatch.
        mask_str: Attention mask name selecting the masking strategy, such as
            "causal", "null", or "sliding_window_causal".
        local_window_size: Sliding-window size in tokens for windowed masks;
            -1 for masks that ignore it (defaults to -1).
        output_dtype: Data type of the `output` tensor (defaults to `dtype`).

    Args:
        q: Query tensor with shape (sum(seq_lens), num_heads, head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each sequence along the ragged sequence dimension.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        scale: The scaling factor in scaled dot-product attention, usually
            rsqrt(head_size).
        output: The pre-allocated output buffer to write results to, with shape
            (sum(seq_lens), num_heads, head_size).
        context: The call context pointer, passed by the graph compiler.
        sink_weights: Per-batch attention sink weights applied to the leading
            cache slots.
        decode_dispatch_metadata: Precomputed dispatch metadata used to select
            decode kernels for the GPU target.
    """

    @always_inline
    def description_fn() {imm} -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q.runtime_layout.shape.value))
        desc_parts.append("scale=" + String(scale))
        desc_parts.append("layer_idx=" + String(layer_idx))
        desc_parts.append(
            "num_heads=" + String(collection_t.kv_params.num_heads)
        )
        desc_parts.append(
            "head_size=" + String(collection_t.kv_params.head_size)
        )
        desc_parts.append("local_window_size=" + String(local_window_size))
        desc_parts.append("sink=True")
        return String(";").join(desc_parts)

    comptime name = "mo.mha.ragged." + collection_t.name_str + "." + mask_str + ".nhead_" + String(
        collection_t.kv_params.num_heads
    ) + ".hdim_" + String(
        collection_t.kv_params.head_size
    )

    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return _flash_attention_dispatch[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            decode_dispatch_metadata,
            as_dynamic_row_major_1d(sink_weights),
        )


# ===-----------------------------------------------------------------------===#
# MLA (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_flare_mla_decode_kv_cache_ragged[
    collection_t: KVCollectionT,
    q_dtype: DType,
    //,
    mask_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
    per_token_scale_rope_aware: Bool = False,
    sparse_mla: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141); threaded to flare_mla_decoding.
    fold_shared_index: Bool = False,
](
    q: TileTensor[mut=False, q_dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[mut=False, .int64, address_space=.GENERIC, ...],
    context: DeviceContext,
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    extra_k: OptionalReg[collection_t.CacheType] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar: forwarded from the MoGG op so SM100 grid
    # sizing matches the kernel's divmod on scalar_args_buf[2].
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """Dispatches MLA decode attention over a ragged batch against a paged KV cache.

    Parameters:
        collection_t: The KV cache collection type storing the K cache for this
            layer (inferred).
        q_dtype: Data type of the query tensor (inferred).
        mask_str: Attention mask name selecting the masking strategy, such as
            "causal", "null", or "sliding_window_causal".
        target: Target device string for kernel dispatch; must be a GPU
            target.
        local_window_size: Sliding-window size in tokens for windowed masks;
            -1 for masks that ignore it (defaults to -1).
        per_token_scale_rope_aware: Whether `q` and the KV cache use the
            interleaved FP8+BF16 rope-aware layout (defaults to `False`).
        sparse_mla: Whether to use sparse attention with pre-computed
            physical KV row indices via gather4 TMA (defaults to `False`).
        fold_shared_index: Whether to use the read-once shared-index MTP
            fold threaded to `flare_mla_decoding` (defaults to `False`).

    Args:
        q: Query tile tensor with shape (batch_size, num_heads, q_head_size).
        input_row_offsets: Tile tensor with shape (batch_size + 1,) denoting
            the start of each Q entry in the batch.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        scale: The scaling factor in scaled dot-product attention, usually
            rsqrt(head_size).
        output: The pre-allocated output tile tensor to write results to.
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        context: The call context pointer, passed by the graph compiler.
        q_scale_ptr: Per-token Q scale pointer (float32 array, one per Q
            token); defaults to null (sigma_Q = 1.0).
        d_indices: Optional device pointer to packed int32 physical KV row
            indices for sparse decode.
        indices_stride: Stride between batch rows in d_indices (e.g. max
            top-k).
        topk_lengths: Optional per-batch valid top-k counts.
        attn_sink_ptr: Optional per-batch attention sink weights.
        extra_k: Optional second KV cache operand for the extra stream.
        extra_d_indices: Optional extra KV stream sparse indices.
        extra_indices_stride: Stride for extra_d_indices.
        extra_topk_lengths: Optional per-batch lengths for the extra stream.
        extra_scales_ptr: Optional extra stream scales.
        num_partitions_in: Capturable-graph num_partitions override forwarded
            from the MoGG op so SM100 grid sizing matches the kernel.
        logical_indices: Logical sparse indices for position-based causal
            masking; `None` keeps the prior slot-count behavior.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    "scale=" + String(scale),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mla.decode.ragged."
        + collection_t.name_str
        + "."
        + mask_str
        + ".nhead_"
        + String(collection_t.kv_params.num_heads)
        + ".hdim_"
        + String(collection_t.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return _flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            per_token_scale_rope_aware=per_token_scale_rope_aware,
            sparse_mla=sparse_mla,
            fold_shared_index=fold_shared_index,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            scalar_args_buf,
            context,
            q_scale_ptr,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            extra_k,
            extra_d_indices,
            extra_indices_stride,
            extra_topk_lengths,
            extra_scales_ptr,
            num_partitions_in,
            logical_indices,
        )


@always_inline
def _flare_mla_decode_kv_cache_ragged[
    q_dtype: DType,
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
    per_token_scale_rope_aware: Bool = False,
    sparse_mla: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141); threaded to flare_mla_decoding.
    fold_shared_index: Bool = False,
](
    q: TileTensor[mut=False, q_dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[mut=False, .int64, address_space=.GENERIC, ...],
    context: DeviceContext,
    # TODO: Must use OptionalReg as Optional does not work with @__copy_capture.
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    extra_k: OptionalReg[collection_t.CacheType] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar from the dispatcher input list. Optional[Int]
    # is not @__copy_capture-able, so we unpack to (has, value) before the
    # closure and rebuild Optional[Int] inside it.
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """Performs flash attention using k and v caches from KVCacheT custom dtypes.

    Args:
        q: Tensor with shape (batch_size, num_heads, seq_len, head_size).
        input_row_offsets: The start and end position of each Q entry in the batch.
        kv_collection: The Collection object storing out KVCache entries for this layer.
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_collection.
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        context: Pointer containing the runtime context for the target device.
        q_scale_ptr: Per-token Q scale pointer (float32 array, one per Q token).
            Default is null (sigma_Q = 1.0).
        d_indices: Optional device pointer to packed int32 physical KV row indices
            for sparse decode (see ``flare_mla_decoding``).
        indices_stride: Stride between batch rows in ``d_indices`` (e.g. max top-k).
        topk_lengths: Optional per-batch valid top-k counts.
        attn_sink_ptr: Optional per-batch attention sink weights.
        extra_k: Optional second KV cache operand (see ``flare_mla_decoding``).
        extra_d_indices: Optional extra KV stream sparse indices.
        extra_indices_stride: Stride for ``extra_d_indices``.
        extra_topk_lengths: Optional per-batch lengths for extra stream.
        extra_scales_ptr: Optional extra stream scales.
        num_partitions_in: Capturable-graph num_partitions override.
        logical_indices: Logical sparse indices for position-based causal
            masking; `None` keeps the prior slot-count behavior.
    """
    comptime assert is_gpu[target](), "MLA is only supported on GPU"

    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)

    var scalar_args_buf_tt = rebind[
        TileTensor[.int64, RowMajorLayout[ComptimeInt[3]], MutAnyOrigin]
    ](scalar_args_buf)

    comptime _q_num_heads = type_of(q).static_shape[q.rank - 2]
    comptime _q_head_dim = type_of(q).static_shape[q.rank - 1]

    # @__copy_capture cannot capture Optional[Int] directly; unpack to a
    # (has, value) pair, capture the primitives, then rebuild Optional[Int]
    # inside the closure.
    var has_num_partitions = num_partitions_in.__bool__()
    var num_partitions_val = (
        num_partitions_in.value() if has_num_partitions else 0
    )

    @always_inline
    def _dispatch_mla[
        mask_t: MHAMask
    ](mask: mask_t) raises {
        var k,
        var scalar_args_buf_tt,
        var q_scale_ptr,
        var d_indices,
        var topk_lengths,
        var attn_sink_ptr,
        var extra_k,
        var extra_d_indices,
        var extra_topk_lengths,
        var extra_scales_ptr,
        var has_num_partitions,
        var num_partitions_val,
        var logical_indices,
        imm,
    }:
        var _num_partitions_in: Optional[Int] = Optional[Int](
            num_partitions_val
        ) if has_num_partitions else Optional[Int](None)
        flare_mla_decoding[
            rank=q.rank,
            config=MHAConfig[q_dtype](_q_num_heads, _q_head_dim),
            ragged=True,
            per_token_scale_rope_aware=per_token_scale_rope_aware,
            sparse=sparse_mla,
            fold_shared_index=fold_shared_index,
        ](
            output,
            q,
            k,
            mask,
            input_row_offsets,
            scale,
            context,
            scalar_args_buf=scalar_args_buf_tt,
            q_scale_ptr=q_scale_ptr,
            d_indices=d_indices,
            indices_stride=indices_stride,
            topk_lengths=topk_lengths,
            attn_sink_ptr=attn_sink_ptr,
            extra_k=extra_k,
            extra_d_indices=extra_d_indices,
            extra_indices_stride=extra_indices_stride,
            extra_topk_lengths=extra_topk_lengths,
            extra_scales_ptr=extra_scales_ptr,
            num_partitions_in=_num_partitions_in,
            logical_indices=logical_indices,
        )

    dispatch_mask[
        mask_str,
        local_window_size,
    ](_dispatch_mla)


@always_inline
def generic_flare_mla_prefill_kv_cache_ragged[
    collection_t: KVCollectionT,
    input_dtype: DType,
    dtype: DType,
    //,
    mask_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: TileTensor[mut=False, input_dtype, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, input_dtype, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, input_dtype, address_space=.GENERIC, ...],
    buffer_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Dispatches MLA prefill attention over a ragged batch against a paged KV cache.

    Parameters:
        collection_t: The KV cache collection type storing the K cache for this
            layer (inferred).
        input_dtype: Data type of the input `q`, `k`, and `v` tile tensors
            (inferred).
        dtype: Data type of the output tile tensor (inferred).
        mask_str: Attention mask name selecting the masking strategy, such as
            "causal", "null", or "sliding_window_causal".
        target: Target device string for kernel dispatch; must be a GPU
            target.
        local_window_size: Sliding-window size in tokens for windowed masks;
            -1 for masks that ignore it (defaults to -1).

    Args:
        q: Query tile tensor with shape (total_seq_len, num_heads, q_head_size).
        k: Key tile tensor with shape (total_seq_len, num_heads, kv_head_size).
        v: Value tile tensor with shape (total_seq_len, num_heads, kv_head_size).
        buffer_row_offsets: Tile tensor denoting the start and end position of
            each K entry in the ragged K/V tensor.
        cache_offsets: Mutable tile tensor denoting the start position of each
            K entry in the PagedKVCacheCollection.
        input_row_offsets: Tile tensor denoting the start and end position of
            each Q entry in the batch.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        scale: The scaling factor in scaled dot-product attention, usually
            rsqrt(head_size).
        output: The pre-allocated output tile tensor to write results to, with
            shape (total_seq_len, num_heads, kv_head_size).
        context: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "buffer_row_offsets",
                        coord_to_index_list(
                            buffer_row_offsets.layout.shape_coord()
                        ),
                    ),
                    trace_arg(
                        "cache_offsets",
                        coord_to_index_list(cache_offsets.layout.shape_coord()),
                    ),
                    trace_arg(
                        "input_row_offsets",
                        coord_to_index_list(
                            input_row_offsets.layout.shape_coord()
                        ),
                    ),
                    "scale=" + String(scale),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mla.prefill.ragged."
        + collection_t.name_str
        + "."
        + mask_str
        + ".nhead_"
        + String(collection_t.kv_params.num_heads)
        + ".hdim_"
        + String(collection_t.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return _flare_mla_prefill_kv_cache_ragged[
            mask_str=mask_str,
            target=target,
            local_window_size=local_window_size,
        ](
            q,
            k,
            v,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
        )


@always_inline
def _flare_mla_prefill_kv_cache_ragged[
    input_dtype: DType,
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: TileTensor[mut=False, input_dtype, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, input_dtype, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, input_dtype, address_space=.GENERIC, ...],
    buffer_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Performs MLA prefill.

    Args:
        q: Tensor with shape (total_seq_len, num_heads, q_head_size).
        k: Tensor with shape (total_seq_len, num_heads, kv_head_size).
        v: Tensor with shape (total_seq_len, num_heads, kv_head_size).
        buffer_row_offsets: The start and end position of each K entry in the ragged K/V tensor.
        cache_offsets: The start position of each K entry in the PagedKVCacheCollection.
        input_row_offsets: The start and end position of each Q entry in the batch.
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_collection
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (total_seq_len, num_heads, kv_head_size).
        context: Pointer containing the runtime context for the target device.
    """
    comptime assert is_gpu[target](), "MLA is only supported on GPU"

    var layer_idx_cast = Int(layer_idx)
    var k_rope = kv_collection.get_key_cache(layer_idx_cast)

    # Convert k and v to LayoutTensors for RaggedMHAOperand wrapping.
    var k_lt = k.to_layout_tensor()
    var v_lt = v.to_layout_tensor()

    def _mla_dispatch[
        mask_t: MHAMask
    ](mask: mask_t) raises {var k_rope, var k_lt, var v_lt, imm}:
        flare_mla_prefill[rank=3,](
            output,
            q,
            k_lt,
            v_lt,
            k_rope,
            mask,
            input_row_offsets,
            buffer_row_offsets,
            scale,
            context,
            cache_offsets=LayoutTensor[
                .uint32,
                Layout.row_major(UNKNOWN_VALUE),
                MutAnyOrigin,
            ](
                cache_offsets.ptr.as_unsafe_any_origin(),
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    coord_to_index_list(
                        cache_offsets.layout.shape_coord()
                    ).canonicalize()
                ),
            ),
        )

    dispatch_mask[
        mask_str,
        local_window_size,
    ](_mla_dispatch)


@always_inline
def generic_flare_mla_prefill_ragged_paged_plan[
    target: StaticString
](
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    buffer_token_size: UInt32,
    buffer_row_offsets: LayoutTensor[
        mut=True, .uint32, address_space=.GENERIC, ...
    ],
    cache_offsets: LayoutTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_lengths: LayoutTensor[mut=True, .int32, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Computes the MLA prefill plan for a ragged paged KV cache.

    Populates buffer_row_offsets, cache_offsets, and buffer_lengths from the
    input row offsets and paged key cache so the subsequent MLA prefill kernel
    can gather K/V entries into a contiguous buffer.

    Parameters:
        target: Target device string for kernel dispatch; must be a GPU
            target.

    Args:
        input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each sequence along the ragged sequence dimension.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        buffer_token_size: Token capacity of each prefill buffer row.
        buffer_row_offsets: Mutable output tensor receiving the start and end
            position of each K entry in the ragged K/V buffer.
        cache_offsets: Mutable output tensor receiving the start position of
            each K entry in the PagedKVCacheCollection.
        buffer_lengths: Mutable output tensor receiving the number of valid
            tokens per buffer row.
        context: The call context pointer, passed by the graph compiler.
    """
    comptime assert is_gpu[target](), "Planning MLA is only supported on GPU"

    var layer_idx_cast = Int(layer_idx)

    var k = kv_collection.get_key_cache(layer_idx_cast)

    with Trace[TraceLevel.OP, target=target](
        "mo.mla.prefill.ragged.paged.plan", task_id=Int(context.id())
    ):
        mla_prefill_plan(
            lt_to_tt(buffer_row_offsets),
            lt_to_tt(cache_offsets),
            lt_to_tt(buffer_lengths),
            lt_to_tt(input_row_offsets),
            k,
            buffer_token_size,
            context,
        )


@always_inline
def kv_cache_row_offsets_ragged_paged[
    target: StaticString,
](
    cache_row_offsets: TileTensor[
        mut=True, .uint32, address_space=.GENERIC, ...
    ],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    cache_lengths: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Builds cumulative valid-cache row offsets for a ragged prefill batch.

    Parameters:
        target: Compilation target string; must be a GPU target.

    Args:
        cache_row_offsets: Output tensor of shape (batch_size + 1) receiving
            the cumulative valid-cache row offsets.
        input_row_offsets: Tensor of shape (batch_size + 1) denoting the
            start of each sequence in the ragged batch.
        cache_lengths: Tensor of shape (batch_size,) holding the valid cache
            length for each batch.
        ctx: The call context pointer, passed by the graph compiler.
    """
    comptime assert is_gpu[
        target
    ](), "Building cache row offsets is only supported on GPU"

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    comptime kernel = kv_cache_row_offsets_ragged_paged_kernel[
        cache_row_offsets.LayoutType,
        input_row_offsets.LayoutType,
        cache_lengths.LayoutType,
    ]
    ctx.enqueue_function[kernel](
        cache_row_offsets,
        input_row_offsets,
        cache_lengths,
        grid_dim=max(ceildiv(batch_size, 128), 1),
        block_dim=128,
    )


def kv_cache_row_offsets_ragged_paged_kernel[
    CacheRowOffsetsLayoutType: TensorLayout,
    InputRowOffsetsLayoutType: TensorLayout,
    CacheLengthsLayoutType: TensorLayout,
](
    cache_row_offsets: TileTensor[
        mut=True, .uint32, CacheRowOffsetsLayoutType, MutUntrackedOrigin
    ],
    input_row_offsets: TileTensor[
        .uint32, InputRowOffsetsLayoutType, ImmUntrackedOrigin
    ],
    cache_lengths: TileTensor[
        .uint32, CacheLengthsLayoutType, ImmUntrackedOrigin
    ],
):
    """Computes cumulative valid-cache row offsets for one batch index in a ragged prefill batch.

    Each thread accumulates the running sum of valid cache lengths plus the
    ragged sequence deltas for all batches before its output index and writes
    the result to cache_row_offsets.

    Parameters:
        CacheRowOffsetsLayoutType: Memory layout of the `cache_row_offsets`
            output tensor.
        InputRowOffsetsLayoutType: Memory layout of the `input_row_offsets`
            tensor.
        CacheLengthsLayoutType: Memory layout of the `cache_lengths` tensor.

    Args:
        cache_row_offsets: Output tensor receiving the cumulative valid-cache
            row offsets, with shape (batch_size + 1,).
        input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each sequence along the ragged sequence dimension.
        cache_lengths: Tensor with shape (batch_size,) giving the number of
            valid cached tokens per batch.
    """
    comptime assert cache_row_offsets.flat_rank == 1
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert cache_lengths.flat_rank == 1

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    var output_idx = global_idx.x
    if output_idx > batch_size:
        return

    var running_length = 0
    var prev_input_row_offset = Int(input_row_offsets[0])
    for batch_idx in range(output_idx):
        var row_offset = Int(input_row_offsets[batch_idx + 1])
        running_length += (
            Int(cache_lengths[batch_idx]) + row_offset - prev_input_row_offset
        )
        prev_input_row_offset = row_offset

    cache_row_offsets[output_idx] = UInt32(running_length)


@always_inline
def generic_flare_mla_decompress_k_cache_ragged_paged[
    target: StaticString, dtype: DType
](
    buffer_row_offsets_1d: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    cache_offsets_1d: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    buffer_length: Int32,
    weight: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    k_latent_buffer: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    k_buffer: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
) raises:
    """Decompresses MLA latent K cache rows into a full K buffer via a matmul.

    Gathers compressed latent K vectors from the paged KV cache into
    k_latent_buffer using buffer_row_offsets and cache_offsets, then multiplies
    by the down-projection weight to produce the decompressed K buffer.

    Parameters:
        target: Target device string for kernel dispatch; must be a GPU
            target.
        dtype: Data type of the weight and K buffers (inferred).

    Args:
        buffer_row_offsets_1d: Tensor denoting the start and end position of
            each K entry in the ragged buffer.
        cache_offsets_1d: Tensor denoting the start position of each K entry in
            the PagedKVCacheCollection.
        buffer_length: Number of K rows to decompress.
        weight: Down-projection weight tensor applied to the latent K vectors.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        k_latent_buffer: Mutable output buffer receiving the gathered latent K
            vectors.
        k_buffer: Mutable output buffer receiving the decompressed K vectors.
        context: The call context pointer, passed by the graph compiler.
    """
    comptime assert is_gpu[target](), "MLA is only supported on GPU"

    var buffer_length_int = Int(buffer_length)
    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)

    comptime latent_dim = Int(k_latent_buffer.layout.shape[1])
    var k_latent_tile = TileTensor(
        k_latent_buffer.ptr,
        row_major(buffer_length_int, Idx[latent_dim]),
    )
    _k_cache_to_buffer(
        lt_to_tt(buffer_row_offsets_1d),
        lt_to_tt(cache_offsets_1d),
        k,
        Int32(buffer_length_int),
        k_latent_tile,
        context,
    )

    # rebind k_latent_buffer with dynamic dim
    comptime latent_last_dim = Int(k_latent_buffer.layout.shape[1])
    comptime k_latent_layout = Layout.row_major(UNKNOWN_VALUE, latent_last_dim)
    var k_latent_dynamic_shape = IndexList[2](
        buffer_length_int, latent_last_dim
    )

    var k_latent_buffer_dynamic = LayoutTensor[dtype, k_latent_layout](
        k_latent_buffer.ptr,
        RuntimeLayout[k_latent_layout].row_major(k_latent_dynamic_shape),
    )

    # rebind k_buffer with dynamic dim
    comptime k_last_dim = Int(k_buffer.layout.shape[1])
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, k_last_dim)
    var k_dynamic_shape = IndexList[2](buffer_length_int, k_last_dim)

    var k_buffer_dynamic = LayoutTensor[dtype, k_layout](
        k_buffer.ptr, RuntimeLayout[k_layout].row_major(k_dynamic_shape)
    )

    matmul[target=target, transpose_b=True](
        lt_to_tt(k_buffer_dynamic),
        lt_to_tt(k_latent_buffer_dynamic),
        lt_to_tt(weight),
        Optional(context),
    )


# ===-----------------------------------------------------------------------===#
# Cross attention (ragged)
# ===-----------------------------------------------------------------------===#


def _cross_attention_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = dtype,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    q_input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    q_max_seq_len: UInt32,
    kv_input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    def _dispatch_flash_attention[
        mask_t: MHAMask
    ](mask: mask_t) raises {
        var q,
        var k,
        var v,
        var output,
        var q_input_row_offsets,
        var kv_input_row_offsets,
        imm,
    }:
        comptime if is_cpu[target]():
            comptime assert output_dtype == dtype, (
                "CPU flash attention requires output dtype == q dtype;"
                " the distinct-output-dtype (fp8->bf16) path is GPU-only."
            )
            return flash_attention_kv_cache_cpu(
                q,
                q_input_row_offsets,
                # Use KV offsets for cross attention.
                kv_input_row_offsets,
                k,
                v,
                mask,
                scale,
                output.bitcast[dtype](),
                sink_weights,
            )
        else:
            gpu_flash_attention[ragged=True, sink=False](
                output,
                q,
                k,
                v,
                mask,
                q_input_row_offsets,
                scale,
                context,
                Int(q_max_seq_len),
                LayoutTensor[
                    kv_input_row_offsets.dtype,
                    Layout.row_major(UNKNOWN_VALUE),
                    ImmutAnyOrigin,
                ](
                    kv_input_row_offsets.ptr.as_imm().as_unsafe_any_origin(),
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                        kv_input_row_offsets.runtime_layout.shape.value.canonicalize()
                    ),
                ),
                None,
            )

    return dispatch_mask[
        mask_str,
        local_window_size,
    ](_dispatch_flash_attention)


@always_inline
def generic_cross_attention_kv_cache[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = dtype,
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    q_input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    q_max_seq_len: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    """Dispatches cross-attention flash attention over a ragged batch against a paged KV cache.

    Parameters:
        collection_t: The KV cache collection type storing the K and V caches
            for this layer (inferred).
        dtype: Data type of the query tensor (inferred).
        target: Target device string for kernel dispatch, such as "cpu" or
            "gpu".
        mask_str: Attention mask name selecting the masking strategy, such as
            "causal", "null", or "sliding_window_causal".
        local_window_size: Sliding-window size in tokens for windowed masks;
            -1 for masks that ignore it (defaults to -1).
        output_dtype: Data type of the output tensor; defaults to `dtype`.

    Args:
        q: Query tensor with shape (sum(q_seq_lens), num_heads, head_size).
        q_input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each query sequence along the ragged sequence dimension.
        q_max_seq_len: Scalar tensor holding the maximum query sequence length.
        kv_input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each KV sequence along the ragged sequence dimension.
        kv_collection: The collection storing the KVCache entries for this
            layer, retrieved via layer_idx.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from kv_collection.
        scale: The scaling factor in scaled dot-product attention, usually
            rsqrt(head_size).
        output: The pre-allocated output buffer to write results to, with shape
            (sum(q_seq_lens), num_heads, head_size).
        context: The call context pointer, passed by the graph compiler.
        sink_weights: Optional per-batch attention sink weights applied to the
            leading cache slots.
    """

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg(
                        "q_input_row_offsets",
                        q_input_row_offsets.runtime_layout.shape.value,
                    ),
                    trace_arg(
                        "kv_input_row_offsets",
                        kv_input_row_offsets.runtime_layout.shape.value,
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.cross_attention.ragged."
        + collection_t.name_str
        + "."
        + mask_str
        + ".nhead_"
        + String(collection_t.kv_params.num_heads)
        + ".hdim_"
        + String(collection_t.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return _cross_attention_dispatch[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q,
            q_input_row_offsets,
            q_max_seq_len[0][0],
            kv_input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            sink_weights,
        )


# ===-----------------------------------------------------------------------===#
# KV cache ragged radd dispatch
# ===-----------------------------------------------------------------------===#


def generic_kv_cache_radd_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    target: StaticString,
](
    a: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    cache: collection_t,
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    batch_offset: UInt32,
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """Adds an input tensor elementwise into the paged KV cache in-place.

    Splits the input tensor's last dimension into key and value halves and
    accumulates each half into the corresponding K or V cache slot, applying
    the batch offset and per-batch cache lengths to locate the target rows.

    Parameters:
        dtype: Element type of the input tensor `a` and the KV cache entries
            (inferred).
        collection_t: Concrete `KVCollectionT` type of the `cache` argument,
            used to recover the cache's static parameters and cache type
            (inferred).
        target: Compilation target string used to dispatch GPU versus CPU
            paths.

    Args:
        a: Input tensor with shape (sum(seq_lens), 2 * hidden_size) where the
            first hidden_size columns target K and the rest target V.
        cache: The collection storing the KVCache entries for this layer,
            retrieved via layer_idx.
        input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each sequence along the ragged sequence dimension.
        batch_offset: Offset added to the computed batch index to support
            batch slicing.
        layer_idx: The index of the layer being executed, used to retrieve the
            KVCache objects from cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    comptime hidden_size = collection_t.kv_params.head_size * collection_t.kv_params.num_heads

    comptime assert (
        dtype == collection_t.dtype
    ), "Mismatch in dtype between computation and KV tensors"
    comptime assert (
        a.layout.shape[1] != UNKNOWN_VALUE
    ), "Input tensor must have known shape in last dim"
    comptime assert Int(a.layout.shape[1]) == hidden_size * 2, (
        "Mismatch in hidden size between input "
        + String(Int(a.layout.shape[1]))
        + " and KV tensors "
        + String(hidden_size)
    )

    var layer_idx_cast = Int(layer_idx)
    var k_cache = cache.get_key_cache(layer_idx_cast)
    var v_cache = cache.get_value_cache(layer_idx_cast)

    # TODO: This elementwise body captures KV cache views (`CacheType`), which
    # fail codegen when stored into a unified closure ('pop.store' pointer
    # element-type verification). Keep using the deprecated parameter-closure
    # overload until cache captures in unified closures are supported.
    @__parameter
    @__copy_capture(k_cache, v_cache, input_row_offsets)
    def do_radd[width: Int, alignment: Int = 1](idx: Coord):
        comptime assert idx.rank == 2, "Rank must be 2"

        # we could be slicing the batch, so we need to add the offset to get the actual index in the flattened batch
        var corrected_token_idx = UInt32(idx[0].value()) + input_row_offsets[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, Int(corrected_token_idx)
        )

        # we also need to add the batch offset to get the actual index in the flattened batch
        var corrected_batch_idx = UInt32(batch_idx) + batch_offset
        var tok_idx = Int(corrected_token_idx - input_row_offsets[batch_idx])

        var cache: collection_t.CacheType
        var corrected_dim: Int
        if Int(idx[1].value()) < hidden_size:
            cache = k_cache
            corrected_dim = Int(idx[1].value())
        else:
            cache = v_cache
            corrected_dim = Int(idx[1].value()) - hidden_size

        var h_idx: Int
        var hd_idx: Int
        h_idx, hd_idx = udivmod(corrected_dim, collection_t.kv_params.head_size)

        var cache_length = cache.cache_length(Int(corrected_batch_idx))
        var cache_token_idx = tok_idx + cache_length

        var old_val = cache.load[width=width](
            Int(corrected_batch_idx), h_idx, cache_token_idx, hd_idx
        )
        var a_val = rebind[type_of(old_val)](
            a.load[width=width](coord_to_index_list(idx))
        )

        cache.store(
            Int(corrected_batch_idx),
            h_idx,
            cache_token_idx,
            hd_idx,
            a_val + old_val,
        )

    comptime if is_gpu[target]():
        comptime compile_target = get_gpu_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[do_radd, simd_width, target=target](
            Coord(a.runtime_layout.shape.value), ctx
        )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[do_radd, simd_width, target=target](
            Coord(a.runtime_layout.shape.value), ctx
        )


def kv_cache_store_ragged[
    cache_t: KVCacheT,
    //,
    target: StaticString,
    input_fn: def[width: Int, alignment: Int](
        idx: IndexList[3]
    ) capturing -> SIMD[cache_t.dtype, width],
](
    cache: cache_t,
    input_shape: IndexList[3],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    """Stores ragged input values into a paged KV cache via an elementwise kernel.

    Invokes the supplied input_fn to load values and writes them into the cache
    at positions determined by the per-batch cache lengths and input row
    offsets.

    Parameters:
        cache_t: The KV cache type used to store key or value entries
            (inferred).
        target: Compilation target string used to dispatch GPU versus CPU
            paths.
        input_fn: Compile-time callback that loads a SIMD vector of
            cache-typed elements at the given 3D index.

    Args:
        cache: The KVCache object to write key or value entries into.
        input_shape: Shape of the input as a 3D index list (tokens, heads,
            head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,) denoting the
            start of each sequence along the ragged sequence dimension.
        context: The call context pointer, passed by the graph compiler.
    """
    comptime assert input_row_offsets.layout.rank() == 1, (
        "Expected input_row_offsets to be a 1D tensor of shape `(batch_size"
        " + 1,)`"
    )

    # TODO: This elementwise body captures a KV cache view (`CacheType`), which
    # fails codegen when stored into a unified closure ('pop.store' pointer
    # element-type verification). Keep using the deprecated parameter-closure
    # overload until cache captures in unified closures are supported.
    @__parameter
    @__copy_capture(cache, input_row_offsets)
    def write_to_cache[
        width: Int,
        alignment: Int = 1,
    ](idx: Coord) capturing:
        var input_idx = IndexList[3](
            Int(idx[0].value()), Int(idx[1].value()), Int(idx[2].value())
        )
        var loaded_val = input_fn[width=width, alignment=alignment](input_idx)
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, Int(idx[0].value())
        )
        var token_idx = Int(
            UInt32(idx[0].value()) - input_row_offsets[batch_idx]
        )
        var h_idx = Int(idx[1].value())
        var hd_idx = Int(idx[2].value())
        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            loaded_val,
        )

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime simd_width = simd_width_of[cache_t.dtype, target=compile_target]()

    elementwise[
        write_to_cache,
        simd_width,
        target=target,
        _trace_description="kv_cache_store_ragged",
    ](Coord(input_shape), context)


def kv_cache_store_padded[
    cache_t: KVCacheT,
    //,
    target: StaticString,
    input_fn: def[width: Int, alignment: Int](
        idx: IndexList[4]
    ) capturing -> SIMD[cache_t.dtype, width],
](
    cache: cache_t,
    input_shape: IndexList[4],
    valid_lengths: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    """Stores padded input values into a paged KV cache via an elementwise kernel.

    Invokes the supplied input_fn to load values and writes them into the cache
    at positions determined by the per-batch cache lengths, skipping tokens
    beyond each batch's valid length.

    Parameters:
        cache_t: The KV cache type used to store key or value entries
            (inferred).
        target: Compilation target string used to dispatch GPU versus CPU
            paths.
        input_fn: Compile-time callback that loads a SIMD vector of
            cache-typed elements at the given 4D index.

    Args:
        cache: The KVCache object to write key or value entries into.
        input_shape: Shape of the input as a 4D index list (batch, tokens,
            heads, head_size).
        valid_lengths: Tensor with shape (batch_size,) giving the number of
            valid tokens per batch; rows beyond this are skipped.
        context: The call context pointer, passed by the graph compiler.
    """
    comptime assert (
        valid_lengths.layout.rank() == 1
    ), "Expected valid_lengths to be a 1D tensor of shape `(batch_size,)`"

    # TODO: This elementwise body captures a KV cache view (`CacheType`), which
    # fails codegen when stored into a unified closure ('pop.store' pointer
    # element-type verification). Keep using the deprecated parameter-closure
    # overload until cache captures in unified closures are supported.
    @__parameter
    @__copy_capture(cache, valid_lengths)
    @always_inline
    def write_to_cache[width: Int, alignment: Int = 1](idx: Coord) capturing:
        var batch_idx = Int(idx[0].value())
        var token_idx = Int(idx[1].value())
        var valid_len = Int(valid_lengths[batch_idx])
        if token_idx >= valid_len:
            return
        var input_idx = IndexList[4](
            batch_idx,
            token_idx,
            Int(idx[2].value()),
            Int(idx[3].value()),
        )
        var loaded_val = input_fn[width=width, alignment=alignment](input_idx)
        var h_idx = Int(idx[2].value())
        var hd_idx = Int(idx[3].value())
        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            loaded_val,
        )

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime simd_width = simd_width_of[cache_t.dtype, target=compile_target]()

    elementwise[
        write_to_cache,
        simd_width,
        target=target,
        _trace_description="kv_cache_store_padded",
    ](Coord(input_shape), context)


# ===-----------------------------------------------------------------------===#
# KV cache ragged 2M iadd dispatch
# ===-----------------------------------------------------------------------===#


def kv_cache_2m_iadd_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    target: StaticString,
](
    kv: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    cache: collection_t,
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    lora_end_idx: LayoutTensor[mut=False, .int64, address_space=.GENERIC, ...],
    batch_seq_len: LayoutTensor[mut=False, .int64, address_space=.GENERIC, ...],
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """
    In-place add to paged KV cache with concatenated K/V layout. This kernel is
    only used for LoRA.

    Performs an in-place addition of new key-value projections to paged KV cache.
    The input tensor `a` uses a "2m" layout where keys and values are concatenated:
    rows [0, m) contain keys and rows [m, 2m) contain values, where m is the number
    of tokens. We use the `lora_end_idx` to index into the K or V tensor.
    We call this value `m` since this value will be a subset of the
    total tokens in the batch. We write tokens to K as [0, m) and V as [m, 2m).

    Parameters:
        dtype: Element type of the `kv` input tensor and the KV cache entries
            (inferred).
        collection_t: The KV cache collection type (inferred).
        target: Compilation target string used to dispatch GPU versus CPU
            paths.

    Args:
        kv: Input tensor with concatenated K/V layout of shape
            (2m, num_heads * head_size), where rows [0, m) hold keys and rows
            [m, 2m) hold values.
        cache: The KV cache collection storing the key and value caches for
            this layer, retrieved via `layer_idx`.
        input_row_offsets: Tensor with shape (batch_size + 1) denoting the
            start of each sequence along the ragged sequence dimension.
        lora_end_idx: Single-element tensor holding `m`, the number of LoRA
            tokens to add; keys occupy rows [0, m) and values occupy rows
            [m, 2m) of `kv`.
        batch_seq_len: Single-element tensor holding the total number of
            tokens in the batch.
        layer_idx: The index of the layer being executed, used to retrieve the
            K and V caches from `cache`.
        ctx: The call context pointer, passed by the graph compiler.
    """
    comptime hidden_size = collection_t.kv_params.head_size * collection_t.kv_params.num_heads
    var kv_shape = kv.runtime_layout.shape.value.canonicalize()
    comptime assert (
        dtype == collection_t.dtype
    ), "Mismatch in dtype between computation and KV tensors"
    comptime assert (
        kv.layout.shape[1] != UNKNOWN_VALUE
    ), "Input tensor must have known shape in last dim"
    comptime assert Int(kv.layout.shape[1]) == hidden_size, (
        "Mismatch in hidden size between input "
        + String(Int(kv.layout.shape[1]))
        + " and KV tensors "
        + String(hidden_size)
    )

    var layer_idx_cast = Int(layer_idx)
    var k_cache = cache.get_key_cache(layer_idx_cast)
    var v_cache = cache.get_value_cache(layer_idx_cast)
    var m = Int(lora_end_idx[0])
    var M = Int(batch_seq_len[0])

    # [2m, N]
    var elementwise_shape = IndexList[2](2 * m, kv_shape[1])

    # TODO: This elementwise body captures KV cache views (`CacheType`), which
    # fail codegen when stored into a unified closure ('pop.store' pointer
    # element-type verification). Keep using the deprecated parameter-closure
    # overload until cache captures in unified closures are supported.
    @__parameter
    @__copy_capture(kv, k_cache, v_cache, input_row_offsets, m, M)
    def iadd[width: Int, alignment: Int = 1](idx: Coord):
        comptime assert idx.rank == 2, "Rank must be 2"

        var cache: collection_t.CacheType
        var row_idx: Int

        if Int(idx[0].value()) < m:
            cache = k_cache
            row_idx = Int(idx[0].value())
        else:
            cache = v_cache
            row_idx = Int(idx[0].value()) - m

        var batch_idx = get_batch_from_row_offsets(input_row_offsets, row_idx)
        var tok_idx = Int(UInt32(row_idx) - input_row_offsets[batch_idx])

        var h_idx: Int
        var hd_idx: Int
        h_idx, hd_idx = udivmod(
            Int(idx[1].value()), collection_t.kv_params.head_size
        )

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = tok_idx + cache_length

        var old_val = cache.load[width=width](
            batch_idx, h_idx, cache_token_idx, hd_idx
        )
        var a_val = rebind[type_of(old_val)](
            kv.load[width=width](Int(idx[0].value()), Int(idx[1].value()))
        )

        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            a_val + old_val,
        )

    comptime if is_gpu[target]():
        with Trace[TraceLevel.OP, target=target](
            "kv-cache-2m-iadd",
            task_id=Int(ctx.id()),
        ):
            comptime compile_target = get_gpu_target()
            comptime simd_width = simd_width_of[dtype, target=compile_target]()

            elementwise[iadd, simd_width, target=target](
                Coord(elementwise_shape), ctx
            )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[iadd, simd_width, target=target](
            Coord(elementwise_shape), ctx
        )
