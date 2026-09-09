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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import Protocol, overload

import max._core
import max._core.dialects.builtin
import max._core.dialects.kgen
import max._core.dialects.m
import max._core.dialects.mosh
import max._core.dtype
from max.mlir import Location

from . import passes as passes

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

class BufferType(max._core.Type):
    """
    This is a close analogue of the existing mo.tensor type but is meant
    to represent tensors that can be mutated.

    In conjunction with the operations mo.mutable.load and mo.mutable.store
    this type can be used to model in-place operations in the MO dialect.

    The `shape` is less permissive than the equivalent for `!mo.tensor`
    values and must be a `MOSH::ShapeAttr` (i.e. statically ranked).

    The element type is an M::DType, with `invalid` denoting an unknown type.

    Examples:
    ```mlir
    !mo.buffer<[4, 16], f32>    // static shape
    !mo.buffer<[N, N, 6], i32>  // parameterized shape
    ```
    """

    @overload
    def __init__(self, tensor_type: TensorType) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.mosh.ShapeAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.mosh.ShapeAttr,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[max._core.dialects.builtin.TypedAttr],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.mosh.ShapeAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr,
        metadata: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @property
    def shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def device_ref(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @property
    def metadata(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class BundleType(max._core.Type):
    """
    A grouping type that bundles multiple per-device tensors into a single SSA
    value.  All elements must be `!mo.tensor` types.  Elements may have
    different devices, shapes, or dtypes.

    The type has two interchangeable surface syntaxes. The verbose form lists
    each element tensor explicitly and is used when the elements are not
    uniform:

    ```mlir
    !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]>
    ```

    The compact form is used when every element shares the same shape, dtype,
    device label, and metadata, and the device IDs are strictly ordered and
    contiguous (`ids[i] == ids[0] + i`). It names the shared attributes once
    and the device range as `label:firstId-lastId`:

    ```mlir
    !mo.bundle<[3], f32, gpu:0-1>
    ```

    A single-element compact bundle omits the range suffix. A single-element
    bundle on the default host device also omits the device section entirely:

    ```mlir
    !mo.bundle<[3], f32, gpu:0>   // single element, explicit device
    !mo.bundle<[3], f32>          // single element, default host device
    ```

    The compact form is purely syntactic sugar; the stored `elementTypes` are
    identical regardless of which form is parsed.
    """

    def __init__(self, element_types: Sequence[max._core.Type]) -> None: ...
    @property
    def element_types(self) -> Sequence[max._core.Type]: ...

class ChainType(max._core.Type):
    """
    This type is used to sequence side-effecting operations. Any operation in
    the MO dialect that has side-effects should both consume and produce a
    chain type.
    """

    def __init__(self) -> None: ...

class OpaqueType(max._core.Type):
    """
    This is a custom user-defined type.
      Example:
      ```mlir
        !mo.opaque<"my_list">
        !mo.opaque<"my_list", {foo = 42}>
      ```
    """

    @overload
    def __init__(
        self, symbol: max._core.dialects.builtin.StringAttr
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.StringAttr,
        parameters: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.StringAttr,
        parameters: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @property
    def symbol(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def parameters(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class ScalarType(max._core.Type):
    """This type represents scalars."""

    def __init__(self, dtype: max._core.dtype.DType) -> None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...

class TensorType(max._core.Type):
    """
    This type represents the shape and element type of a tensor, an optional
    device ref, and an optional dictionary of metadata (e.g., layout, etc.).

    The `shape` is always a `MOSH::ShapeAttr` (e.g., `[D0, 42, N]`). Rank
    is statically known; individual dimensions may be concrete integers or
    parametric.

    The element type is an M::DType, with `invalid` denoting an unknown type.
    The type implements a subset of the methods in ShapedTypeInterface.

    The `deviceRef` optional field denotes the device the tensor lives on.

    Examples:
    ```mlir
    !mo.tensor<[4, 16], f32>           // static shape
    !mo.tensor<[N, N, 6], i32>         // parameterized shape
    !mo.tensor<[1, M, N], i32>         // partially known and parameterized shape
    !mo.tensor<[4, 16], f32, gpu:0>    // optional device
    ```
    """

    @overload
    def __init__(
        self,
        shape: max._core.dialects.mosh.ShapeAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.mosh.ShapeAttr,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[max._core.dialects.builtin.TypedAttr],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: max._core.dialects.mosh.ShapeAttr,
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr,
        metadata: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[max._core.dialects.builtin.TypedAttr],
        dtype: max._core.dtype.DType,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
        metadata: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def device_ref(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @property
    def metadata(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class DTypeAttr(max._core.Attribute):
    """This attribute holds the data type of a tensor."""

    def __init__(self, dtype: max._core.dtype.DType) -> None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...

class LayoutAttr(max._core.Attribute):
    """
    This attribute holds the memory layout information for some tensor value.
    """

    @overload
    def __init__(
        self, format: max._core.dialects.builtin.StringAttr
    ) -> None: ...
    @overload
    def __init__(self, format_str: str) -> None: ...
    @property
    def format(self) -> max._core.dialects.builtin.StringAttr: ...

class CompositeDistributedAllgatherRmsNormOp(max._core.Operation):
    """
    AllGather concatenates the per-device row shards (`inputs`) so every device
    holds the full replicated tensor, then RMSNorms it in the same launch (no
    separate-norm HBM round-trip). Returns the normed tensor (`output`) and the
    raw gathered residual (`outResidual`); a gathered row is a verbatim copy, so
    the residual is bit-identical to a standalone all-gather (no f32 peer-sum).
    `multiply_before_cast=true`, bf16 in/out only (no quantization).

    `group_size` matches `mo.distributed.allgather`: the devices split into
    contiguous groups of that many, each gathering independently, so the op
    works under TP-within-DP topologies and every output is the group's gathered
    tensor rather than the whole world's. It must be at least 2 and must divide
    the device count; `group_size == num_devices` is a full-world collective.
    The builder always sets it -- the `0` attribute default is not a usable
    value.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: Sequence[max._core.Type],
        out_residual: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        gamma: Sequence[max._core.Value[max._core.Type]],
        epsilon: Sequence[max._core.Value[max._core.Type]],
        weight_offset: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        group_size: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def gamma(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def epsilon(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def weight_offset(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class CompositeDistributedAllgatherRmsNormQuantMxfp8Op(max._core.Operation):
    """
    `mo.composite.distributed.allgather_rms_norm` plus an MXFP8 copy of the
    normed `output`: `outQuant` (float8_e4m3fn, same shape) and `outScale`
    (float8_e8m0fnu, `[rows, cols / 32]`, plain rank-2 row-major -- what
    `block_scaled_matmul_amd` takes as `a_scales`, NOT the SM100 SF-atom
    interleave and NOT the preshuffled atom order `block_scaled_matmul_amd_preb`
    requires). Quantized from the bf16 written to `output`, so byte-identical
    to a standalone quantize. Same `group_size` contract.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: Sequence[max._core.Type],
        out_quant: Sequence[max._core.Type],
        out_scale: Sequence[max._core.Type],
        out_residual: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        gamma: Sequence[max._core.Value[max._core.Type]],
        epsilon: Sequence[max._core.Value[max._core.Type]],
        weight_offset: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        group_size: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def gamma(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def epsilon(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def weight_offset(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class CompositeDistributedReduceScatterRmsNormOp(max._core.Operation):
    """
    ReduceScatter takes in inputs each coming from a different device and
    partitions the reduction so each device receives a disjoint row shard of the
    sum. This op keeps that shard's sum in f32 registers and RMSNorm-normalizes
    it in the same launch, avoiding the HBM round-trip of a separate norm kernel.

    It returns both the normed shard (`output`, fed to the next layer) and the
    reduce-scatter sum shard (`outResidual`, the residual stream). The norm is
    inherently `multiply_before_cast=true`: gamma is folded in f32 and the value
    is cast to the input dtype once, last. bf16 in/out only (no quantization).

    When `has_residual` is true, `residuals` is added to the sum in f32 before
    the pre-norm round, each device adding only its own row shard. It must be
    REPLICATED -- bit-identical on every rank of a group -- which is what lets
    a per-rank add reproduce the leader-side pre-add it replaces.

    When false it is ignored and the op is a plain reduce-scatter + norm. The
    operands stay present and group-sized (the variadic groups must all match
    in size), filled with the inputs and never indexed -- the same convention
    as `has_residual` on `mo.composite.distributed.matmul_reduce_scatter.sum`.

    `group_size` matches `mo.distributed.reducescatter.sum`: the devices split
    into contiguous groups of that many, each reducing independently, so the op
    works under TP-within-DP topologies. It must be at least 2 and must divide
    the device count; `group_size == num_devices` is a full-world collective.
    The builder always sets it -- the `0` attribute default is not a usable
    value.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: Sequence[max._core.Type],
        out_residual: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        gamma: Sequence[max._core.Value[max._core.Type]],
        epsilon: Sequence[max._core.Value[max._core.Type]],
        weight_offset: Sequence[max._core.Value[max._core.Type]],
        residuals: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        group_size: max._core.dialects.builtin.IntegerAttr,
        has_residual: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def gamma(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def epsilon(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def weight_offset(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def residuals(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def has_residual(self) -> bool: ...
    @has_residual.setter
    def has_residual(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class CompositeMatmulAddOp(max._core.Operation):
    """
    Computes C = A @ B (optionally transposed) + residual, fusing the matmul
    and the residual addition into a single kernel call.

    `residual` may be rank-2 (same shape as the output, element-wise add) or
    rank-1 (broadcast along the row dimension, i.e. a bias vector).

    This operation is currently only lowered for SM100 (B200) targets, where
    the residual is passed directly to the matmul kernel as an epilogue tensor.

    Example (2-D residual):

    ```mlir
      %res = mo.composite.matmul_add(%a, %b, %residual) {transpose_b = true} : (
        !mo.tensor<[4, 512], bf16>,
        !mo.tensor<[1536, 512], bf16>,
        !mo.tensor<[4, 1536], bf16>
      ) -> !mo.tensor<[4, 1536], bf16>
    ```

    Example (1-D bias broadcast):

    ```mlir
      %res = mo.composite.matmul_add(%a, %b, %bias) {transpose_b = true} : (
        !mo.tensor<[4, 512], bf16>,
        !mo.tensor<[1536, 512], bf16>,
        !mo.tensor<[1536], bf16>
      ) -> !mo.tensor<[4, 1536], bf16>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_a: max._core.Value[TensorType],
        input_b: max._core.Value[TensorType],
        residual: max._core.Value[TensorType],
        transpose_b: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def input_a(self) -> max._core.Value[TensorType]: ...
    @property
    def input_b(self) -> max._core.Value[TensorType]: ...
    @property
    def residual(self) -> max._core.Value[TensorType]: ...
    @property
    def transpose_b(self) -> bool: ...
    @transpose_b.setter
    def transpose_b(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class CompositeGroupedMatmulBlockScaledOp(max._core.Operation):
    """
    The down leg of an NVFP4 MoE FFN: a per-expert (ragged) block-scaled grouped
    matmul of the packed-NVFP4 activations by the down weights, producing the
    bf16 hidden-state output.

    Composite form of the `mo.composite.grouped_matmul_block_scaled` kernel;
    lowers 1:1.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        hidden_states: max._core.Value[TensorType],
        weight: max._core.Value[TensorType],
        a_scales: max._core.Value[TensorType],
        b_scales: max._core.Value[TensorType],
        expert_start_indices: max._core.Value[TensorType],
        expert_ids: max._core.Value[TensorType],
        a_scale_offsets: max._core.Value[TensorType],
        expert_scales: max._core.Value[TensorType],
        estimated_total_m: max._core.Value[TensorType],
        num_active_experts: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def hidden_states(self) -> max._core.Value[TensorType]: ...
    @property
    def weight(self) -> max._core.Value[TensorType]: ...
    @property
    def a_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def b_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def expert_start_indices(self) -> max._core.Value[TensorType]: ...
    @property
    def expert_ids(self) -> max._core.Value[TensorType]: ...
    @property
    def a_scale_offsets(self) -> max._core.Value[TensorType]: ...
    @property
    def expert_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def estimated_total_m(self) -> max._core.Value[TensorType]: ...
    @property
    def num_active_experts(self) -> max._core.Value[TensorType]: ...

class CompositeGroupedMatmulSwigluNvfp4Op(max._core.Operation):
    """
    The gate-up leg of an NVFP4 MoE FFN: a per-expert (ragged) grouped matmul of
    the packed-NVFP4 activations by the gate-up weights, followed by a SwiGLU
    activation and a bf16->nvfp4 quantization of the result. Returns the packed
    NVFP4 activations for the down leg (`c_packed`) and their per-expert SwiGLU
    scale tile (`c_swiglu_scales`).

    `swiglu_alpha`/`swiglu_limit` are host scalars parameterizing the clamped
    SwiGLU (swigluoai) activation, enabled by the `clamp_activation` attribute.

    Composite form of the `mo.composite.grouped_matmul_swiglu_nvfp4` kernel;
    lowers 1:1.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        c_packed: TensorType,
        c_swiglu_scales: TensorType,
        hidden_states: max._core.Value[TensorType],
        weight: max._core.Value[TensorType],
        a_scales: max._core.Value[TensorType],
        b_scales: max._core.Value[TensorType],
        expert_start_indices: max._core.Value[TensorType],
        expert_ids: max._core.Value[TensorType],
        a_scale_offsets: max._core.Value[TensorType],
        expert_scales: max._core.Value[TensorType],
        c_input_scales: max._core.Value[TensorType],
        estimated_total_m: max._core.Value[TensorType],
        num_active_experts: max._core.Value[TensorType],
        swiglu_alpha: max._core.Value[TensorType],
        swiglu_limit: max._core.Value[TensorType],
        clamp_activation: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def hidden_states(self) -> max._core.Value[TensorType]: ...
    @property
    def weight(self) -> max._core.Value[TensorType]: ...
    @property
    def a_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def b_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def expert_start_indices(self) -> max._core.Value[TensorType]: ...
    @property
    def expert_ids(self) -> max._core.Value[TensorType]: ...
    @property
    def a_scale_offsets(self) -> max._core.Value[TensorType]: ...
    @property
    def expert_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def c_input_scales(self) -> max._core.Value[TensorType]: ...
    @property
    def estimated_total_m(self) -> max._core.Value[TensorType]: ...
    @property
    def num_active_experts(self) -> max._core.Value[TensorType]: ...
    @property
    def swiglu_alpha(self) -> max._core.Value[TensorType]: ...
    @property
    def swiglu_limit(self) -> max._core.Value[TensorType]: ...
    @property
    def clamp_activation(self) -> bool: ...
    @clamp_activation.setter
    def clamp_activation(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class CompositeLayerNormRopeRaggedOp(max._core.Operation):
    """
    Fused operation computing LayerNorm followed by ragged RoPE applied to a
    leading slice of the normalized output, with the remaining columns passed
    through unrotated:

      normed = layer_norm(input, gamma, beta, epsilon)
      roped, passthrough = split(normed, axis=-1)
      roped = rope.ragged(roped, input_row_offsets, start_pos, freqs_cis)
      result = concat(roped, passthrough, axis=-1)

    The RoPE width is taken from `freqsCis`'s last dimension.

    Example:

    ```mlir
      %result = mo.composite.layer_norm_rope_ragged(%input, %gamma, %beta,
                                                     %epsilon, %row_offsets,
                                                     %start_pos, %freqs_cis)
        {interleaved = false} :
        (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], f32, gpu:0>,
         !mo.tensor<[128], f32, gpu:0>, !mo.tensor<[], f32>,
         !mo.tensor<[batch_plus_one], ui32, gpu:0>, !mo.tensor<[batch], ui32, gpu:0>,
         !mo.tensor<[1024, 64], f32, gpu:0>)
        -> !mo.tensor<[8, 128], bf16, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        gamma: max._core.Value[TensorType],
        beta: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        input_row_offsets: max._core.Value[TensorType],
        start_pos: max._core.Value[TensorType],
        freqs_cis: max._core.Value[TensorType],
        interleaved: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma(self) -> max._core.Value[TensorType]: ...
    @property
    def beta(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def input_row_offsets(self) -> max._core.Value[TensorType]: ...
    @property
    def start_pos(self) -> max._core.Value[TensorType]: ...
    @property
    def freqs_cis(self) -> max._core.Value[TensorType]: ...
    @property
    def interleaved(self) -> bool: ...
    @interleaved.setter
    def interleaved(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CompositeMaskedFlashAttentionGpuOp(max._core.Operation):
    """
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale + mask) @ V`)
    on GPU. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `masked_flash_attention_gpu` kernel.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        query: max._core.Value[TensorType],
        key: max._core.Value[TensorType],
        value: max._core.Value[TensorType],
        mask: max._core.Value[TensorType],
        scale: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def query(self) -> max._core.Value[TensorType]: ...
    @property
    def key(self) -> max._core.Value[TensorType]: ...
    @property
    def value(self) -> max._core.Value[TensorType]: ...
    @property
    def mask(self) -> max._core.Value[TensorType]: ...
    @property
    def scale(self) -> max._core.Value[TensorType]: ...

class CompositeRmsNormResidualAddOp(max._core.Operation):
    """
    Fused operation computing:
      intermediate = input + residual_input
      output = rms_norm(intermediate, gamma, epsilon, weight_offset)

    Returns both the final normalized output and the post-add intermediate
    tensor. This is the canonical transformer/mamba pre-norm boundary
    `rms_norm(residual + out)`, where the pre-add value is carried forward as
    the next block's residual. Unlike `rms_norm_fused_residual_add`, there is a
    single RMS norm (no inner norm on `input`).

    Example:

    ```mlir
      %output, %intermediate = mo.composite.rms_norm_residual_add(
          %input, %residual, %gamma, %eps, %offset) {
          multiply_before_cast = false} :
        (...) -> (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        intermediate: TensorType,
        input: max._core.Value[TensorType],
        residual_input: max._core.Value[TensorType],
        gamma: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        weight_offset: max._core.Value[TensorType],
        multiply_before_cast: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def residual_input(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def weight_offset(self) -> max._core.Value[TensorType]: ...
    @property
    def multiply_before_cast(self) -> bool: ...
    @multiply_before_cast.setter
    def multiply_before_cast(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CompositeRmsNormFusedResidualAddOp(max._core.Operation):
    """
    Fused operation computing:
      intermediate = rms_norm(input, gamma1, epsilon1, weight_offset1) + residual_input
      output = rms_norm(intermediate, gamma2, epsilon2, weight_offset2)

    Returns both the final output and the post-add intermediate tensor.

    Example:

    ```mlir
      %output, %intermediate = mo.composite.rms_norm_fused_residual_add(
          %input, %residual, %gamma1, %gamma2, %eps1, %eps2, %offset1, %offset2) {
          multiply_before_cast1 = false, multiply_before_cast2 = false} :
        (...) -> (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        intermediate: TensorType,
        input: max._core.Value[TensorType],
        residual_input: max._core.Value[TensorType],
        gamma1: max._core.Value[TensorType],
        gamma2: max._core.Value[TensorType],
        epsilon1: max._core.Value[TensorType],
        epsilon2: max._core.Value[TensorType],
        weight_offset1: max._core.Value[TensorType],
        weight_offset2: max._core.Value[TensorType],
        multiply_before_cast: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def residual_input(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma1(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma2(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon1(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon2(self) -> max._core.Value[TensorType]: ...
    @property
    def weight_offset1(self) -> max._core.Value[TensorType]: ...
    @property
    def weight_offset2(self) -> max._core.Value[TensorType]: ...
    @property
    def multiply_before_cast(self) -> bool: ...
    @multiply_before_cast.setter
    def multiply_before_cast(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CompositeRmsNormRopeOp(max._core.Operation):
    """
    Fused operation computing RMS normalization followed by Rotary Position
    Embedding (RoPE):

      normed = rms_norm(input, weight, epsilon, weight_offset)
      x1, x2 = split(normed, axis=-1)
      rotated = concat(-x2, x1, axis=-1)
      result = normed * cos_vals + rotated * sin_vals

    Example:

    ```mlir
      %result = mo.composite.rms_norm_rope(%input, %weight, %epsilon, %offset,
                                           %cos_vals, %sin_vals)
        {multiply_before_cast = false} :
        (!mo.tensor<[2, 3, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>,
         !mo.tensor<[], bf16>, !mo.tensor<[], bf16>,
         !mo.tensor<[2, 3, 128], f32, gpu:0>, !mo.tensor<[2, 3, 128], f32, gpu:0>)
        -> !mo.tensor<[2, 3, 128], bf16, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        weight: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        weight_offset: max._core.Value[TensorType],
        cos_vals: max._core.Value[TensorType],
        sin_vals: max._core.Value[TensorType],
        multiply_before_cast: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def weight(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def weight_offset(self) -> max._core.Value[TensorType]: ...
    @property
    def cos_vals(self) -> max._core.Value[TensorType]: ...
    @property
    def sin_vals(self) -> max._core.Value[TensorType]: ...
    @property
    def multiply_before_cast(self) -> bool: ...
    @multiply_before_cast.setter
    def multiply_before_cast(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CompositeRopeRaggedOp(max._core.Operation):
    """
    Applies Rotary Position Embedding (RoPE) to `input`, a ragged batch of
    tokens. Per-token absolute positions are derived from `inputRowOffsets`
    (the ragged batch boundaries) and `startPos` (each sequence's current
    cache length) and used to index `freqsCis`. When `freqsCis`'s last
    dimension is smaller than `input`'s, RoPE is applied to only
    `freqsCis`-many columns of each head and the rest pass through
    unrotated: the trailing columns are rotated by default (the MLA
    layout), or the leading ones when `rope_first` is set (the
    DeepSeekV3.2/GLM Indexer layout, where Q and K are chunked as
    `pe, nope`). `rope_first` is meaningless -- and must be false -- when
    `freqsCis` is as wide as `input`, since then no column passes through.

    Example:

    ```mlir
      %result = mo.composite.rope.ragged(%input, %row_offsets, %start_pos,
                                          %freqs_cis)
        {interleaved = false, rope_first = false} :
        (!mo.tensor<[8, 1, 64], bf16, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
         !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 64], f32, gpu:0>)
        -> !mo.tensor<[8, 1, 64], bf16, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        input_row_offsets: max._core.Value[TensorType],
        start_pos: max._core.Value[TensorType],
        freqs_cis: max._core.Value[TensorType],
        interleaved: max._core.dialects.builtin.BoolAttr,
        rope_first: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def input_row_offsets(self) -> max._core.Value[TensorType]: ...
    @property
    def start_pos(self) -> max._core.Value[TensorType]: ...
    @property
    def freqs_cis(self) -> max._core.Value[TensorType]: ...
    @property
    def interleaved(self) -> bool: ...
    @interleaved.setter
    def interleaved(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def rope_first(self) -> bool: ...
    @rope_first.setter
    def rope_first(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CoordinateTransformMode(enum.Enum):
    half_pixel = 0

    align_corners = 1

    asymmetric = 2

    half_pixel_1D = 3

class CoordinateTransformModeAttr(max._core.Attribute):
    """This attribute is used by `mo.resize`."""

    def __init__(self, value: CoordinateTransformMode) -> None: ...
    @property
    def value(self) -> CoordinateTransformMode: ...

class ParamDeclarationInterface(Protocol):
    """
    Interface to be implemented by ops that declare shape or dimension
    parameters.
    """

    @property
    def implicitly_parametric(self) -> bool: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def get_effects(
        self, arg: Sequence[max._core._MemoryEffect], /
    ) -> None: ...
    def walk_declarations(
        self, arg: Callable[[max._core.dialects.kgen.ParamDeclAttr], None], /
    ) -> None: ...
    def walk_definitions(
        self,
        arg: Callable[
            [
                max._core.dialects.kgen.ParamDeclAttr,
                max._core.dialects.kgen.ParamDefValue,
            ],
            None,
        ],
        /,
    ) -> None: ...
    def rename_declarations(
        self, arg: Sequence[max._core.dialects.kgen.ParamDeclAttr], /
    ) -> None: ...
    def collect_parameter_uses(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...
    def collect_parameter_uses_below(
        self,
        arg0: Callable[[max._core.Attribute], None],
        arg1: Callable[[max._core.Type], None],
        /,
    ) -> None: ...

class ShapeFromTensorOp(max._core.Operation):
    """
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mosh.ShapeType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ShapeToTensorOp(max._core.Operation):
    """
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[max._core.dialects.mosh.ShapeType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mosh.ShapeType]: ...

class StaticBroadcastToOp(max._core.Operation):
    """
    Broadcasts the input tensor to the result tensor. The shape of the input and
    result tensors must not be unknown or contain unknown dimensions, but can be
    parametric.

    This op only has limited compile-time check on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.broadcast_to` with a constant shape-like tensor instead).

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %res1 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
      kgen.param.declare N = <...>
      %res2 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[N, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class StaticReshapeOp(max._core.Operation):
    """
    Returns a tensor with the same underlying data, but different shape. The
    shape of the input and result tensors must not be unknown or contain unknown
    dimensions, but can be parametric. We do not allow inferred dimensions
    (e.g. -1 in MO_ReshapeOp).

    The op has no compile-time or runtime checks on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.reshape` with a constant shape-like tensor instead).

    Example:

    ```mlir
      %from: !mo.tensor<[2, 3], f32>
      %res = mo.static.reshape(%from)
        : !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class AbsOp(max._core.Operation):
    """
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class AddOp(max._core.Operation):
    """
    Returns `x + y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class AddSingletonDimOp(max._core.Operation):
    """
    Adds a dimension of `1` to a shape at the given axis.

    Example:
    ```mlir
      mo.add_singleton_dim[1](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class DistributedAllgatherOp(max._core.Operation):
    """
    AllGather takes in inputs each coming from a different device and collects
    the data into an output tensor along the 0th dimension. The output is
    replicated across the same devices.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        group_size: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class DistributedAllreduceSumOp(max._core.Operation):
    """
    Allreduce takes in inputs each coming from a different device with
    the same shape as the final output and performs a sum reduction
    across the devices.

    When `group_size` is set to a nonzero value smaller than the number of
    inputs, contiguous runs of `group_size` devices form independent
    allreduce groups (e.g. tensor-parallel groups under data parallelism);
    shapes must then only match within each group.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        group_size: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class AndOp(max._core.Operation):
    """
    Returns `x and y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class ReduceArgMaxOp(max._core.Operation):
    """
    This op is equivalent to reduce_max, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical maximum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %1 = mo.arg_max(%0) {axis = 1 : index} : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceArgMinOp(max._core.Operation):
    """
    This op is equivalent to reduce_min, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical minimum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %1 = mo.arg_min(%0) {axis = 1 : index} : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ArgNonzeroOp(max._core.Operation):
    """
    Returns a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>
      } : !mo.tensor<[3, 3], si32>
      %1 = mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AtanhOp(max._core.Operation):
    """
    Returns `atanh(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.atanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class AvgPoolCeilModeTrueOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2d avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        count_boundary: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AvgPoolOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2D avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. `strides`, `dilations`, `padding`) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        count_boundary: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class LinalgBandPartOp(max._core.Operation):
    """
    Copies a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matrices.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    If `exclude` is set, the selection is reverted: The elements in band are set
    to zero while the elements outside the band are copied to the output tensor.

    Please explicitly note that with negative values, this kernel returns the
    entire lower or upper triangle of the matrix, and otherwise returns
    a diagonal band around the main diagonal of the matrix.

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 2, 3], f32>
      %num_lower = mo.constant {
        value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %num_upper = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %exclude = mo.constant {
        value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[], bool>
      %res = mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        num_lower: max._core.Value[TensorType],
        num_upper: max._core.Value[TensorType],
        exclude: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def num_lower(self) -> max._core.Value[TensorType]: ...
    @property
    def num_upper(self) -> max._core.Value[TensorType]: ...
    @property
    def exclude(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BatchMatmulOp(max._core.Operation):
    """
    Performs matrix multiplication on two batches of matrices, represented by
    two N-dimensional tensors.

    The last two dimensions of each input are the matrix dimensions.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[3, 4, 5], f32>
      %rhs: ... !mo.tensor<[3, 5, 6], f32>
      %res = mo.batch_matmul(%lhs, %rhs) :
        mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3, 5, 6], f32>
      ) -> !mo.tensor<[3, 4, 6], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_a: max._core.Value[TensorType],
        input_b: max._core.Value[TensorType],
        transpose_b: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        type: max._core.Type,
        input_a: max._core.Value,
        input_b: max._core.Value,
        decls: Sequence[max._core.dialects.kgen.ParamDeclAttr],
    ) -> None: ...
    @property
    def input_a(self) -> max._core.Value[TensorType]: ...
    @property
    def input_b(self) -> max._core.Value[TensorType]: ...
    @property
    def transpose_b(self) -> bool: ...
    @transpose_b.setter
    def transpose_b(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BottomKOp(max._core.Operation):
    """
    Computes the bottom (lowest) values and their corresponding indices in a
    tensor along a specified axis. Returned values along the axis are always
    sorted (stable).

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: TensorType,
        indices: TensorType,
        input: max._core.Value[TensorType],
        _k: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        sorted: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def _k(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def sorted(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BroadcastShapeOp(max._core.Operation):
    """
    Returns the shape that two shapes would broadcast to under the numpy rules:

    1. Make the shapes the same rank by adding 1 to the front
       of the shorter shape
    2. Ensure each dimension is equal, either by matching them or promoting a 1
       to a larger number from the other shape

    Examples:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %arg2: !mo.tensor<[4, 2, 1], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[4, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, `shape1` will compute to [4, 2, 3]

    ```mlir
    %arg1: !mo.tensor<[10, 2, 1], f32>
    %arg2: !mo.tensor<[5], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[10, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[5], f32>) -> !mo.tensor<[1], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, `shape1` will compute to [10, 2, 5]
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BroadcastToOp(max._core.Operation):
    """
    Broadcasts the input tensor to the specified shape.

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %to = mo.constant {
        value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %res = mo.broadcast_to(%from, %to) : (
        !mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        new_shape: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        broadcast_like: max._core.Value[TensorType],
        shape_of_input: max._core.Value[TensorType] = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def new_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BufferCreateOp(max._core.Operation):
    """
    This operation creates a buffer with the specified shape and data type on a given device.

    By default the buffer is uninitialized and (re)allocated on every execution,
    for use cases where it will be filled with data later in the computation.

    When the optional `initValue` scalar attribute is set, the buffer instead
    becomes persistent state: it is allocated once and filled with `initValue`
    a single time during the model's `init` phase, and the same buffer is reused
    across every `execute` call. This lets a kernel keep and mutate persistent
    scratch state that is only initialized once (e.g. a counter buffer that a
    kernel zeroes at the end of each invocation and expects to find zeroed on the
    first call). `initValue` must have the same element type as the buffer.

    Example:
    ```mlir
    %buf = mo.buffer.create : !mo.buffer<[20, 20], f32, gpu:0>
    %zeroed = mo.buffer.create { initValue = 0.0 : f32 } : !mo.buffer<[20, 20], f32, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        init_value: max._core.Attribute = ...,
    ) -> None: ...
    @property
    def init_value(self) -> max._core.Attribute | None: ...
    @init_value.setter
    def init_value(self, arg: max._core.Attribute, /) -> None: ...

class BufferTransferOp(max._core.Operation):
    """
    This operation transfers data from a source buffer to a destination buffer.
    Both buffers must have the same shape and data type. The operation takes an input
    chain and produces an output chain to sequence the transfer with other operations.

    Example:
    ```mlir
    %outChain = mo.buffer.transfer[%inChain](%src, %dst) : !mo.buffer<[2,3], f32, gpu:0>, !mo.buffer<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        src: max._core.Value[BufferType],
        dst: max._core.Value[BufferType],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def src(self) -> max._core.Value[BufferType]: ...
    @property
    def dst(self) -> max._core.Value[BufferType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class BundledAllreduceSumOp(max._core.Operation):
    """
    Per-device entry point for allreduce sum, used inside an `mo.parallel`
    region.  Takes N peer tensor inputs (from `mo.bundled.expand`), N
    signal buffers (captured from graph scope — the `buffers(...)` clause
    on the parent parallel op provides chain guarding), and a chain.

    Example:
    ```mlir
    mo.parallel (%arg) in (%dt : !mo.bundle<[...]>)
        buffers(%sig0 : ..., %sig1 : ...) chain(%ch) -> (...) {
      %peer0, %peer1 = mo.bundled.expand(%arg)
          : !mo.tensor<[3], f32, gpu:0>
         -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
      %out, %ch_out = mo.bundled.allreduce.sum(
          %peer0, %peer1, %sig0, %sig1, %ch)
          : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
             !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>,
             !mo.chain)
          -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
      mo.yield %out : !mo.tensor<[3], f32, gpu:0>
    }
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class BundledExpandOp(max._core.Operation):
    """
    Inside an `mo.parallel` body, takes a single-device tensor (typically a
    block argument) and produces one tensor per launch with the corresponding
    device placement.  This makes collective N-expansion explicit in the IR
    rather than implicit during lowering.

    The number of results must equal the parent parallel op's launch count.
    Result types must have the same shape and dtype as the input but with
    devices matching each launch (derived from the first input bundle).

    Example:
    ```mlir
    mo.parallel (%arg) in (%dt : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                                              !mo.tensor<[3], f32, gpu:1>]>)
        -> (!mo.bundle<[...]>) {
      %peer0, %peer1 = mo.bundled.expand(%arg)
          : !mo.tensor<[3], f32, gpu:0>
         -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
      // ... use %peer0, %peer1 as inputs to a collective ...
    }
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class CallOp(max._core.Operation):
    """
    This op calls a computation graph.

    Example:

    ```mlir
      %res = mo.call @gelu(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        operands: Sequence[max._core.Value[max._core.Type]],
        callee: max._core.dialects.builtin.FlatSymbolRefAttr,
        prefix: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        arg_attrs: max._core.dialects.builtin.ArrayAttr,
        res_attrs: max._core.dialects.builtin.ArrayAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def callee(self) -> str: ...
    @callee.setter
    def callee(
        self, arg: max._core.dialects.builtin.FlatSymbolRefAttr, /
    ) -> None: ...
    @property
    def prefix(self) -> str: ...
    @prefix.setter
    def prefix(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def arg_attrs(self) -> max._core.dialects.builtin.ArrayAttr | None: ...
    @arg_attrs.setter
    def arg_attrs(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def res_attrs(self) -> max._core.dialects.builtin.ArrayAttr | None: ...
    @res_attrs.setter
    def res_attrs(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...

class CastOp(max._core.Operation):
    """
    Returns the input tensor, cast to the specified element type.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], i32>
      %res = mo.cast(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        dtype: max._core.dtype.DType,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class CeilOp(max._core.Operation):
    """
    Returns the smallest largest integer greater than `x`, where `x` is input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ChainCreateOp(max._core.Operation):
    """
    This operation consumes an arbitrary number of values and produces a chain.
    Can be used for the sequencing for side-effecting ops when they might
    depend on multiple other ops producing chains.

    ```mlir

    %ch0 = mo.assert ...
    %x = mo.matmul ....

    %ch1 = mo.chain.create(%ch0: !mo.chain, %x: mo.tensor)
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class ConcatOp(max._core.Operation):
    """
    Concatenates the input tensors along a given dimension.

    `mo.concat` concatenates the `inputs` tensors into an output tensor. There
    must be at least 1 input tensor.

    The following constraints apply to the inputs/outputs:

    - The input tensors and output tensors all has the same shape except along
      the concatenation dimension `axis`.
    - The size of the concatenation dimension in output tensor have be the sum
      of sizes of the concatenation dimension in input tensors.
    - The element type of the input and output tensors must match.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %res = mo.concat[1](%arg0, %arg1) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        axis: max._core.dialects.builtin.IntegerAttr,
        inputs: Sequence[max._core.Value[max._core.Type]],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ConstantExternalOp(max._core.Operation):
    """
    Represents an undefined reference to an "external" constant.
    This constant's backing data is undefined at graph compile time, but all
    other attributes of the tensor are statically known.
    In a given mo.graph, mo.constant.external ops should be uniquely named.

    Currently, the `device` attribute on the op determines which device the
    weights registry pointers reside on.
    This should be checked at runtime.
    The device attribute and result type can mismatch only when the `device`
    attribute is on the host.
    This is to support mmap'ed weights, a common use case for external
    constants.

    TODO(MSDK-1060): implement the runtime check by storing device alongside
    runtime pointer in the weights registry.
    And emit an explicit mo.transfer in the graph API instead of implicitly
    inserting an HtoD copy in converting mo.constant.external to
    mgp.buffer.constant.external.

    If the `hasAlias` attribute is set to true, the external constant might be
    updated through external alias between graph invocation, and it will not
    be lifted to the init phase.

    Example:

    ```mlir
    %weight = mo.constant.external {
      align = 16 : ui64, device = #M.device_ref<"gpu", 0>, name = "foo"
    } : !mo.tensor<[4096, 4096], bf16>
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        name: max._core.dialects.builtin.StringAttr,
        align: max._core.dialects.builtin.IntegerAttr,
        device: max._core.dialects.m.DeviceRefAttr,
        has_alias: max._core.dialects.builtin.BoolAttr,
        is_placeholder: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def align(self) -> int: ...
    @align.setter
    def align(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def device(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @device.setter
    def device(self, arg: max._core.dialects.m.DeviceRefAttr, /) -> None: ...
    @property
    def has_alias(self) -> bool: ...
    @has_alias.setter
    def has_alias(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def is_placeholder(self) -> bool: ...
    @is_placeholder.setter
    def is_placeholder(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class ConstantOp(max._core.Operation):
    """
    This op allows storing literal values inside the graph.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result_type: max._core.Type,
        value: max._core.dialects.builtin.ElementsAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[int],
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: int,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[int],
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: int,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[float],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: float,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[int],
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: int,
        element_type: max._core.Type,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: float,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: float,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: Sequence[bool],
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: bool,
        device_ref: max._core.dialects.m.DeviceRefAttr = ...,
    ) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.ElementsAttr: ...
    @value.setter
    def value(
        self, arg: max._core.dialects.builtin.ElementsAttr, /
    ) -> None: ...

class ConstantScalarOp(max._core.Operation):
    """
    Same as `mo.constant`, but specialized to scalar types.

    Example:

    ```mlir
      %0 = mo.constant.scalar { value = 3 : si64 } : !mo.scalar<si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: ScalarType,
        value: max._core.Attribute,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def value(self) -> max._core.Attribute | None: ...
    @value.setter
    def value(self, arg: max._core.Attribute, /) -> None: ...

class ConvOp(max._core.Operation):
    """
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout FCRS, i.e.,
      (out_channels, in_channels / num_groups, height, width)

    The filter_layout attribute specifies the memory layout of the filter
    tensor. If empty, the layout is inferred by the InferLayouts pass
    (defaults to FCRS for 2D, FCQRS for 3D). Supported layouts include
    FCRS, RSCF (legacy), and packed variants like FRSCf.

    `strides`, `dilations`, and `padding` must be of rank 1, or unranked.
    If the input has static rank, all hyperparameters with static shape must
    have sizes of `input_rank - 2`, except padding, which must have
    size `2 * (input_rank - 2)`. Individual elements in the hyperparameters
    apply to corresponding dimensions of the input (after ignoring the batch
    and channel dimensions), with padding representing a before/after pair for
    each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here represents H and dim2 represents W. In python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter tensors' ranks must match if statically known.

    `num_groups` must be a ranked scalar. The number of input and output
    channels must both be divisible by the number of groups `num_groups`.

    This op currently only supports strides and padding on the input.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>}
        : %!mo.tensor<[], si64>
      %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[64, 32, 2, 2], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        num_groups: max._core.Value[TensorType],
        input_layout: max._core.dialects.builtin.StringAttr,
        filter_layout: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def num_groups(self) -> max._core.Value[TensorType]: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def filter_layout(self) -> str: ...
    @filter_layout.setter
    def filter_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ConvTransposeOp(max._core.Operation):
    """
    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The op supports 1D-3D spatial dimensions, with the following layout
    assumptions (note the `out_channel` is w.r.t. the original convolution):
    - input has channel last layout.For 2D, that's NHWC, i.e.,
      (batch_size, height, width, out_channels)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters (i.e. strides, dilations, padding, output_paddings) must
    be of rank 1, or unranked. If the input has static rank, all hyperparameters
    with static shape must have sizes of `input_rank - 2`, except padding, which
    must have size `2 * (input_rank - 2)`. Individual elements in the
    hyperparameters applies to corresponding dimensions of the input (after
    ignoring the batch and channel dimensions), with padding representing a
    before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    The `output_paddings` argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.

    The input, output and filter tensors' ranks must match if statically known.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %op = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %res = mo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_paddings: max._core.Value[TensorType],
        input_layout: max._core.dialects.builtin.StringAttr,
        filter_layout: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def filter_layout(self) -> str: ...
    @filter_layout.setter
    def filter_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class CosOp(max._core.Operation):
    """
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class CumsumOp(max._core.Operation):
    """
    Returns the cumulative summation of input tensors along an axis. By default,
    it copies the first element as is. If the `exclusive` attribute is set to 1,
    then the first element is excluded. The `reverse` attribute causes the
    summation to be done in the opposite direction of the axis.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.cumsum(%arg) {axis = 0 : index, exclusive = 1 : index, reverse = 0 : index} : (
      !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        exclusive: max._core.dialects.builtin.IntegerAttr,
        reverse: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def exclusive(self) -> int: ...
    @exclusive.setter
    def exclusive(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def reverse(self) -> int: ...
    @reverse.setter
    def reverse(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class CustomOp(max._core.Operation):
    """
    The `symbol` attribute specifies which underlying Mojo kernel implements
    this operation. This kernel must be decorated with the appropriate decorator
    with the exact same symbol string.

    The `function` attribute specifies which labeled function the operation
    refers to. Examples: `mogg.shape`.

    Example:

    ```mlir
      %0 = mo.custom {symbol = "test_custom_op"}(%arg0)
        (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
    ```

    Corresponding kernel definition:

    ```mojo
      @register_internal("test_custom_op")
      def foo(...):
        pass
    ```

    Also allows the definition of custom kernels that have side-effects on
    mo.buffer values via the use of chains.
    (Currently limited to NDBuffer based kernels).

    Nothing needs to change about the kernel definition, but now the custom op
    must take a `mo.buffer` operand for the value it wants to mutate and
    produce an output chain as its final result and take an input chain as its
    final operand.

    Example:

    ```mlir
    %ch1 = mo.custom {symbol = "test_mutable_op"}(%arg0, %ch0) :
      (!mo.buffer<[D1, D2], f32>, !mo.chain) -> !mo.chain
    ```
    ```mojo
      @register_internal("test_custom_op")
      def foo(...):
        pass
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        operands: Sequence[max._core.Value[max._core.Type]],
        symbol: max._core.dialects.builtin.StringAttr,
        device: max._core.dialects.m.DeviceRefAttr,
        parameters: max._core.dialects.builtin.DictionaryAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def symbol(self) -> str: ...
    @symbol.setter
    def symbol(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def device(self) -> max._core.dialects.m.DeviceRefAttr: ...
    @device.setter
    def device(self, arg: max._core.dialects.m.DeviceRefAttr, /) -> None: ...
    @property
    def parameters(self) -> max._core.dialects.builtin.DictionaryAttr: ...
    @parameters.setter
    def parameters(
        self, arg: max._core.dialects.builtin.DictionaryAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class DebugPrintOp(max._core.Operation):
    """
    Prints a debug string. If a label attribute is supplied the string is printed with that label.
    Otherwise just the string is printed. For debugging and testing only.

    Example:
    ```mlir
      %ch0: !mo.chain
      %ch1 = mo.debug.print(%ch0) {value = "message", label = "label"}
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_chain: max._core.Value[ChainType],
        value: max._core.dialects.builtin.StringAttr,
        label: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def value(self) -> str: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def label(self) -> str: ...
    @label.setter
    def label(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...

class DebugTensorPrintOp(max._core.Operation):
    """
    Prints a debug representation of argument input. If a label attribute
    is supplied the tensor contents is printed with that label. Otherwise
    just the tensor metadata is printed. For debugging and testing only.

    Example:
    ```mlir
      %arg: !mo.tensor<[5], f32>
      %ch0: !mo.chain
      %ch1 = mo.debug.tensor.print(%ch0, %arg) {label = "label"} : !mo.tensor<[5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_chain: max._core.Value[ChainType],
        input: max._core.Value[TensorType],
        label: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def label(self) -> str: ...
    @label.setter
    def label(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...

class DistributedBroadcastOp(max._core.Operation):
    """
    Broadcast takes a single input tensor from the root device and replicates
    it to all participating devices. The root device is identified by the
    `root` attribute (0-indexed position in the signal buffer list).
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        input: max._core.Value[TensorType],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        root: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def root(self) -> int: ...
    @root.setter
    def root(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class DistributedScatterOp(max._core.Operation):
    """
    Scatter takes in ngpus input tensors (one per GPU, padded from dp_size
    distinct chunks) all residing on the root device, and distributes each
    to the corresponding GPU's output. The root attribute identifies which
    device holds the source data.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        root: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def root(self) -> int: ...
    @root.setter
    def root(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class DivOp(max._core.Operation):
    """
    Returns `x / y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.div(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class DistributedEpCombineOp(max._core.Operation):
    """
    Combines expert outputs back to their original devices across N GPUs.
    Each device sends its expert outputs to the appropriate peers, waits
    for all transfers, and computes the weighted sum of routed expert
    outputs. The output supports epilogue fusion.

    All variadic operands and results have the same size (one per device).
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output_tokens: Sequence[max._core.Type],
        out_chain: ChainType,
        input_tokens: Sequence[max._core.Value[max._core.Type]],
        src_info: Sequence[max._core.Value[max._core.Type]],
        send_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_count_ptrs: Sequence[max._core.Value[max._core.Type]],
        router_weights: Sequence[max._core.Value[max._core.Type]],
        atomic_counters: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        hidden_size: max._core.dialects.builtin.IntegerAttr,
        top_k: max._core.dialects.builtin.IntegerAttr,
        n_experts: max._core.dialects.builtin.IntegerAttr,
        max_token_per_rank: max._core.dialects.builtin.IntegerAttr,
        n_gpus_per_node: max._core.dialects.builtin.IntegerAttr,
        n_nodes: max._core.dialects.builtin.IntegerAttr,
        fused_shared_expert: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def input_tokens(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def src_info(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def send_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_count_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def router_weights(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def atomic_counters(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def hidden_size(self) -> int: ...
    @hidden_size.setter
    def hidden_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def top_k(self) -> int: ...
    @top_k.setter
    def top_k(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def n_experts(self) -> int: ...
    @n_experts.setter
    def n_experts(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def max_token_per_rank(self) -> int: ...
    @max_token_per_rank.setter
    def max_token_per_rank(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_gpus_per_node(self) -> int: ...
    @n_gpus_per_node.setter
    def n_gpus_per_node(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @n_nodes.setter
    def n_nodes(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def fused_shared_expert(self) -> bool: ...
    @fused_shared_expert.setter
    def fused_shared_expert(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class DistributedEpDispatchBlockScaledNvOp(max._core.Operation):
    """
    Dispatches input tokens to expert devices across N GPUs using the Expert
    Parallelism protocol with NVIDIA block-scaled quantized output format.
    Each device routes its tokens based on top-k expert IDs, quantizes them to
    NVFP4, MXFP4 or MXFP8, and sends them to the appropriate peer via
    shared-memory or NVSHMEM pointers.

    All variadic operands and results have the same size (one per device).
    The `sendPtrs`, `recvPtrs`, and `recvCountPtrs` are host-side pointer
    tensors that are typically identical across devices (replicated N times
    to satisfy the same-variadic-size constraint).
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output_tokens: Sequence[max._core.Type],
        output_scales: Sequence[max._core.Type],
        row_offsets: Sequence[max._core.Type],
        scales_offsets: Sequence[max._core.Type],
        expert_ids: Sequence[max._core.Type],
        src_info: Sequence[max._core.Type],
        out_chain: ChainType,
        input_tokens: Sequence[max._core.Value[max._core.Type]],
        topk_ids: Sequence[max._core.Value[max._core.Type]],
        send_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_count_ptrs: Sequence[max._core.Value[max._core.Type]],
        input_scales: Sequence[max._core.Value[max._core.Type]],
        atomic_counters: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        hidden_size: max._core.dialects.builtin.IntegerAttr,
        top_k: max._core.dialects.builtin.IntegerAttr,
        n_experts: max._core.dialects.builtin.IntegerAttr,
        max_token_per_rank: max._core.dialects.builtin.IntegerAttr,
        n_gpus_per_node: max._core.dialects.builtin.IntegerAttr,
        n_nodes: max._core.dialects.builtin.IntegerAttr,
        fused_shared_expert: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def input_tokens(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def topk_ids(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def send_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_count_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def input_scales(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def atomic_counters(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def hidden_size(self) -> int: ...
    @hidden_size.setter
    def hidden_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def top_k(self) -> int: ...
    @top_k.setter
    def top_k(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def n_experts(self) -> int: ...
    @n_experts.setter
    def n_experts(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def max_token_per_rank(self) -> int: ...
    @max_token_per_rank.setter
    def max_token_per_rank(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_gpus_per_node(self) -> int: ...
    @n_gpus_per_node.setter
    def n_gpus_per_node(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @n_nodes.setter
    def n_nodes(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def fused_shared_expert(self) -> bool: ...
    @fused_shared_expert.setter
    def fused_shared_expert(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class DistributedEpDispatchFp8Op(max._core.Operation):
    """
    Dispatches input tokens to expert devices across N GPUs using the Expert
    Parallelism protocol with blockwise FP8 quantized output format.

    All variadic operands and results have the same size (one per device).
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output_tokens: Sequence[max._core.Type],
        output_scales: Sequence[max._core.Type],
        row_offsets: Sequence[max._core.Type],
        expert_ids: Sequence[max._core.Type],
        src_info: Sequence[max._core.Type],
        out_chain: ChainType,
        input_tokens: Sequence[max._core.Value[max._core.Type]],
        topk_ids: Sequence[max._core.Value[max._core.Type]],
        send_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_count_ptrs: Sequence[max._core.Value[max._core.Type]],
        atomic_counters: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        hidden_size: max._core.dialects.builtin.IntegerAttr,
        top_k: max._core.dialects.builtin.IntegerAttr,
        n_experts: max._core.dialects.builtin.IntegerAttr,
        max_token_per_rank: max._core.dialects.builtin.IntegerAttr,
        n_gpus_per_node: max._core.dialects.builtin.IntegerAttr,
        n_nodes: max._core.dialects.builtin.IntegerAttr,
        fused_shared_expert: max._core.dialects.builtin.BoolAttr,
        dispatch_scale_granularity: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def input_tokens(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def topk_ids(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def send_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_count_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def atomic_counters(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def hidden_size(self) -> int: ...
    @hidden_size.setter
    def hidden_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def top_k(self) -> int: ...
    @top_k.setter
    def top_k(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def n_experts(self) -> int: ...
    @n_experts.setter
    def n_experts(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def max_token_per_rank(self) -> int: ...
    @max_token_per_rank.setter
    def max_token_per_rank(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_gpus_per_node(self) -> int: ...
    @n_gpus_per_node.setter
    def n_gpus_per_node(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @n_nodes.setter
    def n_nodes(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def fused_shared_expert(self) -> bool: ...
    @fused_shared_expert.setter
    def fused_shared_expert(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def dispatch_scale_granularity(self) -> str: ...
    @dispatch_scale_granularity.setter
    def dispatch_scale_granularity(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...

class DistributedEpDispatchMxfp4Op(max._core.Operation):
    """
    Dispatches input tokens to expert devices across N GPUs using the Expert
    Parallelism protocol with MXFP4 quantized output format. Each device
    routes its tokens based on top-k expert IDs, quantizes them to MXFP4,
    and sends them to the appropriate peer via shared-memory or ROCSHMEM
    pointers.

    All variadic operands and results have the same size (one per device).
    The `sendPtrs`, `recvPtrs`, and `recvCountPtrs` are host-side pointer
    tensors that are typically identical across devices (replicated N times
    to satisfy the same-variadic-size constraint).
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output_tokens: Sequence[max._core.Type],
        output_scales: Sequence[max._core.Type],
        row_offsets: Sequence[max._core.Type],
        expert_ids: Sequence[max._core.Type],
        src_info: Sequence[max._core.Type],
        out_chain: ChainType,
        input_tokens: Sequence[max._core.Value[max._core.Type]],
        topk_ids: Sequence[max._core.Value[max._core.Type]],
        send_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_count_ptrs: Sequence[max._core.Value[max._core.Type]],
        atomic_counters: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        hidden_size: max._core.dialects.builtin.IntegerAttr,
        top_k: max._core.dialects.builtin.IntegerAttr,
        n_experts: max._core.dialects.builtin.IntegerAttr,
        max_token_per_rank: max._core.dialects.builtin.IntegerAttr,
        n_gpus_per_node: max._core.dialects.builtin.IntegerAttr,
        n_nodes: max._core.dialects.builtin.IntegerAttr,
        fused_shared_expert: max._core.dialects.builtin.BoolAttr,
        fuse_a_scale_preshuffle: max._core.dialects.builtin.BoolAttr,
        max_padded_m: max._core.dialects.builtin.IntegerAttr,
        mx_format: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def input_tokens(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def topk_ids(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def send_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_count_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def atomic_counters(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def hidden_size(self) -> int: ...
    @hidden_size.setter
    def hidden_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def top_k(self) -> int: ...
    @top_k.setter
    def top_k(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def n_experts(self) -> int: ...
    @n_experts.setter
    def n_experts(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def max_token_per_rank(self) -> int: ...
    @max_token_per_rank.setter
    def max_token_per_rank(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_gpus_per_node(self) -> int: ...
    @n_gpus_per_node.setter
    def n_gpus_per_node(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @n_nodes.setter
    def n_nodes(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def fused_shared_expert(self) -> bool: ...
    @fused_shared_expert.setter
    def fused_shared_expert(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def fuse_a_scale_preshuffle(self) -> bool: ...
    @fuse_a_scale_preshuffle.setter
    def fuse_a_scale_preshuffle(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def max_padded_m(self) -> int: ...
    @max_padded_m.setter
    def max_padded_m(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def mx_format(self) -> str: ...
    @mx_format.setter
    def mx_format(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...

class DistributedEpDispatchOp(max._core.Operation):
    """
    Dispatches input tokens to expert devices across N GPUs using the Expert
    Parallelism protocol with BF16 output format.

    All variadic operands and results have the same size (one per device).
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output_tokens: Sequence[max._core.Type],
        row_offsets: Sequence[max._core.Type],
        expert_ids: Sequence[max._core.Type],
        src_info: Sequence[max._core.Type],
        out_chain: ChainType,
        input_tokens: Sequence[max._core.Value[max._core.Type]],
        topk_ids: Sequence[max._core.Value[max._core.Type]],
        send_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_ptrs: Sequence[max._core.Value[max._core.Type]],
        recv_count_ptrs: Sequence[max._core.Value[max._core.Type]],
        atomic_counters: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        hidden_size: max._core.dialects.builtin.IntegerAttr,
        top_k: max._core.dialects.builtin.IntegerAttr,
        n_experts: max._core.dialects.builtin.IntegerAttr,
        max_token_per_rank: max._core.dialects.builtin.IntegerAttr,
        n_gpus_per_node: max._core.dialects.builtin.IntegerAttr,
        n_nodes: max._core.dialects.builtin.IntegerAttr,
        fused_shared_expert: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def input_tokens(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def topk_ids(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def send_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def recv_count_ptrs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def atomic_counters(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def hidden_size(self) -> int: ...
    @hidden_size.setter
    def hidden_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def top_k(self) -> int: ...
    @top_k.setter
    def top_k(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def n_experts(self) -> int: ...
    @n_experts.setter
    def n_experts(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def max_token_per_rank(self) -> int: ...
    @max_token_per_rank.setter
    def max_token_per_rank(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_gpus_per_node(self) -> int: ...
    @n_gpus_per_node.setter
    def n_gpus_per_node(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def n_nodes(self) -> int: ...
    @n_nodes.setter
    def n_nodes(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def fused_shared_expert(self) -> bool: ...
    @fused_shared_expert.setter
    def fused_shared_expert(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class EqualOp(max._core.Operation):
    """
    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                    !mo.tensor<[2, 3], f32>
                                    ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class ErfOp(max._core.Operation):
    """
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ExpOp(max._core.Operation):
    """
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class FloorOp(max._core.Operation):
    """
    Returns the elementwise largest integer not greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class GatherNdOp(max._core.Operation):
    """
    Variant of `mo.gather` that accepts multi-dimensional indices.

    The last dimension stores the index whereas
    the outer dimensions act like batch dimensions. The size of the last
    dimension is at most the rank of the input. When the dimension size is less
    than the rank of the input, slices of the input are gathered, starting from
    the leftmost dimension.

    ```
    output_shape = (
          input.shape[:batch_dims]
        + indices.shape[batch_dims:-1]
        + data.shape[batch_dims + indices.shape[-1]:]
    )
    ```

    ```mlir
      %input = mo.constant {device = #M.device_ref<"cpu", 0>, value =
        #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> :
        tensor<2x2x4xsi64>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} :
        !mo.tensor<[3], si64>

      %result = mo.gather_nd(%input, %indices) {batchDims = 0} :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        batch_dims: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def batch_dims(self) -> int: ...
    @batch_dims.setter
    def batch_dims(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class GatherOp(max._core.Operation):
    """
    Gathers slices from input's axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    The values of `axis` and `indices` follows numpy semantics, e.g., -1
    represents the last axis. `axis` is a compile-time `index` attribute.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %res = mo.gather(%input, %indices) {axis = 0 : index} : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class GatherSumOp(max._core.Operation):
    """
    A temporary composite op that composes a gather with sum reduction.
    The gather axis is 0 and reduction axis is 1.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...

class GeluOp(max._core.Operation):
    """
    Returns the exact GELU activation `0.5 * x * (1 + erf(x / sqrt(2)))`, where
    `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class GeluQuickOp(max._core.Operation):
    """
    Returns the quick approximation of GELU `x * sigmoid(1.702 * x)`, where `x`
    is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.gelu_quick(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class GeluTanhOp(max._core.Operation):
    """
    Returns the tanh approximation of GELU
    `0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))`, where
    `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.gelu_tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class GraphOp(max._core.Operation):
    """
    This op represents a computation graph that consists of:
    - input data and their types
    - output types
    - other ops representing computations on input and intermediate data
    - a terminating output op that returns the outputs

    Example:

    ```mlir
      mo.graph @example<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        // ... intermediate computations ...
        %res : !mo.tensor<[D3], f32>
        mo.output<D3> %arg : !mo.tensor<[D3], f32>
      }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        sym_visibility: max._core.dialects.builtin.StringAttr,
        signature: max._core.dialects.builtin.TypeAttr,
        function_type: max._core.dialects.builtin.TypeAttr,
        input_parameters: max._core.dialects.kgen.ParamDeclArrayAttr,
        result_parameters: max._core.dialects.kgen.ParamDeclArrayAttr,
        counter: int,
        is_subgraph: max._core.dialects.builtin.UnitAttr,
        is_device_graph: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        name: str,
        input_types: Sequence[max._core.Type],
        result_types: Sequence[max._core.Type],
        is_subgraph: bool = False,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def sym_visibility(self) -> str | None: ...
    @sym_visibility.setter
    def sym_visibility(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def signature(self) -> max._core.dialects.kgen.FuncTypeGeneratorType: ...
    @signature.setter
    def signature(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def function_type(self) -> max._core.dialects.builtin.FunctionType: ...
    @function_type.setter
    def function_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def input_parameters(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @input_parameters.setter
    def input_parameters(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def result_parameters(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @result_parameters.setter
    def result_parameters(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def is_subgraph(self) -> bool: ...
    @is_subgraph.setter
    def is_subgraph(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...
    @property
    def is_device_graph(self) -> bool: ...
    @is_device_graph.setter
    def is_device_graph(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...

class GreaterEqualOp(max._core.Operation):
    """
    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class GreaterOp(max._core.Operation):
    """
    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                      !mo.tensor<[2, 3], f32>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class IndexToTensorOp(max._core.Operation):
    """
    Example:

    ```mlir
      %c: scalar<si64>
      %scalarT = mo.index.to_tensor(%c) -> !mo.tensor<[], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value: ...

class IsInfOp(max._core.Operation):
    """
    Returns true if `x` represents a floating point Inf, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...

class IsNanOp(max._core.Operation):
    """
    Returns true if `x` represents a floating point NaN, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...

class ReduceLayerNormOp(max._core.Operation):
    """
    Layer normalization operation which operates on the last dimension of
    `input`:

      meanInput = mean(input)
      varInput = var(input)
      result = (input - meanInput) / sqrt(varInput + epsilon) * gamma + beta.

    We expect gamma and beta to be shape [channels], where channels is the size
    of the last input dimension.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        gamma: max._core.Value[TensorType],
        beta: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma(self) -> max._core.Value[TensorType]: ...
    @property
    def beta(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class Log1pOp(max._core.Operation):
    """
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class LogOp(max._core.Operation):
    """
    Returns the natural logarithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ReduceLogsoftmaxOp(max._core.Operation):
    """
    Returns `log(softmax(x, axis))`, where `x` is input tensor, and `axis` is
    the axis along which `softmax` is applied.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.logsoftmax(%arg, %axis) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class MatmulOp(max._core.Operation):
    """
    Performs matrix multiplication on two 2D tensors.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[10, 20], f32>
      %rhs: ... !mo.tensor<[20, 5], f32>
      %res = mo.matmul(%lhs, %rhs) : (
        !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
      ) -> !mo.tensor<[10, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_a: max._core.Value[TensorType],
        input_b: max._core.Value[TensorType],
        transpose_b: max._core.dialects.builtin.BoolAttr,
        packed_b: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def input_a(self) -> max._core.Value[TensorType]: ...
    @property
    def input_b(self) -> max._core.Value[TensorType]: ...
    @property
    def transpose_b(self) -> bool: ...
    @transpose_b.setter
    def transpose_b(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def packed_b(self) -> bool: ...
    @packed_b.setter
    def packed_b(self, arg: max._core.dialects.builtin.BoolAttr, /) -> None: ...

class MaxOp(max._core.Operation):
    """
    Returns `max(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.max(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class MaxPoolCeilModeTrueOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    The op supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MaxPoolOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        filter_shape: max._core.Value[TensorType],
        strides: max._core.Value[TensorType],
        dilations: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def filter_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def strides(self) -> max._core.Value[TensorType]: ...
    @property
    def dilations(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMeanOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their mean value, changng that
    axis's dimension to 1.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.mean(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MergeDimOp(max._core.Operation):
    """
    Merges two adjacent dimensions of a tensor into one. Example:
    Input=[A, B, C, D], Axis=1

    Output=[A, B*C, D].

    We merge axis i and i+1 into one dimension.

    Example:
    ```mlir
      %out = mo.merge_dim[1](%res): (!mo.tensor<[1, 2, 3, 4], f32>) -> !mo.tensor<[1, 6, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class MinOp(max._core.Operation):
    """
    Returns `min(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.min(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class ModOp(max._core.Operation):
    """
    Returns `x mod y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], si32>
      %rhs: !mo.tensor<[2, 3], si32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class MulOp(max._core.Operation):
    """
    Returns `x * y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.mul(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class MutableLoadOp(max._core.Operation):
    """
    Allows modelling of in-place operations in MO in conjunction with mo.mutable.store.

    This is semantically equivalent to a copy from `inBuffer` to `outTensor`

    The output chain of this operation is only allowed to have at most one use.

    If the value semantic output of this operation has more than one use the
    operation becomes ineligibile for fusion with compute operations.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_tensor: TensorType,
        out_chain: ChainType,
        in_buffer: max._core.Value[BufferType],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def in_buffer(self) -> max._core.Value[BufferType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class MutableStoreOp(max._core.Operation):
    """
    Allows modelling of in-place operations in MO in conjunction with mo.mutable.load.

    This is semantically equivalent to a copy from `inTensor` to `inBuffer`

    The output chain of this operation is only allowed to have at most one use.

    If the value semantic tensor input of this operation has more than one use
    the operation becomes ineligibile for fusion with compute operations.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_buffer: max._core.Value[BufferType],
        in_tensor: max._core.Value[TensorType],
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @property
    def in_buffer(self) -> max._core.Value[BufferType]: ...
    @property
    def in_tensor(self) -> max._core.Value[TensorType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class MutableStoreSliceOp(max._core.Operation):
    """
    Stores the  tensor `slice` to a subset of the elements in `inBuffer`.

    The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
    ```mlir
      %buffer: !mo.buffer<[20, 20], f32>
      %slice: !mo.tensor<[D0, D1], f32>
      %ch: !mo.chain

      %start: !mo.tensor<[2], si64> // [1,  -6]
      %stop: !mo.tensor<[2], si32>  // [-3, -3]
      %step: !mo.tensor<[2], si64>  // [5,   1]

      // equivalent to this in numpy: `buffer[1:-3:5, -6:-3:1] = slice`
      %ch' = mo.mutable.store.slice(%ch, %buffer, %slice, %start, %stop, %step)
    ```

    Both consumes and produces a chain. The output chain is allowed to have at
    most one use.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: ChainType,
        in_buffer: max._core.Value[BufferType],
        slice: max._core.Value[TensorType],
        start: max._core.Value[TensorType],
        stop: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
        in_chain: max._core.Value[ChainType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def in_buffer(self) -> max._core.Value[BufferType]: ...
    @property
    def slice(self) -> max._core.Value[TensorType]: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def stop(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class NegativeOp(max._core.Operation):
    """
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class NonMaximumSuppressionOp(max._core.Operation):
    """
    Filters out boxes that have high intersection-over-union (IOU).

    `boxes` is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the
    coordinates of any diagonal pair of box corners and the coordinates can be
    provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

     Example:

     ```mlir
       %boxes : !mo.tensor<[1, 6, 4], f32>
       %scores : !mo.tensor<[1, 1, 6], f32>
       %maxOutputBoxesPerClass : !mo.tensor<[], si64>
       %iouThreshold : !mo.tensor<[], si64>
       %scoreThreshold : !mo.tensor<[], si64>
       %res = mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
     ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: TensorType,
        boxes: max._core.Value[TensorType],
        scores: max._core.Value[TensorType],
        max_output_boxes_per_class: max._core.Value[TensorType],
        iou_threshold: max._core.Value[TensorType],
        score_threshold: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def boxes(self) -> max._core.Value[TensorType]: ...
    @property
    def scores(self) -> max._core.Value[TensorType]: ...
    @property
    def max_output_boxes_per_class(self) -> max._core.Value[TensorType]: ...
    @property
    def iou_threshold(self) -> max._core.Value[TensorType]: ...
    @property
    def score_threshold(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class NotEqualOp(max._core.Operation):
    """
    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class NotOp(max._core.Operation):
    """
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class OrOp(max._core.Operation):
    """
    Returns `x or y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class OutputOp(max._core.Operation):
    """
    This op specifies the output parameters and values for a `mo.graph`. The
    op takes variable number of operands and produces no results. The operand
    number and types must match the signature of the `mo.graph` that contains
    the op (after substituting the bindings to signature output type).

    Examples:

    ```mlir
      mo.graph @no_params(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
        mo.output %arg0 : !mo.tensor<?, f32>
      }

      mo.graph @with_params<D1 -> D2>(
          %arg0: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        mo.output<D1> %arg0 : !mo.tensor<[D1], f32>
      }
    ```

    Note that in the parameterized example, the output operands type annotation
    and the graph's output type don't necessarily need to match textually, but
    the compiler will eventually verify that they do.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        parameters: max._core.dialects.kgen.ParameterExprArrayAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def parameters(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @parameters.setter
    def parameters(
        self, arg: max._core.dialects.kgen.ParameterExprArrayAttr, /
    ) -> None: ...

class PadConstantOp(max._core.Operation):
    """
    Pads the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assumes input has rank `N`, the `paddings` tensor should have
    shape `(2 * N)`, where each consecutive pair of elements has the form
    `[before, after]`, indicating how many of `constant` to add before and after
    the contents of `input` in that dimension. The size of each dimension D of
    the padded output is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant {
        value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        constant: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def constant(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class PadReflectOp(max._core.Operation):
    """
    Pads the `input` tensor by reflecting it according to the `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class PadRepeatOp(max._core.Operation):
    """
    Pads the `input` tensor by repeating border values according to `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        paddings: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ParallelOp(max._core.Operation):
    """
    The `mo.parallel` operation takes a single "body" block, which is executed
    in parallel for each set of inputs.  Each input is an `!mo.bundle` whose
    elements are the per-device values for one input group.  All bundles must
    have the same number of elements (= number of launches).  The body block
    receives one block argument per bundle input, typed as a representative
    single-device value (the first element's type).

    The yield may return one or more values.  Each yield operand produces one
    `!mo.bundle` result whose elements are derived from the yield type with
    per-launch devices taken from the `launchDevices` attribute.

    The `launchDevices` attribute is the source of truth for the per-launch
    device list.  In the printed form it appears as a `devices(...)` clause.
    When the list matches the elements of `inputs[0]` the clause may be
    omitted, and the parser/printer recover it from the bundle.  When
    `inputs` is empty the clause is required because there is nothing to
    derive from.  The verifier requires every input and result bundle to
    agree with `launchDevices` element-by-element.

    An optional `buffers(...)` clause declares per-launch signal buffers for
    collective operations (e.g. allreduce).  The number of buffers must equal
    the number of launches.  Buffers are operands of the parallel op for
    chain guarding (memory effect tracking) but do NOT produce block
    arguments.  Ops inside the body capture buffer values directly from the
    enclosing scope.

    `buffers(...)` and `chain(...)` must be both present or both absent.  When
    present, the chain is threaded *explicitly through the body* rather than
    captured: the body block gains a trailing `!mo.chain` block argument that
    carries the in-chain, the body's side-effecting (buffer-argument) ops
    consume and thread that chain, and `mo.yield` produces the resulting
    chain as its last operand.  That yielded chain becomes the op's trailing
    `!mo.chain` result (completion of all launches). A chainless parallel has
    neither the chain block argument nor a yielded chain.

    Example with one bundle input (no buffers, no chain):
    ```mlir
    %dt = mo.tensor.bundle(%a, %b) : (...) -> (...)
    // devices(...) elided; auto-derived from %dt.
    %res = mo.parallel (%arg) in (%dt : !mo.bundle<[...]>)
        -> (!mo.bundle<[...]>) {
      %1 = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
      mo.yield %1 : !mo.tensor<[3], f32, gpu:0>
    }
    ```

    Example with buffers and chain (bundled allreduce). The in-chain enters
    through the trailing block arg `%bch`; the collective consumes it and the
    body yields the resulting out-chain:
    ```mlir
    %dt = mo.tensor.bundle(%a, %b) : (...) -> (...)
    %res, %ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[...]>)
        buffers(%s0 : !mo.buffer<[1], ui8, gpu:0>,
                %s1 : !mo.buffer<[1], ui8, gpu:1>)
        chain(%ch_in)
        -> (!mo.bundle<[...]>) {
      %p0, %p1 = mo.bundled.expand(%arg) : ...
      %out, %ch1 = mo.bundled.allreduce.sum(%p0, %p1, %s0, %s1, %bch) : ...
      mo.yield %out, %ch1 : !mo.tensor<[3], f32, gpu:0>, !mo.chain
    }
    ```

    Example with no bundle inputs (clause required, body captures a value
    from the enclosing scope):
    ```mlir
    %res = mo.parallel () in () devices(gpu:0-1)
        -> (!mo.bundle<[...]>) {
      %1 = mo.relu(%a) : !mo.tensor<[3], f32, gpu:0>
      mo.yield %1 : !mo.tensor<[3], f32, gpu:0>
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        inputs: Sequence[max._core.Value[max._core.Type]],
        buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        launch_devices: max._core.dialects.builtin.ArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        inputs: Sequence[max._core.Value[max._core.Type]],
        result_types: Sequence[max._core.Type],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        inputs: Sequence[max._core.Value[max._core.Type]],
        buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value,
        result_types: Sequence[max._core.Type],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        inputs: Sequence[max._core.Value[max._core.Type]],
        buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value,
        result_types: Sequence[max._core.Type],
        launch_devices: Sequence[max._core.dialects.m.DeviceRefAttr],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def launch_devices(self) -> max._core.dialects.builtin.ArrayAttr: ...
    @launch_devices.setter
    def launch_devices(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...

class PowOp(max._core.Operation):
    """
    Computes `x ** y`, where `x` and `y` are input tensors.

    Examples:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], f32>
      ) : !mo.tensor<[2, 3], f32>

      %x: !mo.tensor<[2, 3], f32>
      %y_int: !mo.tensor<[2, 3], si32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], si32>
      ) : !mo.tensor<[2, 3], f32>
      %res = mo.pow(%x, %y_int) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class RandomNormalOp(max._core.Operation):
    """
    Returns a tensor with shape `shape` populated with random values from a
      normal distribution, with the mean of the distribution equal to `mean`
      and the standard deviation equal to `variance`.

    Example:
      ```mlir
        %size = mo.constant {
          value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
        %mean = mo.constant {
          value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
        %variance = mo.constant {
          value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
        %seed = mo.constant {
          value = #M.dense_array<1> : tensor<1xui64> } : !mo.tensor<[1], ui64>
        %res = mo.random.normal(%size, %mean, %variance, %seed) :
              (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
              !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
      ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        shape: max._core.Value[TensorType],
        mean: max._core.Value[TensorType],
        variance: max._core.Value[TensorType],
        seed: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def shape(self) -> max._core.Value[TensorType]: ...
    @property
    def mean(self) -> max._core.Value[TensorType]: ...
    @property
    def variance(self) -> max._core.Value[TensorType]: ...
    @property
    def seed(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RandomUniformOp(max._core.Operation):
    """
    Returns a tensor with shape `shape` populated with random values from a
    uniform distribution on the half-open interval [lowerBound, upperBound).

    Example:
    ```mlir
    %size = mo.constant {
      value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %lowerBound = mo.constant {
      value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
    %upperBound = mo.constant {
      value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
    %seed = mo.constant {
      value = #M.dense_array<1> : tensor<1xui64> } : !mo.tensor<[1], ui64>
    %res = mo.random.uniform(%size, %lowerBound, %upperBound, %seed) :
          (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
          !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        shape: max._core.Value[TensorType],
        lower_bound: max._core.Value[TensorType],
        upper_bound: max._core.Value[TensorType],
        seed: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def shape(self) -> max._core.Value[TensorType]: ...
    @property
    def lower_bound(self) -> max._core.Value[TensorType]: ...
    @property
    def upper_bound(self) -> max._core.Value[TensorType]: ...
    @property
    def seed(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RangeOp(max._core.Operation):
    """
    Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `limit`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant {
        value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
      %step = mo.constant {
        value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
      %res = mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant {
        value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
      %stepInt = mo.constant {
        value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
      %limitInt = mo.constant {
        value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
      %oddNumbersBelowTen = mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        start: max._core.Value[TensorType],
        limit: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def limit(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RebindOp(max._core.Operation):
    """
    This op represents the "rebinding" of a type to another type. This is
    typically used for rebinding the shape of !mo.tensor types to other
    !mo.tensor types to get things to type-check due to strict shape-related
    verifiers.

    Rebinding is similar to casting, except no data conversion takes place.
    It is assumed that the two types ultimately prove to be the same at runtime.

    Therefore, in cases where it is not statically provable, the left and right
    runtime types are the same, it is expected we will insert runtime assertions
    of some sort. Note, this operation does not do this, and any assertions
    must be inserted separately.

    Note, rebinds which use parameters they declare essentially "rename"
    the associated dimensions which is a useful tool. These types of rebinds
    are guaranteed to be removed by shape inference unless they are used to
    name unknown dims (denoted by `?`) from 3P dialects like those in PT.

    Examples:
    ```mlir
      // something like ASSERT N == 3 AND Sh == {3, 1} if we aren't statically
      // sure of this fact.
      %1 = mo.rebind(%0) : !mo.tensor<[N, 1], f32> -> !mo.tensor<Sh, f32>
      %2 = mo.rebind(%1) : !mo.tensor<Sh, f32> -> !mo.tensor<[3, 1], f32>

      // Renames dims. K == 3, M == 1
      %3 = mo.rebind<() -> K, M>(%0) :
        !mo.tensor<[3, 1], f32> -> !mo.tensor<[K, M], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceAddOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their sum, changing that axis's
    dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.add(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceGroupNormOp(max._core.Operation):
    """
    Applies Group Normalization to the input tensor.

    Divides channels into groups and computes normalization statistics
    within each group.

    Example:

    ```mlir
      %res = mo.reduce.group_norm(%input, %gamma, %beta, %epsilon, %num_groups) :
        (!mo.tensor<[1, 32, 64, 64], f32, gpu:0>, !mo.tensor<[32], f32, gpu:0>,
         !mo.tensor<[32], f32, gpu:0>, !mo.tensor<[], f32>, !mo.tensor<[], si32>)
        -> !mo.tensor<[1, 32, 64, 64], f32, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        gamma: max._core.Value[TensorType],
        beta: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        num_groups: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def gamma(self) -> max._core.Value[TensorType]: ...
    @property
    def beta(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def num_groups(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMaxOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their maximum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.max(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceReduceMinAndMaxOp(max._core.Operation):
    """
    Reduces the input tensor along the given axis, returning a single tensor
    where the last dimension contains both the minimum and maximum values (in
    that order).

    For an input of shape [d0, ..., dN] reduced along axis `a`, the output
    shape is [d0, ..., 2] (the reduced axis is replaced by a dimension of 2).

    Example:

    ```mlir
      %res = mo.reduce.reduce_min_and_max(%input) {axis = 1 : index} :
        (!mo.tensor<[2, 10], f32>) -> !mo.tensor<[2, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMinOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their minimum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.min(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceMulOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their product, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.mul(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        axis: int,
        output_ty: TensorType = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceRmsNormOp(max._core.Operation):
    """
    Applies Root Mean Square normalization to the input tensor.

    output = input / rms(input) * weight

    where rms(x) = sqrt(mean(x^2) + epsilon).

    When `multiply_before_cast` is false (Llama-style), the input is cast to
    the output dtype before multiplication by the weight. When true
    (Gemma-style), the multiplication is performed before the cast.

    Example:

    ```mlir
      %res = mo.reduce.rms_norm(%input, %weight, %epsilon, %weight_offset)
        {multiply_before_cast = false} :
        (!mo.tensor<[2, 3], bf16, gpu:0>, !mo.tensor<[3], bf16, gpu:0>,
         !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[2, 3], bf16, gpu:0>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        weight: max._core.Value[TensorType],
        epsilon: max._core.Value[TensorType],
        weight_offset: max._core.Value[TensorType],
        multiply_before_cast: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def weight(self) -> max._core.Value[TensorType]: ...
    @property
    def epsilon(self) -> max._core.Value[TensorType]: ...
    @property
    def weight_offset(self) -> max._core.Value[TensorType]: ...
    @property
    def multiply_before_cast(self) -> bool: ...
    @multiply_before_cast.setter
    def multiply_before_cast(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class DistributedReducescatterSumOp(max._core.Operation):
    """
    ReduceScatter takes in inputs each coming from a different device, and
    partitions the reduction such that each device receives a disjoint subset
    of the result.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        out_chain: ChainType,
        inputs: Sequence[max._core.Value[max._core.Type]],
        signal_buffers: Sequence[max._core.Value[max._core.Type]],
        in_chain: max._core.Value[ChainType],
        axis: max._core.dialects.builtin.IntegerAttr,
        group_size: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def signal_buffers(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class ReluOp(max._core.Operation):
    """
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ReshapeOp(max._core.Operation):
    """
    Returns a tensor with the same underlying data, but different shape.

    The first argument is the tensor to reshape.  The second tensor is the
    shape to reshape the first tensor to.  The second tensor may contain a
    single "-1" element, which signifies that that dimension should be
    automatically computed.

    Example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant {
        value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```

    Auto-sizing example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant {
        value = #M.dense_array<3, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        new_shape: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def new_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ResizeBicubicOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the bicubic interpolation algorithm.

    Bicubic interpolation uses a 4x4 pixel neighborhood and cubic polynomials
    to produce smoother results than linear interpolation. This implementation
    uses Keys' cubic convolution with a = -0.5.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 3, 224, 224], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 3, 448, 448> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.resize.bicubic(%input, %size) :
        (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 3, 448, 448], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        size: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def size(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ResizeLinearOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the linear algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    When set to true, the antialias attribute causes an antialiasing filter to be applied
    when downscaling.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        size: max._core.Value[TensorType],
        coordinate_transform_mode: CoordinateTransformModeAttr,
        antialias: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def size(self) -> max._core.Value[TensorType]: ...
    @property
    def coordinate_transform_mode(self) -> CoordinateTransformMode: ...
    @coordinate_transform_mode.setter
    def coordinate_transform_mode(
        self, arg: CoordinateTransformModeAttr, /
    ) -> None: ...
    @property
    def antialias(self) -> bool: ...
    @antialias.setter
    def antialias(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ResizeNearestOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the nearest-neighbor algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    The values for round mode are:
      - 0: HalfDown
      - 1: HalfUp
      - 2: Floor
      - 3: Ceil

    Round mode is HalfDown (0) by default.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.resize.nearest(%input, %size) {
        coordinate_transform_mode = 0,
        round_mode = 2}:
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        size: max._core.Value[TensorType],
        coordinate_transform_mode: CoordinateTransformModeAttr,
        round_mode: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def size(self) -> max._core.Value[TensorType]: ...
    @property
    def coordinate_transform_mode(self) -> CoordinateTransformMode: ...
    @coordinate_transform_mode.setter
    def coordinate_transform_mode(
        self, arg: CoordinateTransformModeAttr, /
    ) -> None: ...
    @property
    def round_mode(self) -> int: ...
    @round_mode.setter
    def round_mode(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RoiAlignOp(max._core.Operation):
    """
    ROI align consumes an input tensor and regions of interest in which to apply pooling.

    Example:
    ```mlir
      %inp: !mo.tensor<[1, 10, 10, 1], f32>
      %rois: !mo.tensor<[1, 5], f32>
      %output_height = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %spatial_scale = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
      %sampling_ratio = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

      %res = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
        {aligned = false,  mode = "AVG"}
        : (!mo.tensor<[1, 10, 10, 1], f32>,
          !mo.tensor<[1, 5], f32>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], f32>,
          !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        rois: max._core.Value[TensorType],
        output_height: max._core.Value[TensorType],
        output_width: max._core.Value[TensorType],
        spatial_scale: max._core.Value[TensorType],
        sampling_ratio: max._core.Value[TensorType],
        aligned: max._core.dialects.builtin.BoolAttr,
        mode: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def rois(self) -> max._core.Value[TensorType]: ...
    @property
    def output_height(self) -> max._core.Value[TensorType]: ...
    @property
    def output_width(self) -> max._core.Value[TensorType]: ...
    @property
    def spatial_scale(self) -> max._core.Value[TensorType]: ...
    @property
    def sampling_ratio(self) -> max._core.Value[TensorType]: ...
    @property
    def aligned(self) -> bool: ...
    @aligned.setter
    def aligned(self, arg: max._core.dialects.builtin.BoolAttr, /) -> None: ...
    @property
    def mode(self) -> max._core.dialects.builtin.StringAttr: ...
    @mode.setter
    def mode(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RoundOp(max._core.Operation):
    """
    Returns the elementwise nearest integer, with ties going towards the
    nearest even number.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class RsqrtOp(max._core.Operation):
    """
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class ScatterAddOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the sum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via addition.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterMaxOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the maximum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via maximum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterMinOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the minimum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via minimum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterMulOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the product of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via multiplication.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdAddOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdMaxOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdMinOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdMulOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterNdOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ScatterOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar `axis` compile-time `index` attribute. The output is a copy of the
    input, with certain elements updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is updated to the corresponding entry in `updates`.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter(%inputs, %updates, %indices) {axis = 0 : index} : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        updates: max._core.Value[TensorType],
        indices: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def updates(self) -> max._core.Value[TensorType]: ...
    @property
    def indices(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SelectOp(max._core.Operation):
    """
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        cond: max._core.Value[TensorType],
        x: max._core.Value[TensorType],
        y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def cond(self) -> max._core.Value[TensorType]: ...
    @property
    def x(self) -> max._core.Value[TensorType]: ...
    @property
    def y(self) -> max._core.Value[TensorType]: ...

class SequenceOp(max._core.Operation):
    """
    The `mo.sequence` operation wraps a region of ops that should be bound to a
    single device stream selected by `streamId` (0 is the default stream). It
    exists for two reasons: (1) the region is a fusion boundary -- ops inside it
    are not fused with ops outside it -- and (2) it marks which ops should be
    enqueued on the side stream, so the MOGG->MGP lowering can bind the body's
    device context to a `mgp.device_context.select_stream` view.

    Operands are a plain variadic list (tensors / chains) mapped 1:1 into the
    body's block arguments, and `mo.yield` maps 1:1 to the op's results.
    The body is a graph region (unordered); chains encode any mutable ordering
    exactly as in the enclosing graph.

    Example:
    ```mlir
    %out = mo.sequence[1] (%a : !mo.tensor<[3], f32, gpu:0>) (%arg)
        -> (!mo.tensor<[3], f32, gpu:0>) {
      %1 = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
      mo.yield %1 : !mo.tensor<[3], f32, gpu:0>
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        inputs: Sequence[max._core.Value[max._core.Type]],
        stream_id: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        inputs: Sequence[max._core.Value[max._core.Type]],
        result_types: Sequence[max._core.Type],
        stream_id: int,
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def stream_id(self) -> int: ...
    @stream_id.setter
    def stream_id(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...

class ShapeOfOp(max._core.Operation):
    """
    Returns the shape of a tensor.

    Examples:

    ```mlir
      // statically ranked
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape1 = mo.shape_of(%arg1) : !mo.tensor<[3], si64>

      // dynamically ranked
      %arg2: !mo.tensor<?, f32>
      %shape2 = mo.shape_of(%arg2) : !mo.tensor<[?], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: TensorType,
        input: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        width: int,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SigmoidOp(max._core.Operation):
    """
    Returns the sigmoid activation `1 / (1 + exp(-x))`, where `x` is the input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class SiluOp(max._core.Operation):
    """
    Returns the SiLU (Swish) activation `x * sigmoid(x)`, where `x` is the input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.silu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class SinOp(max._core.Operation):
    """
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class SliceOp(max._core.Operation):
    """
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        start: max._core.Value[TensorType],
        stop: max._core.Value[TensorType],
        step: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def start(self) -> max._core.Value[TensorType]: ...
    @property
    def stop(self) -> max._core.Value[TensorType]: ...
    @property
    def step(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ReduceSoftmaxOp(max._core.Operation):
    """
    Returns `exp(input) / sum(exp(input))`, where `x` is input tensor.

    The `sum` reduction is applied along `axis`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.softmax(%arg, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class SplitDimOp(max._core.Operation):
    """
    Splits tensor at `axis` into two dimensions. Example:
    Input=[N, K], Axis=0

    Output=[S1, S2, K], where S1 = N / S2.

    Value of S2 is taken from the output shape.

    Example:
    ```mlir
      %out = mo.split_dim[0](%res): (!mo.tensor<[4, 9], f32>) -> !mo.tensor<[2, 2, 9], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class SplitOp(max._core.Operation):
    """
    Splits the input tensor into multiple tensors along a given dimension.

    `mo.split` splits the tensor `input` into multiple output tensors.
    The number of output tensors is equal to the number of elements in
    `splitSizes`, which is a rank-1 tensor of integers.
    Each of the output tensors has the same shape as `input` except along the
    split dimension `axis`, where the size is given by the corresponding
    element in `splitSizes`.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 8], f32>
      %splitSizes = mo.constant {
        value = #M.dense_array<3, 5> : tensor<2xsi64>
      } : !mo.tensor<[2], si64>
      %res:2 = mo.split(%input, %splitSizes) {axis = 1 : index} : (
        !mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>
      ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        input: max._core.Value[TensorType],
        split_sizes: max._core.Value[TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def split_sizes(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SqrtOp(max._core.Operation):
    """
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class SqueezeShapeOp(max._core.Operation):
    """
    Calculates the shape from squeeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N - M`. The
    indices represent the 0-based index of dimensions in the original rank `N`
    tensor.

    The indicated indices must represent dimensions of size 1. If
    an index does not point to a dimension to size 1, an error is thrown
    instead.

    This operator supports negative indexing with python-like semantics.
    That is all indices must be in [-N, N), if an index is < 0, it is as if
    we added `N` to it.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_shape: max._core.Value[TensorType],
        remove_indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def remove_indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SubOp(max._core.Operation):
    """
    Returns `x - y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.sub(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class TanhOp(max._core.Operation):
    """
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class TensorBundleOp(max._core.Operation):
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: BundleType,
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        inputs: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class TensorUnbundleOp(max._core.Operation):
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Type],
        input: max._core.Value[BundleType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[BundleType]: ...

class TileOp(max._core.Operation):
    """
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        repeats: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def repeats(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TopKOp(max._core.Operation):
    """
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis. Returned values along the axis are always sorted
    (stable).

    axis: The axis to compute the largest values over.
      The axis must be in [-rank, rank).
    k: The number of values to compute.
    sorted: Whether to return the values and indices sorted or not.

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant() { value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = mo.top_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: TensorType,
        indices: TensorType,
        input: max._core.Value[TensorType],
        _k: max._core.Value[TensorType],
        axis: max._core.Value[TensorType],
        sorted: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def _k(self) -> max._core.Value[TensorType]: ...
    @property
    def axis(self) -> max._core.Value[TensorType]: ...
    @property
    def sorted(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TransferOp(max._core.Operation):
    """
    This op represents a possible copy or aliasing operation to make the
    contents of the operand tensor available on the (virtual) device of the
    result tensor.

    It is valid for the source and destination devices to be identical. If the
    `alwaysElideSameDeviceCopy` flag is not set, it is implementation defined as
    to whether the result tensor is a copy or alias of the operand tensor; if
    this flag is true, transfers to the same device never result in a copy.

    Example:

    ```mlir
      %arg : !mo.tensor<[N, 8], f32, gpu:3>
      %onOtherDevice = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:3> to <"gpu", 1>
      %onHost = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:0> to <"cpu", 1>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        out_chain: ChainType,
        input: max._core.Value[TensorType],
        always_elide_same_device_copy: max._core.dialects.builtin.BoolAttr,
        in_chain: max._core.Value[ChainType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[TensorType],
        dest_device: max._core.dialects.m.DeviceRefAttr,
        in_chain: max._core.Value[ChainType],
        always_elide_same_device_copy: bool = True,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def always_elide_same_device_copy(self) -> bool: ...
    @always_elide_same_device_copy.setter
    def always_elide_same_device_copy(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[ChainType]: ...

class TransposeOp(max._core.Operation):
    """
    Returns a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Note that `perm` must contain unique values from `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
        perm: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...
    @property
    def perm(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TruncOp(max._core.Operation):
    """
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[TensorType]: ...

class UnsqueezeShapeOp(max._core.Operation):
    """
    Calculates the shape from unsqueeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N + M`.

    The indices in the given list map to the new vector of length `N + M` where
    the indicated dimensions are replaced with `1`. The remaining dimension in
    the original input shape vector are copied over in the non-1 dimensions.

    This operator supports negative indexing.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[3], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_shape: max._core.Value[TensorType],
        padding_indices: max._core.Value[TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @property
    def input_shape(self) -> max._core.Value[TensorType]: ...
    @property
    def padding_indices(self) -> max._core.Value[TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class XorOp(max._core.Operation):
    """
    Returns `x xor y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: TensorType,
        input_x: max._core.Value[TensorType],
        input_y: max._core.Value[TensorType],
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[TensorType]: ...

class YieldOp(max._core.Operation):
    """
    This op specifies the output values for control flow blocks. The op
    takes variable number of operands and produces no results.

    Example:

    ```mlir
      mo.if $cond : !mo.tensor<[], bool> (!mo.tensor<?, f32>) {
        mo.yield %arg0 : !mo.tensor<?, f32>
      } else {
        mo.yield %arg1 : !mo.tensor<?, f32>
      }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        parameters: max._core.dialects.kgen.ParameterExprArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def parameters(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @parameters.setter
    def parameters(
        self, arg: max._core.dialects.kgen.ParameterExprArrayAttr, /
    ) -> None: ...

class ParamExprBuilder: ...
class ShapeMaterializeResult: ...
