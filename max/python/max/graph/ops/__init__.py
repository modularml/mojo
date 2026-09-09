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
"""Implements operations used when staging a graph.

This module provides operations for building a :class:`~max.graph.Graph` in
MAX. Most operations return a :class:`~max.graph.TensorValue`, which supports
standard Python operators such as ``+``, ``*``, and ``@`` (matrix
multiplication), as well as convenience methods like
:meth:`~max.graph.TensorValue.reshape` and
:meth:`~max.graph.TensorValue.flatten`. Ops like
:func:`~max.graph.ops.constant` can also add constant values to your graph.

When an operation receives inputs with different data types
(:class:`~max.dtype.DType`), MAX promotes them to a common type before
computing the result. To avoid silently widening a type and hurting
performance, the common type is always one of the input types; MAX never
invents a new, wider type.

To choose between the input types, MAX ranks each one along two axes:

- Category, ordered ``bool < unsigned int < signed int < float``.
- Bit width (for example, ``8``, ``16``, ``32``, or ``64`` bits).

The common type is the input type with the highest category and the largest
bit width. For example, promoting ``int8`` and ``float16`` yields ``float16``:
``float`` outranks ``signed int``, and ``16`` bits is wider than ``8``.

If an input can't be safely represented in the chosen type, MAX raises an
error rather than widening to a different type. For example, promoting
``uint8`` and ``int8`` selects ``int8`` (``signed int`` outranks
``unsigned int`` at the same bit width), but ``int8`` can't represent the
largest ``uint8`` values, so MAX raises an error.

When inputs have different shapes, MAX broadcasts them to a common shape.
Shapes are aligned from the trailing dimension, and each pair of dimensions
must either match exactly, be ``1``, or be absent. Size-1 dimensions (and
any missing leading dimensions) are stretched to match the corresponding
dimension of the other input. If the shapes can't be reconciled under
these rules, MAX raises an error."""

from __future__ import annotations

# Import types for type annotations
from ..value import TensorValue, TensorValueLike
from . import (
    allreduce,
    bundled_allreduce,
    distributed_ep,
    random,
    reducescatter,
)
from .allgather import allgather
from .allgather_rms_norm import (
    allgather_rms_norm,
    allgather_rms_norm_quant_mxfp8,
)
from .argsort import argsort
from .band_part import band_part
from .bottom_k import bottom_k
from .broadcast import distributed_broadcast
from .broadcast_to import broadcast_to
from .buffer import buffer_create, buffer_load, buffer_store, buffer_store_slice
from .call import call
from .cast import cast
from .chunk import chunk
from .complex import as_interleaved_complex
from .concat import concat
from .conditional import cond
from .constant import constant, constant_external
from .conv import conv2d, conv3d
from .conv_transpose import conv2d_transpose
from .cumsum import cumsum
from .custom import custom, inplace_custom
from .debug import print
from .distributed_scatter import distributed_scatter
from .elementwise import *

# `floor_div` is re-exported by name (not only via the `*` above) to make its
# public-API visibility explicit; `elementwise` defines no `__all__`.
from .elementwise import floor_div
from .elementwise import max as _elementwise_max
from .elementwise import min as _elementwise_min
from .flatten import flatten
from .fold import fold
from .gather import gather, gather_nd
from .group_norm import group_norm
from .hann_window import hann_window
from .irfft import irfft
from .layer_norm import layer_norm
from .matmul import matmul
from .nms import non_maximum_suppression
from .nonzero import nonzero
from .outer import outer
from .pad import pad
from .parallel import parallel
from .permute import permute
from .pooling import avg_pool2d, max_pool2d, roi_align
from .quantized import dequantize, qmatmul
from .range import range
from .rebind import rebind
from .reduce_scatter_rms_norm import reduce_scatter_rms_norm
from .reduction import argmax, argmin, mean, prod, sum
from .reduction import max as _reduce_max
from .reduction import min as _reduce_min
from .repeat_interleave import repeat_interleave
from .reshape import reshape
from .resize import (
    InterpolationMode,
    resize,
    resize_bicubic,
    resize_linear,
    resize_nearest,
)
from .rms_norm import rms_norm
from .scatter import (
    masked_scatter,
    scatter,
    scatter_add,
    scatter_max,
    scatter_min,
    scatter_mul,
    scatter_nd,
    scatter_nd_add,
    scatter_nd_max,
    scatter_nd_min,
    scatter_nd_mul,
)
from .sequence import side_stream
from .shape_to_tensor import shape_to_tensor
from .shard_and_stack import shard_and_stack
from .slice_tensor import slice_tensor
from .split import split
from .squeeze import squeeze
from .stack import stack
from .tile import tile
from .top_k import top_k
from .transfer_to import transfer_to
from .transpose import transpose
from .unsqueeze import unsqueeze
from .where import where
from .while_loop import while_loop


def min(  # type: ignore[no-redef]
    x: TensorValueLike,
    y: TensorValueLike | None = None,
    /,
    axis: int | None = None,
) -> TensorValue:
    """Overload for ops.elementwise.min and ops.reduction.min.

    - If two tensors are provided, `axis` is ignored and returns an elementwise minimum.
    - If one tensor is provided, compute `ops.reduction.min` on the tensor and axis.
    """
    if y is not None and axis is not None:
        raise ValueError("Axis not allowed for elementwise min.")
    axis = -1 if axis is None else axis
    return _reduce_min(x, axis=axis) if y is None else _elementwise_min(x, y)


def max(  # type: ignore[no-redef]
    x: TensorValueLike,
    y: TensorValueLike | None = None,
    /,
    axis: int | None = None,
) -> TensorValue:
    """Overload for ops.elementwise.max and ops.reduction.max.

    - If two tensors are provided, `axis` is ignored and returns an elementwise maximum.
    - If one tensor is provided, compute `ops.reduction.max` on the tensor and axis.
    """
    if y is not None and axis is not None:
        raise ValueError("Axis not allowed for elementwise max.")
    axis = -1 if axis is None else axis
    return _reduce_max(x, axis=axis) if y is None else _elementwise_max(x, y)
