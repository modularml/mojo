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

"""Distributed functional ops with explicit per-op SPMD dispatch.

Usage::

    from max.experimental import functional as F

    y = F.matmul(a, b)
    z = F.add(x, y)
    w = F.transfer_to(z, new_mapping)

Layout:

- :mod:`.spmd_ops` -- ``per_shard_dispatch`` engine and per-op functions.
- :mod:`.collective_ops` -- collectives and ``transfer_to``.
- :mod:`.creation_ops` -- ``full`` / ``ones`` / ``zeros`` and friends.
- :func:`custom` / :func:`inplace_custom` live here because they
  combine graph ops with extension loading.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from max._mlir_context import MLIRThreadPoolExecutor
from max.driver import Device
from max.dtype import DType
from max.experimental.realization_context import (
    ensure_context,
    in_graph_context,
    lazy,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType, Type, ops

from .collective_ops import (
    allgather,
    allreduce_sum,
    distributed_broadcast,
    reduce_scatter,
    transfer_to,
)
from .creation_ops import (
    arange,
    constant,
    constant_external,
    full,
    full_like,
    gaussian,
    gaussian_like,
    hann_window,
    normal,
    normal_like,
    ones,
    ones_like,
    range,
    uniform,
    uniform_like,
    zeros,
    zeros_like,
)
from .print import print
from .spmd_ops import (
    abs,
    acos,
    add,
    any_distributed,
    argmax,
    argmin,
    argsort,
    as_interleaved_complex,
    atanh,
    avg_pool2d,
    band_part,
    bottom_k,
    broadcast_to,
    buffer_store,
    buffer_store_slice,
    cast,
    ceil,
    chunk,
    clamp,
    clip,
    complex_mul,
    concat,
    cond,
    conv2d,
    conv2d_transpose,
    conv3d,
    cos,
    cumsum,
    dequantize,
    div,
    elementwise_max,
    elementwise_min,
    equal,
    erf,
    exp,
    flatten,
    floor,
    floor_div,
    fold,
    functional,
    gather,
    gather_nd,
    gelu,
    greater,
    greater_equal,
    group_norm,
    irfft,
    is_inf,
    is_nan,
    layer_norm,
    log,
    log1p,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    logsoftmax,
    map_tensors,
    masked_scatter,
    matmul,
    max,
    max_pool2d,
    mean,
    min,
    mod,
    mul,
    negate,
    non_maximum_suppression,
    nonzero,
    not_equal,
    outer,
    pad,
    per_shard_dispatch,
    permute,
    pow,
    prod,
    qmatmul,
    rebind,
    relu,
    repeat_interleave,
    reshape,
    resize,
    resize_bicubic,
    resize_linear,
    resize_nearest,
    rms_norm,
    roi_align,
    round,
    rsqrt,
    scatter_add,
    scatter_max,
    scatter_min,
    scatter_mul,
    scatter_nd,
    scatter_nd_add,
    scatter_nd_max,
    scatter_nd_min,
    scatter_nd_mul,
    sigmoid,
    silu,
    sin,
    slice_tensor,
    softmax,
    split,
    sqrt,
    squeeze,
    stack,
    sub,
    sum,
    tanh,
    tensor_to_layout,
    tile,
    to_tensors,
    top_k,
    transpose,
    trunc,
    unsqueeze,
    where,
    while_loop,
)

# Per-element scatter (not the collective scatter); shadows the collective one here.
from .spmd_ops import scatter as scatter

__all__ = [
    "abs",
    "acos",
    "add",
    "allgather",
    "allreduce_sum",
    "any_distributed",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "as_interleaved_complex",
    "atanh",
    "avg_pool2d",
    "band_part",
    "bottom_k",
    "broadcast_to",
    "buffer_store",
    "buffer_store_slice",
    "cast",
    "ceil",
    "chunk",
    "clamp",
    "clip",
    "complex_mul",
    "concat",
    "cond",
    "constant",
    "constant_external",
    "conv2d",
    "conv2d_transpose",
    "conv3d",
    "cos",
    "cumsum",
    "custom",
    "dequantize",
    "distributed_broadcast",
    "div",
    "elementwise_max",
    "elementwise_min",
    "ensure_context",
    "equal",
    "erf",
    "exp",
    "flatten",
    "floor",
    "floor_div",
    "fold",
    "full",
    "full_like",
    "functional",
    "gather",
    "gather_nd",
    "gaussian",
    "gaussian_like",
    "gelu",
    "greater",
    "greater_equal",
    "group_norm",
    "hann_window",
    "in_graph_context",
    "inplace_custom",
    "irfft",
    "is_inf",
    "is_nan",
    "layer_norm",
    "lazy",
    "log",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logsoftmax",
    "map_tensors",
    "masked_scatter",
    "matmul",
    "max",
    "max_pool2d",
    "mean",
    "min",
    "mod",
    "mul",
    "negate",
    "non_maximum_suppression",
    "nonzero",
    "normal",
    "normal_like",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "pad",
    "per_shard_dispatch",
    "permute",
    "pow",
    "print",
    "prod",
    "qmatmul",
    "range",
    "rebind",
    "reduce_scatter",
    "relu",
    "repeat_interleave",
    "reshape",
    "resize",
    "resize_bicubic",
    "resize_linear",
    "resize_nearest",
    "rms_norm",
    "roi_align",
    "round",
    "rsqrt",
    "scatter",
    "scatter_add",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "scatter_nd",
    "scatter_nd_add",
    "scatter_nd_max",
    "scatter_nd_min",
    "scatter_nd_mul",
    "sigmoid",
    "silu",
    "sin",
    "slice_tensor",
    "softmax",
    "split",
    "sqrt",
    "squeeze",
    "stack",
    "sub",
    "sum",
    "tanh",
    "tensor_to_layout",
    "tile",
    "to_tensors",
    "top_k",
    "transfer_to",
    "transpose",
    "trunc",
    "uniform",
    "uniform_like",
    "unsqueeze",
    "where",
    "while_loop",
    "zeros",
    "zeros_like",
]

_Result = TypeVar("_Result")


def _in_running_loop() -> bool:
    """Returns ``True`` when called inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run(coro: Coroutine[Any, Any, _Result]) -> _Result:
    """Runs ``coro`` synchronously, even from inside an event loop.

    Outside an event loop, uses ``asyncio.run``. Inside one (for
    example in Jupyter), runs the coroutine on a worker thread.
    """
    if not _in_running_loop():
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    with MLIRThreadPoolExecutor() as pool:
        fut = pool.submit(loop.run_until_complete, coro)
    return fut.result()


def _load_custom_extensions(
    custom_extensions: str | Path | Sequence[str | Path] | None,
) -> None:
    """Loads custom-kernel extensions into the current graph."""
    if custom_extensions is None:
        return
    graph = Graph.current
    if isinstance(custom_extensions, (str, Path)):
        custom_extensions = [custom_extensions]
    paths = [Path(p) for p in custom_extensions]
    graph._import_kernels(paths)


def custom(
    name: str,
    device: Device | DeviceRef,
    values: Sequence[Any],
    out_types: Sequence[Type[Any]],
    parameters: Mapping[str, bool | int | str | DType] | None = None,
    custom_extensions: str | Path | Sequence[str | Path] | None = None,
) -> list[Tensor]:
    """Calls a custom op, optionally loading custom Mojo extensions first.

    Args:
        name: The registered name of the custom op.
        device: The device on which to execute the op.
        values: The input values passed to the op.
        out_types: The expected output types.
        parameters: Optional compile-time parameters for the op.
        custom_extensions: Optional path or sequence of paths to custom
            Mojo extensions (``.mojoc`` or ``.mojo`` sources) to load
            before invoking the op.

    Returns:
        A list of tensors produced by the custom op.
    """
    with ensure_context():
        _load_custom_extensions(custom_extensions)
        return [
            Tensor.from_graph_value(v)
            for v in ops.custom(
                name=name,
                device=device,
                values=values,
                out_types=out_types,
                parameters=parameters,
            )
        ]


def inplace_custom(
    name: str,
    device: Device | DeviceRef,
    values: Sequence[Any],
    out_types: Sequence[Type[Any]] | None = None,
    parameters: dict[str, bool | int | str | DType] | None = None,
    custom_extensions: str | Path | Sequence[str | Path] | None = None,
) -> list[Tensor]:
    """Calls an in-place custom op that mutates one or more of its inputs.

    Like :func:`custom`, but for ops that mutate buffer values rather
    than returning new tensors.

    Args:
        name: The registered name of the custom op.
        device: The device on which to execute the op.
        values: The input values; one or more are mutated in place.
        out_types: Optional expected output types. Most in-place ops
            return no outputs and can leave this as :obj:`None`.
        parameters: Optional compile-time parameters for the op.
        custom_extensions: Optional path or sequence of paths to custom
            Mojo extensions to load before invoking the op.

    Returns:
        A list of tensors produced by the custom op, or an empty list
        when the op produces no outputs.
    """
    with ensure_context():
        _load_custom_extensions(custom_extensions)
        result = ops.inplace_custom(
            name=name,
            device=device,
            values=values,
            out_types=out_types,
            parameters=parameters,
        )
        return [Tensor.from_graph_value(v) for v in result] if result else []
