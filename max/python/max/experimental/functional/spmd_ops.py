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

"""SPMD op dispatch — explicit per-op wiring of graph ops and sharding rules.

Each op is an explicit function that:

1. Calls the sharding rule (with ``tensor_to_layout()`` to convert Tensors to TensorLayouts).
2. Redistributes tensors to match the rule's suggestions.
3. Dispatches per-shard via ``per_shard_dispatch``.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from max.driver import CPU, Buffer
from max.experimental import tensor
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    TensorLayout,
)
from max.experimental.sharding.per_shard_dim import global_dim
from max.experimental.tensor import Tensor
from max.graph import ShapeLike, TensorValue, TensorValueLike, Type, ops
from max.graph.dim import Dim, DimLike, StaticDim
from max.graph.ops.slice_tensor import SliceIndices
from max.graph.quantization import QuantizationEncoding

from ..sharding import (
    ActionSet,
    PerShard,
    mode,
)
from ..sharding.mode import ShardingError
from ..sharding.rules import (
    argsort_rule,
    as_interleaved_complex_rule,
    band_part_rule,
    binary_rule,
    broadcast_to_rule,
    buffer_store_rule,
    buffer_store_slice_rule,
    chunk_rule,
    cond_rule,
    conv2d_rule,
    conv2d_transpose_rule,
    conv3d_rule,
    dequantize_rule,
    flatten_rule,
    fold_rule,
    gather_nd_rule,
    gather_rule,
    irfft_rule,
    layer_norm_rule,
    linear_binary_rule,
    linear_pool_rule,
    linear_reduce_rule,
    linear_unary_rule,
    masked_scatter_rule,
    matmul_rule,
    mean_rule,
    nonzero_rule,
    outer_rule,
    pad_rule,
    permute_rule,
    pool_rule,
    qmatmul_rule,
    rebind_rule,
    reduce_rule,
    repeat_interleave_rule,
    reshape_rule,
    resize_bicubic_rule,
    resize_linear_rule,
    resize_nearest_rule,
    resize_rule,
    rms_norm_rule,
    same_placement_multi_input_rule,
    scatter_add_rule,
    scatter_nd_add_rule,
    scatter_nd_rule,
    scatter_rule,
    slice_tensor_rule,
    softmax_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    ternary_rule,
    tile_rule,
    top_k_rule,
    transpose_rule,
    unary_rule,
    unsqueeze_rule,
    while_loop_rule,
)
from ._signatures import install_tensor_signature
from .creation_ops import full_like

# Re-exported; user-facing factory lives in ``max.experimental.sharding``.
__all__ = [
    "ShardingError",
    "any_distributed",
    "map_tensors",
    "mode",
    "to_tensors",
]


def to_tensors(values: Any) -> Any:
    """Converts graph op results to :class:`Tensor`, preserving container type.

    Recurses one level into ``list`` and ``tuple`` containers; unknown
    types pass through unchanged. Returns ``Tensor`` for ``Buffer`` and
    ``TensorValue`` leaves, and a same-shape container for list/tuple
    inputs (each leaf converted independently). ``Any`` reflects that
    leaves change type while the container type is preserved.
    """

    def _one(value: Any) -> Tensor | Any:
        if isinstance(value, Tensor):
            return value
        if isinstance(value, Buffer):
            return Tensor(storage=value)
        if isinstance(value, TensorValue):
            return Tensor.from_graph_value(value)
        return value

    if values is None:
        return None
    if isinstance(values, (Buffer, Tensor, TensorValue)):
        return _one(values)
    if isinstance(values, (list, tuple)):
        return type(values)(_one(v) for v in values)
    return values


def map_tensors(
    fn: Callable[[Tensor], Any], args: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Applies ``fn`` to every :class:`Tensor` leaf in ``args``.

    Recurses into ``list`` and ``tuple`` containers; non-tensor leaves
    pass through unchanged.
    """

    def _walk(x: Any) -> Any:
        if isinstance(x, Tensor):
            return fn(x)
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        return x

    return tuple(_walk(a) for a in args)


def tensor_to_layout(t: Tensor) -> TensorLayout:
    """Converts a :class:`Tensor` to a :class:`TensorLayout` for sharding-rule evaluation.

    ``t.shape`` already carries per-device cells on :class:`Sharded` axes
    (via :class:`PerShardDim`), so the rules that fold per-rank cells
    (notably ``reshape_rule``) can do the correct shape arithmetic
    directly. Non-distributed tensors fall back to a plain :class:`Shape`.
    """
    if t.is_distributed:
        return TensorLayout(
            t.dtype,
            t.shape,
            PlacementMapping(t.mesh, t.placements),
        )
    return TensorLayout(
        t.dtype,
        t.shape,
        PlacementMapping(DeviceMesh.single(t.device), (Replicated(),)),
    )


def any_distributed(args: tuple[object, ...]) -> bool:
    """True if any :class:`Tensor` in ``args`` is distributed (multi-device)."""
    for a in args:
        if isinstance(a, Tensor) and a.is_distributed:
            return True
        if isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Tensor) and item.is_distributed:
                    return True
    return False


def per_shard_dispatch(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    output_mappings: tuple[DeviceMapping, ...],
    filtered_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Runs ``graph_op`` once per shard and reassembles distributed outputs.

    Args:
        graph_op: The per-rank graph op to run.
        args: Already-redistributed args.
        output_mappings: One :class:`DeviceMapping` per output.
        filtered_kwargs: Non-tensor or non-distributed tensor keyword arguments.
    """
    mesh = output_mappings[0].mesh

    with ensure_context():
        per_shard = _run_per_shard(
            graph_op, args, mesh.num_devices, filtered_kwargs
        )
        first = per_shard[0]
        if first is None:
            return None

        multi = isinstance(first, (list, tuple))
        num_out = len(first) if multi else 1
        outputs = [
            _reassemble_output(
                per_shard,
                j,
                output_mappings[builtins.min(j, len(output_mappings) - 1)],
                multi=multi,
            )
            for j in builtins.range(num_out)
        ]
        return type(first)(outputs) if multi else outputs[0]


def _run_per_shard(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    num_devices: int,
    filtered_kwargs: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Calls ``graph_op`` once per shard with per-rank arg unwrapping."""
    per_shard: list[Any] = []
    if filtered_kwargs is None:
        filtered_kwargs = {}

    for i in builtins.range(num_devices):

        def _per_rank(t: Tensor, _i: int = i) -> TensorValue:
            return (
                TensorValue(t.local_shards[_i])
                if t.is_distributed
                else TensorValue(t)
            )

        shard_args = map_tensors(_per_rank, args)
        shard_args = tuple(
            a[i] if isinstance(a, PerShard) else a for a in shard_args
        )
        per_shard.append(graph_op(*shard_args, **filtered_kwargs))
    return per_shard


def _reassemble_output(
    per_shard: Sequence[Any],
    j: int,
    out_mapping: DeviceMapping,
    *,
    multi: bool,
) -> Tensor:
    """Reassembles output ``j`` from per-shard results into one distributed Tensor."""
    tvs = [TensorValue(s[j] if multi else s) for s in per_shard]
    return Tensor.from_shard_values(tvs, out_mapping)


def functional(
    graph_op: Callable[..., Any],
    rule: Callable[..., ActionSet] | None = None,
) -> Callable[..., Any]:
    """Wraps a graph op as a distributed dispatch entry.

    Returns a callable that local-auto-shards when any argument is a
    distributed :class:`Tensor` (and a rule is bound), and otherwise
    forwards to the bare ``graph_op``. The returned wrapper carries
    ``graph_op`` and ``rule`` as attributes; reassign ``wrapper.rule``
    to swap the sharding rule at runtime without re-wrapping.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        active_rule = getattr(wrapper, "rule", None)
        if any_distributed(args) and active_rule is not None:
            return _local_dispatch(graph_op, active_rule, args, kwargs)
        with ensure_context():
            return to_tensors(graph_op(*args, **kwargs))

    # ``Any``-typed alias so attribute writes are dynamic;
    # ``functools.wraps`` types the closure as ``_Wrapped[...]`` which
    # rejects arbitrary attribute assignment under mypy.
    w: Any = wrapper
    w.__name__ = getattr(graph_op, "__name__", "wrapper")
    w.__qualname__ = getattr(graph_op, "__qualname__", w.__name__)
    w.__module__ = getattr(graph_op, "__module__", w.__module__)
    w.__wrapped__ = graph_op
    w.graph_op = graph_op
    w.rule = rule
    # Rewrite the wrapper's signature/annotations so inspect.signature and
    # Sphinx show ``Tensor`` instead of the graph op's ``TensorValueLike``
    # parameter types (reads graph_op via ``__wrapped__``). From #87216.
    install_tensor_signature(wrapper)
    return wrapper


def _local_dispatch(
    graph_op: Callable[..., Any],
    rule: Callable[..., ActionSet],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Picks one :class:`Action` for this call and applies it."""
    from max.experimental.sharding._diagnostics import report_reshard
    from max.experimental.sharding.mode import current_solver

    # TODO: keyword-only distributed tensor arguments are not supported.
    flat_args, filtered_kwargs = _canonicalize_call(graph_op, args, kwargs)
    layout_args = map_tensors(tensor_to_layout, flat_args)
    in_layouts = _walk_tensor_layouts(layout_args)

    menu = rule(*layout_args)
    solver = current_solver()
    action = solver(menu, in_layouts)

    op_name = getattr(graph_op, "__name__", "<op>")
    report_reshard(solver, op_name, layout_args, menu, action)
    redistributed = _transfer_args(flat_args, action.inputs)

    if action.outputs:
        out_mappings = action.outputs
    else:
        out_mappings = (
            next(
                t.mapping for t in _walk_tensors(flat_args) if t.is_distributed
            ),
        )
    return per_shard_dispatch(
        graph_op,
        redistributed,
        out_mappings,
        filtered_kwargs,
    )


def _canonicalize_call(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Normalizes ``args`` + ``kwargs`` into a positional tuple.

    Binds against ``graph_op``'s signature so kwargs become positional.
    Falls back to ``args`` when the signature is uninspectable.
    """
    import inspect

    sig_source = getattr(graph_op, "graph_op", graph_op)
    try:
        bound = inspect.signature(sig_source).bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.args), bound.kwargs
    except (TypeError, NotImplementedError, ValueError):
        return args + tuple(kwargs.values()), {}


def _walk_tensors(value: Any) -> Iterable[Tensor]:
    """Yields every :class:`Tensor` reachable through tuples/lists."""
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_tensors(v)


def _walk_tensor_layouts(value: Any) -> list[Any]:
    """Flattens TensorLayout leaves out of arbitrary nested args."""
    out: list[Any] = []
    if isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_walk_tensor_layouts(v))
        return out
    if hasattr(value, "mapping") and hasattr(value, "shape"):
        out.append(value)
    return out


def _transfer_args(
    args: tuple[Any, ...],
    suggested: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Reshards Tensor args to match the action's per-slot mappings."""
    from .collective_ops import transfer_to

    result: list[object] = []
    for orig, sugg in zip(args, suggested, strict=False):
        if isinstance(sugg, PerShard):
            result.append(sugg)
        elif isinstance(orig, Tensor) and isinstance(sugg, DeviceMapping):
            result.append(transfer_to(orig, sugg))
        elif isinstance(orig, (list, tuple)) and isinstance(
            sugg, (list, tuple)
        ):
            items = [
                transfer_to(o, s)
                if isinstance(o, Tensor) and isinstance(s, DeviceMapping)
                else s
                for o, s in zip(orig, sugg, strict=False)
            ]
            result.append(type(orig)(items))
        elif not isinstance(orig, Tensor) and sugg is not None:
            result.append(sugg)
        else:
            result.append(orig)
    if len(args) > len(suggested):
        result.extend(args[len(suggested) :])
    return tuple(result)


def _binary_with_scalar_promotion(
    inner: Callable[..., object],
) -> Callable[..., Tensor]:
    """Wraps a binary dispatch with scalar promotion.

    Scalar promotion is gated on ``any_distributed`` because the
    single-device graph-op path handles scalar + tensor natively. Rank
    differences are not equalized here: broadcasting is handled by the
    RMO dialect per shard, and the placement rules express trailing-axis
    alignment directly.
    """

    def wrapper(lhs: Tensor | float, rhs: Tensor | float) -> Tensor:
        if any_distributed((lhs, rhs)):
            if isinstance(lhs, (int, float)) and isinstance(rhs, Tensor):
                lhs = full_like(rhs, lhs)
            elif isinstance(rhs, (int, float)) and isinstance(lhs, Tensor):
                rhs = full_like(lhs, rhs)
        result = inner(lhs, rhs)
        assert isinstance(result, Tensor)
        return result

    wrapper.__module__ = getattr(inner, "__module__", wrapper.__module__)
    wrapper.__name__ = getattr(inner, "__name__", wrapper.__name__)
    wrapper.__qualname__ = getattr(inner, "__qualname__", wrapper.__qualname__)
    return wrapper


#: Adds two tensors element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.add` for details.
add = _binary_with_scalar_promotion(
    functional(ops.add, rule=linear_binary_rule)
)
add.__doc__ = """Adds two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([10.0, 20.0, 30.0])
    result = F.add(a, b)
    # result is [11.0, 22.0, 33.0]

    # Scalar is auto-promoted to a tensor.
    result = F.add(a, 0.5)
    # result is [1.5, 2.5, 3.5]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A ``Tensor`` containing the element-wise sums.
"""

sub = _binary_with_scalar_promotion(
    functional(ops.sub, rule=linear_binary_rule)
)
sub.__doc__ = """Subtracts two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([10.0, 20.0, 30.0])
    b = Tensor([1.0, 2.0, 3.0])
    result = F.sub(a, b)
    # result is [9.0, 18.0, 27.0]

Args:
    lhs: The minuend (left-hand side) tensor or scalar.
    rhs: The subtrahend (right-hand side) tensor or scalar.

Returns:
    A ``Tensor`` containing the result of ``lhs - rhs`` element-wise.
"""

mul = _binary_with_scalar_promotion(functional(ops.mul, rule=binary_rule))
mul.__doc__ = """Multiplies two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    result = F.mul(a, b)
    # result is [4.0, 10.0, 18.0]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A ``Tensor`` containing the element-wise products.
"""

div = _binary_with_scalar_promotion(functional(ops.div, rule=binary_rule))
div.__doc__ = """Divides two tensors element-wise using true division (Python ``/``).

For integer operands, this performs true division by promoting to float,
matching Python's ``/`` operator behavior. For floating-point operands,
this performs standard floating-point division.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([10.0, 6.0, 3.0])
    b = Tensor([2.0, 3.0, 4.0])
    result = F.div(a, b)
    # result is [5.0, 2.0, 0.75]

Args:
    lhs: The numerator tensor or scalar.
    rhs: The denominator tensor or scalar.

Returns:
    A ``Tensor`` with the broadcast shape containing ``lhs / rhs``
    element-wise. The result has a floating-point dtype for integer
    operands and the promoted dtype for mixed types.
"""

floor_div = _binary_with_scalar_promotion(
    functional(ops.floor_div, rule=binary_rule)
)
floor_div.__doc__ = """Divides two tensors element-wise using floor division (Python ``//``).

The result is rounded toward negative infinity, matching Python's ``//``.
Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor. Integer operands stay in the integer
domain (no ``float64`` promotion), unlike :func:`div`.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    a = Tensor([7, 10, 18], dtype=DType.int32)
    b = Tensor([2, 5, 6], dtype=DType.int32)
    result = F.floor_div(a, b)
    # result is [3, 2, 3]

    # Floating-point operands are supported and still round toward -inf.
    result = F.floor_div(Tensor([7.5, -7.5], dtype=DType.float32), 2.0)
    # result is [3.0, -4.0]

Args:
    lhs: The numerator tensor or scalar.
    rhs: The denominator tensor or scalar.

Returns:
    A ``Tensor`` with the broadcast shape containing the element-wise
    floor division of ``lhs`` by ``rhs``.
"""

pow = _binary_with_scalar_promotion(functional(ops.pow, rule=binary_rule))
pow.__doc__ = """Raises elements of one tensor to the power of another element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([2.0, 3.0, 4.0])
    b = Tensor([3.0, 2.0, 0.5])
    result = F.pow(a, b)
    # result is [8.0, 9.0, 2.0]

Args:
    lhs: The base tensor or scalar.
    rhs: The exponent tensor or scalar.

Returns:
    A ``Tensor`` with the broadcast shape containing ``lhs ** rhs``
    element-wise.
"""

mod = _binary_with_scalar_promotion(functional(ops.mod, rule=binary_rule))
mod.__doc__ = """Computes the element-wise modulus of two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. Skipped: Tensor defaults to bfloat16 on an accelerator, and Metal cannot
   compile a bf16 ``mod``. Remove this skip once MOCO-4826 is fixed.
.. skip: next if(__import__("sys").platform == "darwin", "no bf16 mod on Metal (MOCO-4826)")

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([7.0, 10.0, 15.0])
    b = Tensor([3.0, 4.0, 6.0])
    result = F.mod(a, b)
    # result is [1.0, 2.0, 3.0]

Args:
    lhs: The dividend tensor or scalar.
    rhs: The divisor tensor or scalar.

Returns:
    A ``Tensor`` containing ``lhs % rhs`` element-wise.
"""

#: Negates a tensor element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.negate` for details.
negate = functional(ops.negate, rule=linear_unary_rule)
negate.__doc__ = """Negates a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 2.0])
    result = F.negate(x)
    # result is [1.0, 0.0, -2.0]

Args:
    x: The input tensor.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    negation of each element of ``x``.
"""

relu = functional(ops.relu, rule=unary_rule)
relu.__doc__ = """Applies the ReLU (Rectified Linear Unit) activation element-wise.

ReLU is defined as ``relu(x) = max(0, x)``, meaning negative values are set
to zero while positive values are unchanged.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    result = F.relu(x)
    # result is [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]

Args:
    x: The input to the ReLU computation.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing ``x`` with
    its negative elements replaced by ``0``.
"""

abs = functional(ops.abs, rule=unary_rule)
abs.__doc__ = """Computes the absolute value of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = F.abs(x)
    # result is [2.0, 1.0, 0.0, 1.0, 2.0]

Args:
    x: The input tensor.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    absolute value of each element of ``x``.
"""

exp = functional(ops.exp, rule=unary_rule)
exp.__doc__ = """Computes the exponential of a tensor element-wise.

This applies ``exp(x) = e^x``, where ``e`` is Euler's number.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 1.0, 2.0])
    result = F.exp(x)
    # result is approximately [1.0, 2.718, 7.389]

Args:
    x: The input to the exponential function. Must have a floating-point
        dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing ``e``
    raised to the power of each element of ``x``.
"""

log = functional(ops.log, rule=unary_rule)
log.__doc__ = """Computes the natural logarithm of a tensor element-wise.

This applies ``log(x)``. It is the inverse of the exponential
function ``x = e^y``, where ``e`` is Euler's number.
Note that ``log(x)`` is undefined for ``x <= 0`` and complex numbers
are not currently supported.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 2.718, 7.389, 20.0])
    result = F.log(x)
    # result is approximately [0.0, 1.0, 2.0, 2.996]

Args:
    x: The input to the log computation. Must have a floating-point dtype
        and contain positive values only.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    natural logarithm of each element of ``x``.
"""

sqrt = functional(ops.sqrt, rule=unary_rule)
sqrt.__doc__ = """Computes the square root of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 4.0, 9.0, 16.0])
    result = F.sqrt(x)
    # result is [1.0, 2.0, 3.0, 4.0]

Args:
    x: The input tensor. Must have a floating-point dtype. Negative values
        produce ``NaN`` since MAX doesn't support complex numbers.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    square root of each element of ``x``.
"""

rsqrt = functional(ops.rsqrt, rule=unary_rule)
rsqrt.__doc__ = """Computes the reciprocal square root of a tensor element-wise.

Computes ``1 / sqrt(x)`` for each element.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 4.0, 16.0])
    result = F.rsqrt(x)
    # result is [1.0, 0.5, 0.25]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    reciprocal square root of each element of ``x``.
"""

_sigmoid_impl = functional(ops.sigmoid, rule=unary_rule)


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid activation function element-wise.

    Computes ``sigmoid(x) = 1 / (1 + exp(-x))``, mapping all values to the
    range ``(0, 1)``.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
        result = F.sigmoid(x)
        # result is approximately:
        # [[0.119, 0.269, 0.5], [0.731, 0.881, 0.953]]

    Args:
        x: The input to the sigmoid computation. Must have a floating-point
            dtype.

    Returns:
        A ``Tensor`` of the same shape and dtype as ``x`` containing each
        element of ``x`` mapped to the range ``(0, 1)``.
    """
    return _sigmoid_impl(x)


_silu_impl = functional(ops.silu, rule=unary_rule)


def silu(x: Tensor) -> Tensor:
    """Applies the SiLU (Swish) activation function element-wise.

    Computes ``silu(x) = x * sigmoid(x)``.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([-1.0, 0.0, 1.0, 2.0])
        result = F.silu(x)
        # result is approximately [-0.269, 0.0, 0.731, 1.762]

    Args:
        x: The input to the SiLU computation. Must have a floating-point
            dtype.

    Returns:
        A ``Tensor`` of the same shape and dtype as ``x`` containing the
        SiLU activation applied to each element of ``x``.
    """
    return _silu_impl(x)


_gelu_impl = functional(ops.gelu, rule=unary_rule)


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """Applies the GELU (Gaussian Error Linear Unit) activation element-wise.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([-1.0, 0.0, 1.0])
        result = F.gelu(x)
        # result is approximately [-0.159, 0.0, 0.841]

    Args:
        x: The input to the GELU computation. Must have a floating-point
            dtype.
        approximate: The approximation method. Defaults to ``"none"``
            (exact form using ``erf``). Use ``"tanh"`` for the tanh-based
            approximation or ``"quick"`` for the sigmoid-based
            approximation.

    Returns:
        A ``Tensor`` of the same shape and dtype as ``x`` containing the
        GELU activation applied to each element of ``x``.
    """
    return _gelu_impl(x, approximate)


tanh = functional(ops.tanh, rule=unary_rule)
tanh.__doc__ = """Computes the hyperbolic tangent of a tensor element-wise.

This applies ``tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))``, which
maps all values to the range ``(-1, 1)``.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    result = F.tanh(x)
    # result is approximately:
    # [[-0.964, -0.762, 0.0], [0.762, 0.964, 0.995]]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing each
    element of ``x`` mapped to the range ``(-1, 1)``.
"""

cos = functional(ops.cos, rule=unary_rule)
cos.__doc__ = """Computes the cosine of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 0.5, 1.0])
    result = F.cos(x)
    # result is approximately [1.0, 0.878, 0.540]

Args:
    x: The input interpreted as radians. Must have a floating-point
        dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    cosine of each element of ``x``.
"""

sin = functional(ops.sin, rule=unary_rule)
sin.__doc__ = """Computes the sine of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 0.5, 1.0])
    result = F.sin(x)
    # result is approximately [0.0, 0.479, 0.841]

Args:
    x: The input interpreted as radians. Must have a floating-point
        dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    sine of each element of ``x``.
"""

erf = functional(ops.erf, rule=unary_rule)
erf.__doc__ = """Computes the error function of a tensor element-wise.

The error function ``erf`` is the probability that a randomly sampled
normal distribution falls within a given range.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 1.0])
    result = F.erf(x)
    # result is approximately [-0.843, 0.0, 0.843]

Args:
    x: The input to the error function. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the error
    function applied to each element of ``x``.
"""

ceil = functional(ops.ceil, rule=unary_rule)
ceil.__doc__ = """Computes the ceiling of a tensor element-wise.

.. Skipped: Tensor defaults to bfloat16 on an accelerator, and Metal cannot
   compile a bf16 ``ceil``. Remove this skip once MOCO-4826 is fixed.
.. skip: next if(__import__("sys").platform == "darwin", "no bf16 ceil on Metal (MOCO-4826)")

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.5, 2.0, -1.5, -2.7])
    result = F.ceil(x)
    # result is [2.0, 2.0, -1.0, -2.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing each
    element of ``x`` rounded up toward positive infinity.
"""

floor = functional(ops.floor, rule=unary_rule)
floor.__doc__ = """Computes the floor of a tensor element-wise.

.. Skipped: Tensor defaults to bfloat16 on an accelerator, and Metal cannot
   compile a bf16 ``floor``. Remove this skip once MOCO-4826 is fixed.
.. skip: next if(__import__("sys").platform == "darwin", "no bf16 floor on Metal (MOCO-4826)")

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.5, 2.0, -1.5, -2.7])
    result = F.floor(x)
    # result is [1.0, 2.0, -2.0, -3.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing each
    element of ``x`` rounded down toward negative infinity.
"""

round = functional(ops.round, rule=unary_rule)
round.__doc__ = """Rounds a tensor to the nearest integer element-wise.

Values exactly halfway between two integers round to the nearest even integer
(for example, ``2.5`` rounds to ``2.0`` and ``3.5`` rounds to ``4.0``). All
other values follow normal rounding to the nearest integer.

.. Skipped: Tensor defaults to bfloat16 on an accelerator, and Metal cannot
   compile a bf16 ``round``. Remove this skip once MOCO-4826 is fixed.
.. skip: next if(__import__("sys").platform == "darwin", "no bf16 round on Metal (MOCO-4826)")

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.5, 1.5, 2.5, -0.5])
    result = F.round(x)
    # Ties round to the nearest even integer:
    # result is [0.0, 2.0, 2.0, 0.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing each
    element of ``x`` rounded to the nearest integer.
"""

trunc = functional(ops.trunc, rule=unary_rule)
trunc.__doc__ = """Truncates a tensor toward zero element-wise.

.. Skipped: Tensor defaults to bfloat16 on an accelerator, and Metal cannot
   compile a bf16 ``trunc``. Remove this skip once MOCO-4826 is fixed.
.. skip: next if(__import__("sys").platform == "darwin", "no bf16 trunc on Metal (MOCO-4826)")

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.5, 2.7, -1.5, -2.7])
    result = F.trunc(x)
    # result is [1.0, 2.0, -1.0, -2.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing each
    element of ``x`` truncated toward zero.
"""


is_inf = functional(ops.is_inf, rule=unary_rule)
is_inf.__doc__ = """Tests element-wise whether a tensor contains infinite values.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, float("inf"), float("-inf"), float("nan")])
    result = F.is_inf(x)
    # result is [False, True, True, False]

Args:
    x: The input tensor.

Returns:
    A ``Tensor`` with ``bool`` dtype and the same shape as ``x`` that is
    ``True`` where ``x`` is positive or negative infinity.
"""

is_nan = functional(ops.is_nan, rule=unary_rule)
is_nan.__doc__ = """Tests element-wise whether a tensor contains NaN values.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, float("inf"), float("nan"), 0.0])
    result = F.is_nan(x)
    # result is [False, False, True, False]

Args:
    x: The input tensor.

Returns:
    A ``Tensor`` with ``bool`` dtype and the same shape as ``x`` that is
    ``True`` where ``x`` is NaN.
"""

logical_not = functional(ops.logical_not, rule=unary_rule)
logical_not.__doc__ = """Computes the element-wise logical NOT of a boolean tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    x = Tensor([True, False, True], dtype=DType.bool)
    result = F.logical_not(x)
    # result is [False, True, False]

Args:
    x: The input boolean tensor.

Returns:
    A ``Tensor`` with ``bool`` dtype and the same shape as ``x`` containing
    the element-wise logical NOT of ``x``.
"""

log1p = functional(ops.log1p, rule=unary_rule)
log1p.__doc__ = """Computes ``log(1 + x)`` element-wise.

Note that ``log(1 + x)`` is undefined for ``x <= -1`` and complex
numbers are not currently supported.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 1e-7, 1.0])
    result = F.log1p(x)
    # result is approximately [0.0, 1e-7, 0.693]

Args:
    x: The input to the log computation. Must have a floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing
    ``log(1 + x)`` for each element of ``x``.
"""

atanh = functional(ops.atanh, rule=unary_rule)
atanh.__doc__ = """Computes the inverse hyperbolic tangent of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-0.5, 0.0, 0.5])
    result = F.atanh(x)
    # result is approximately [-0.549, 0.0, 0.549]

Args:
    x: The input tensor, with values in the range ``(-1, 1)``. Must have a
        floating-point dtype.

Returns:
    A ``Tensor`` of the same shape and dtype as ``x`` containing the
    inverse hyperbolic tangent of each element of ``x``.
"""

_acos_impl = functional(ops.acos, rule=unary_rule)


def acos(x: Tensor) -> Tensor:
    """Computes the arccosine of a tensor element-wise.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([-1.0, 0.0, 1.0])
        result = F.acos(x)
        # result is approximately [3.1416, 1.5708, 0.0] or [pi, pi/2, 0]

    Args:
        x: The input tensor with values in ``[-1, 1]``. Must have a
            floating-point dtype. For ``float16``, ``bfloat16``, and
            ``float32``, values outside this domain are clamped to the valid
            range. For ``float64``, out-of-domain values produce ``NaN``.

    Returns:
        A ``Tensor`` of the same shape and dtype as ``x`` containing the
        arccosine of each element of ``x``. Values range from ``[0, π]``
        (radians).
    """
    return _acos_impl(x)


#: Dequantizes a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.dequantize` for details.
_dequantize_impl = functional(ops.dequantize, rule=dequantize_rule)


def dequantize(encoding: QuantizationEncoding, quantized: Tensor) -> Tensor:
    """Dequantizes a quantized tensor to floating point.

    .. note::

        This currently supports the ``Q4_0``, ``Q4_K``, and ``Q6_K``
        encodings only.

    Args:
        encoding: The quantization encoding to use.
        quantized: The quantized tensor to dequantize.

    Returns:
        A ``Tensor`` containing the dequantized, floating point result.

    Raises:
        ValueError: If ``encoding`` is not a supported quantization encoding,
            or if the last dimension isn't divisible by the encoding's block
            size.
        TypeError: If the last dimension of ``quantized`` isn't static.
    """
    return _dequantize_impl(encoding, quantized)


equal = _binary_with_scalar_promotion(functional(ops.equal, rule=binary_rule))
equal.__doc__ = """Tests element-wise equality between two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 5.0, 3.0])
    result = F.equal(a, b)
    # result is [True, False, True]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise result
    of ``lhs == rhs``.
"""

not_equal = _binary_with_scalar_promotion(
    functional(ops.not_equal, rule=binary_rule)
)
not_equal.__doc__ = """Tests element-wise inequality between two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 5.0, 3.0])
    result = F.not_equal(a, b)
    # result is [False, True, False]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise result
    of ``lhs != rhs``.
"""

greater = _binary_with_scalar_promotion(
    functional(ops.greater, rule=binary_rule)
)
greater.__doc__ = """Tests element-wise whether one tensor is greater than another.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 5.0, 3.0])
    b = Tensor([2.0, 3.0, 3.0])
    result = F.greater(a, b)
    # result is [False, True, False]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise result
    of ``lhs > rhs``.
"""

greater_equal = _binary_with_scalar_promotion(
    functional(ops.greater_equal, rule=binary_rule)
)
greater_equal.__doc__ = """Tests element-wise whether one tensor is greater than or equal to another.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 5.0, 3.0])
    b = Tensor([2.0, 3.0, 3.0])
    result = F.greater_equal(a, b)
    # result is [False, True, True]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise result
    of ``lhs >= rhs``.
"""

logical_and = _binary_with_scalar_promotion(
    functional(ops.logical_and, rule=binary_rule)
)
logical_and.__doc__ = """Computes the element-wise logical AND of two boolean tensors.

Only supports boolean inputs.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    a = Tensor([True, True, False], dtype=DType.bool)
    b = Tensor([True, False, False], dtype=DType.bool)
    result = F.logical_and(a, b)
    # result is [True, False, False]

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise logical
    AND of ``lhs`` and ``rhs``.
"""

logical_or = _binary_with_scalar_promotion(
    functional(ops.logical_or, rule=binary_rule)
)
logical_or.__doc__ = """Computes the element-wise logical OR of two boolean tensors.

Only supports boolean inputs.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    a = Tensor([True, True, False], dtype=DType.bool)
    b = Tensor([True, False, False], dtype=DType.bool)
    result = F.logical_or(a, b)
    # result is [True, True, False]

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise logical
    OR of ``lhs`` and ``rhs``.
"""

logical_xor = _binary_with_scalar_promotion(
    functional(ops.logical_xor, rule=binary_rule)
)
logical_xor.__doc__ = """Computes the element-wise logical XOR of two boolean tensors.

Only supports boolean inputs.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    a = Tensor([True, True, False], dtype=DType.bool)
    b = Tensor([True, False, False], dtype=DType.bool)
    result = F.logical_xor(a, b)
    # result is [False, True, False]

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A ``Tensor`` with ``bool`` dtype containing the element-wise logical
    XOR of ``lhs`` and ``rhs``.
"""

#: SPMD-distributed wrapper around :func:`max.graph.ops.where`.
_where_inner = functional(ops.where, rule=ternary_rule)


def where(
    cond: Tensor,
    x: Tensor | float,
    y: Tensor | float,
) -> Tensor:
    """Selects elements from two tensors element-wise based on a condition.

    At each position, takes the element from ``x`` where ``cond`` is true and
    the element from ``y`` where it's false. Scalar ``x`` or ``y`` operands are
    promoted to tensors, and the inputs are broadcast to a common shape.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F
        from max.dtype import DType

        cond = Tensor([True, False, True], dtype=DType.bool)
        x = Tensor([1, 2, 3], dtype=DType.int32)
        y = Tensor([10, 20, 30], dtype=DType.int32)
        # Take x where True and y where False, producing [1, 20, 3].
        result = F.where(cond, x, y)
        # result is [1, 20, 3]

    Args:
        cond: The tensor selecting which input to take at each
            position. Must have a boolean dtype.
        x: The tensor to select from where ``cond`` is true.
        y: The tensor to select from where ``cond`` is false.

    Returns:
        A ``Tensor`` containing the element-wise selection from ``x`` and
        ``y`` according to ``cond``. It has the promoted dtype of ``x`` and
        ``y``, lives on their shared device, and has the broadcast shape of
        the inputs.

    Raises:
        ValueError: If ``cond`` doesn't have a boolean dtype, if the inputs
            aren't all on the same device, or if the dtypes of ``x`` and
            ``y`` can't be safely promoted.
        Error: If the input shapes aren't broadcast-compatible.
    """
    if isinstance(x, (int, float)) and isinstance(y, Tensor):
        x = full_like(y, x)
    elif isinstance(x, (int, float)):
        x = full_like(cond, x)
    if isinstance(y, (int, float)) and isinstance(x, Tensor):
        y = full_like(x, y)
    elif isinstance(y, (int, float)):
        y = full_like(cond, y)
    result = _where_inner(cond, x, y)
    assert isinstance(result, Tensor)
    return result


elementwise_min = _binary_with_scalar_promotion(
    functional(ops.elementwise.min, rule=binary_rule)
)
elementwise_min.__doc__ = """Computes the element-wise minimum of two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([3.0, 1.0, 4.0])
    b = Tensor([1.0, 5.0, 9.0])
    result = F.elementwise_min(a, b)
    # result is [1.0, 1.0, 4.0]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A tensor with the broadcast shape containing the smaller value at
    each position.
"""

elementwise_max = _binary_with_scalar_promotion(
    functional(ops.elementwise.max, rule=binary_rule)
)
elementwise_max.__doc__ = """Computes the element-wise maximum of two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([3.0, 1.0, 4.0])
    b = Tensor([1.0, 5.0, 9.0])
    result = F.elementwise_max(a, b)
    # result is [3.0, 5.0, 9.0]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A tensor with the broadcast shape containing the larger value at each
    position.
"""

#: Casts a tensor to a different data type. Distributed via SPMD.
#: See :func:`max.graph.ops.cast` for details.
cast = functional(ops.cast, rule=unary_rule)
cast.__doc__ = """Casts a tensor to a different data type.

Values may change when the source and target types can't represent each
other exactly. Float-to-integer casts truncate toward zero; float-to-float
casts with lower precision round to the nearest representable value.

.. code-block:: python

    from max.dtype import DType
    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.7, -1.7, 2.5])  # float32 on CPU by default
    result = F.cast(x, DType.int32)
    # result has dtype int32 and values [1, -1, 2]

Args:
    x: The input tensor.
    dtype: The target data type.

Returns:
    A tensor with the same shape but the new dtype.
"""


#: Performs matrix multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.matmul` for details.
matmul = functional(ops.matmul, rule=matmul_rule)
matmul.__doc__ = """Performs matrix multiplication between two tensors.

Treats the innermost two dimensions of each input as a matrix: ``lhs``
of shape ``(..., M, K)`` and ``rhs`` of shape ``(..., K, N)`` produce
an output of shape ``(..., M, N)``. The ``K`` dimensions must match.
Any outer batch dimensions are broadcast.

When inputs are distributed across devices, the operation is sharded
according to the matmul sharding rule.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    result = F.matmul(a, b)
    # result has shape (2, 2):
    # [[19.0, 22.0], [43.0, 50.0]]

    # The ``@`` operator on Tensor also calls matmul.
    result = a @ b

Args:
    lhs: The left-hand side input tensor.
    rhs: The right-hand side input tensor.

Returns:
    A tensor representing the matrix product of ``lhs`` and ``rhs``.
"""

#: Applies layer normalization. Distributed via SPMD.
#: See :func:`max.graph.ops.layer_norm` for details.
layer_norm = functional(ops.layer_norm, rule=layer_norm_rule)
#: Performs quantized matrix multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.qmatmul` for details.
qmatmul = functional(ops.qmatmul, rule=qmatmul_rule)

avg_pool2d = functional(ops.avg_pool2d, rule=linear_pool_rule)
avg_pool2d.__doc__ = """Applies 2D average pooling.

Slides a window of size ``kernel_size`` over the spatial dimensions and
replaces each window with its average value.

Args:
    input: The input tensor in channels-last (NHWC) layout,
        ``(batch_size, height, width, channels)``.
    kernel_size: A tuple ``(kernel_h, kernel_w)`` giving the height and
        width of the sliding window.
    stride: The stride of the sliding window. Either a single ``int``
        applied to both spatial dimensions, or a tuple
        ``(stride_h, stride_w)``. Defaults to ``1``.
    dilation: The spacing between kernel elements. Either a single
        ``int`` applied to both spatial dimensions, or a tuple
        ``(dilation_h, dilation_w)``. Defaults to ``1``.
    padding: Zero-padding added to both sides of each spatial dimension.
        Either a single ``int`` applied to both spatial dimensions, or a
        tuple ``(pad_h, pad_w)``. Defaults to ``0``.
    ceil_mode: When ``True``, uses ceil instead of floor when computing
        the output spatial shape. Defaults to ``False``.
    count_boundary: When ``True``, includes padding elements in the
        divisor when computing the average. Defaults to ``True``.

Returns:
    A ``Tensor`` containing the averaged values, with shape
    ``(batch_size, height_out, width_out, channels)``.
"""

max_pool2d = functional(ops.max_pool2d, rule=pool_rule)
max_pool2d.__doc__ = """Applies 2D max pooling to a tensor.

Slides a window of size ``kernel_size`` over the spatial dimensions and
replaces each window with its maximum value.

Args:
    input: The input tensor in channels-last (NHWC) layout,
        ``(batch_size, height, width, channels)``.
    kernel_size: A tuple ``(kernel_h, kernel_w)`` giving the height and
        width of the sliding window.
    stride: The stride of the sliding window. Either a single ``int``
        applied to both spatial dimensions, or a tuple
        ``(stride_h, stride_w)``. Defaults to ``1``.
    dilation: The spacing between kernel elements. Either a single
        ``int`` applied to both spatial dimensions, or a tuple
        ``(dilation_h, dilation_w)``. Defaults to ``1``.
    padding: Padding added to both sides of each spatial dimension.
        Out-of-bounds positions are excluded from the maximum (equivalently,
        they use the dtype's minimum value or negative infinity), so padding
        cannot win over negative input values. Either a single ``int`` applied
        to both spatial dimensions, or a tuple ``(pad_h, pad_w)``. Defaults
        to ``0``.
    ceil_mode: When ``True``, uses ceil instead of floor when computing
        the output spatial shape. Defaults to ``False``.

Returns:
    A ``Tensor`` containing the max-pooled values, with shape
    ``(batch_size, height_out, width_out, channels)``.
"""

#: Permutes the dimensions of a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.permute` for details.
permute = functional(ops.permute, rule=permute_rule)
permute.__doc__ = """Permutes all dimensions of a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # x has shape (1, 2, 3).
    x = Tensor.ones([1, 2, 3])
    # Reorder the dimensions to (2, 0, 1), producing shape (3, 1, 2).
    result = F.permute(x, [2, 0, 1])

Args:
    x: The input tensor to permute.
    dims: The target order of the dimensions as a list of axis indices.
        Each axis may be negative to index from the end of the tensor.

Returns:
    A ``Tensor`` containing ``x`` with its dimensions reordered to match
    ``dims``. It has the same elements and dtype as ``x``, with the order of
    the elements changed according to the permutation.

Raises:
    ValueError: If the length of ``dims`` does not match the rank of the
        input, or if ``dims`` contains duplicate dimensions.
    IndexError: If any dimension in ``dims`` is out of range.
"""

transpose = functional(ops.transpose, rule=transpose_rule)
transpose.__doc__ = """Transposes two axes of a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # x has shape (2, 3).
    x = Tensor.ones([2, 3])
    # Swap axes 0 and 1, producing shape (3, 2).
    result = F.transpose(x, 0, 1)

Args:
    x: The input tensor to transpose.
    axis_1: One of the two axes to transpose. If negative, this indexes from
        the end of the tensor. For example, a value of ``-1`` refers to the
        last axis.
    axis_2: The other axis to transpose. If negative, this indexes from the
        end of the tensor.

Returns:
    A ``Tensor`` containing the input with ``axis_1`` and ``axis_2``
    transposed. It has the same elements and dtype as ``x``, with the order
    of the elements changed according to the transposition. For a rank-zero
    tensor, axes ``-1`` and ``0`` are accepted and the scalar is returned
    unchanged.

Raises:
    IndexError: If ``axis_1`` or ``axis_2`` is out of range.
"""

unsqueeze = functional(ops.unsqueeze, rule=unsqueeze_rule)
unsqueeze.__doc__ = """Inserts a dimension of size ``1`` into a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # x has shape (3,).
    x = Tensor.ones([3])
    # Insert a size-1 dimension at axis 0, producing shape (1, 3).
    result = F.unsqueeze(x, 0)

Args:
    x: The input tensor to unsqueeze.
    axis: The index at which to insert a new dimension into the input's
        shape. Elements at that index or higher are shifted back. If
        negative, it indexes relative to ``1`` plus the rank of the tensor.
        For example, a value of ``-1`` adds a new dimension at the end, and
        ``-2`` inserts the dimension immediately before the last dimension.

Returns:
    A ``Tensor`` containing ``x`` with a new dimension inserted at ``axis``.
    That dimension has a size of ``1``, so the result holds the same elements
    as ``x`` with one more dimension.

Raises:
    ValueError: If ``axis`` is out of bounds.
"""

squeeze = functional(ops.squeeze, rule=squeeze_rule)
#: SPMD-distributed wrapper around :func:`max.graph.ops.reshape`.
reshape = functional(ops.reshape, rule=reshape_rule)
reshape.__doc__ = """Reshapes a tensor.

If a value of ``-1`` is present in ``shape``, that dimension becomes an
automatically calculated dimension collecting all unspecified dimensions.
Its length becomes the number of elements in the original tensor divided by
the product of the other dimensions of ``shape``.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # x has shape (2, 3).
    x = Tensor.ones([2, 3])
    # Reshape the same 6 elements into shape (3, 2).
    result = F.reshape(x, [3, 2])

Args:
    x: The input tensor to reshape.
    shape: The new shape as an iterable of dimensions (a list, tuple, or
        ``Dim`` values). A single dimension may be ``-1``.

Returns:
    A ``Tensor`` containing ``x`` with a new ``shape``. The order and total
    number of elements stays the same as the input.

Raises:
    ValueError: If ``shape`` contains more than one ``-1`` dimension, if a
        ``-1`` dimension is requested while another dimension is ``0``, or if
        the input and target shapes have a different number of elements.
"""
#: Flattens a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.flatten` for details.
flatten = functional(ops.flatten, rule=flatten_rule)
flatten.__doc__ = """Flattens the specified dimensions of a tensor.

This does not change the order or total number of elements in the tensor.
All dimensions from ``start_dim`` to ``end_dim`` (inclusive) are merged into
a single output dimension.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # x has shape (2, 2, 2).
    x = Tensor.ones([2, 2, 2])
    # Merge dimensions 1 and 2 into one, producing shape (2, 4).
    result = F.flatten(x, start_dim=1)

Args:
    x: The input tensor to flatten.
    start_dim: The first dimension to flatten. Supports negative indexing.
        Defaults to ``0``.
    end_dim: The last dimension to flatten (inclusive). Supports negative
        indexing. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the ``start_dim`` through ``end_dim`` of ``x``
    merged into one dimension.

Raises:
    IndexError: If ``start_dim`` or ``end_dim`` is out of range.
    ValueError: If ``start_dim`` comes after ``end_dim``.
"""

tile = functional(ops.tile, rule=tile_rule)
tile.__doc__ = """Repeats a tensor along each of its dimensions.

Each dimension ``i`` is copied ``repeats[i]`` times, so its output size is
``x.shape[i] * repeats[i]``.

This op runs on CPU. An input on another device is copied to CPU for the
operation and the result is copied back.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1, 2], [3, 4]])

    # Repeat the columns twice, leaving the rows unchanged.
    result = F.tile(x, [1, 2])
    # [[1, 2, 1, 2], [3, 4, 3, 4]]

Args:
    x: The tensor to tile.
    repeats: The number of copies for each dimension, one positive value
        per dimension of ``x``.

Returns:
    A ``Tensor`` containing the tiled input.

Raises:
    ValueError: If ``repeats`` doesn't have one value per dimension, if any
        statically known value isn't positive, or if ``x`` is on a non-CPU
        device and ``strict_device_placement=DevicePlacementPolicy.Error``.
"""

pad = functional(ops.pad, rule=pad_rule)
#: SPMD-distributed wrapper around :func:`max.graph.ops.broadcast_to`.
_broadcast_to_impl = functional(ops.broadcast_to, rule=broadcast_to_rule)


def broadcast_to(x: Tensor, shape: ShapeLike) -> Tensor:
    """Broadcasts a tensor to a target shape.

    Each input dimension must either equal the corresponding target
    dimension or be ``1`` (which is then stretched to match). This
    follows NumPy broadcasting semantics and is equivalent to PyTorch's
    :func:`torch.broadcast_to`.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor.ones([3, 1])
        result = F.broadcast_to(x, [3, 4])
        # result has shape (3, 4)

        # Add a new leading dimension
        result = F.broadcast_to(x, [2, 3, 4])
        # result has shape (2, 3, 4)

    Args:
        x: The input tensor. Must not contain any dynamic dimensions.
        shape: The target shape. A static shape (no dynamic dimensions).

    Returns:
        A ``Tensor`` with the same elements as ``x`` but with the target
        shape.
    """
    return _broadcast_to_impl(x, shape)


#: Repeats elements of a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.repeat_interleave` for details.
repeat_interleave = functional(
    ops.repeat_interleave, rule=repeat_interleave_rule
)
repeat_interleave.__doc__ = """Repeats each element of a tensor along an axis.

Unlike :func:`tile`, which repeats whole blocks, this repeats each
element ``repeats`` times consecutively.

This op runs on CPU only; a GPU input raises an error.

.. note::

    The functional API currently supports only integer ``repeats``. Use
    :func:`max.graph.ops.repeat_interleave` for per-element tensor repeats.

The examples below use an input containing ``[[1.0, 2.0], [3.0, 4.0]]``:

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.driver import CPU

    input = Tensor([[1.0, 2.0], [3.0, 4.0]], device=CPU())

    # Repeat each row twice.
    output = F.repeat_interleave(input, repeats=2, axis=0)
    # [[1, 2], [1, 2], [3, 4], [3, 4]], shape (4, 2)

    # Repeat each column twice.
    output = F.repeat_interleave(input, repeats=2, axis=1)
    # [[1, 1, 2, 2], [3, 3, 4, 4]], shape (2, 4)

    # With no axis, flatten the input first, then repeat each element.
    output = F.repeat_interleave(input, repeats=2)
    # [1, 1, 2, 2, 3, 3, 4, 4], shape (8,)

Args:
    x: The input tensor.
    repeats: The integer number of times to repeat each element.
    axis: The axis to repeat along. If ``None`` (the default), the input
        is flattened first.
    out_dim: The output size along ``axis``. This is inferred when
        ``repeats`` is an integer.

Returns:
    A ``Tensor`` containing the input with its elements interleaved.

Raises:
    ValueError: If ``repeats`` is non-positive, if ``axis`` is out of
        range, or if the input is on a GPU device.
"""

slice_tensor = functional(ops.slice_tensor, rule=slice_tensor_rule)
slice_tensor.__doc__ = """Slices out a subtensor of the input tensor based on ``indices``.

The semantics of :func:`slice_tensor()` follow basic NumPy slicing
semantics, with one index per dimension. Each index is one of:

- An integer.
- A scalar tensor (a dynamic integer index).
- A ``slice``.
- A ``(slice, out_dim)`` tuple, which names the output dimension when
  slicing a dynamic dimension.
- ``None`` (to insert a size-1 dimension).
- ``Ellipsis`` (to fill in full slices for the remaining dimensions).

Slice indices must stay within ``[-dim, dim]``, and slice steps must be
positive.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Take rows 0 and 1 and columns 1 and 2, producing [[2, 3], [5, 6]].
    result = F.slice_tensor(x, [slice(0, 2), slice(1, 3)])
    # result is [[2, 3], [5, 6]]

Args:
    x: The input tensor to slice.
    indices: The per-dimension index expressions. Each entry is an integer,
        a scalar tensor, a ``slice``, a ``(slice, out_dim)`` tuple,
        ``None``, or ``Ellipsis``.

Returns:
    A ``Tensor`` containing the sliced subtensor of ``x``.

Raises:
    IndexError: If a slice bound or integer index is out of range for its
        dimension.
    ValueError: If ``x`` is a scalar, if more indices than dimensions are
        given, if more than one ``Ellipsis`` appears, or if a slice step
        is ``0``.
    NotImplementedError: If a plain ``slice`` targets a dynamic dimension.
        Pass a ``(slice, out_dim)`` tuple instead.
"""

concat = functional(ops.concat, rule=same_placement_multi_input_rule)
concat.__doc__ = """Concatenates tensors along an axis.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    vertical = F.concat([a, b], axis=0)
    # vertical has shape (4, 2):
    # [[1, 2], [3, 4], [5, 6], [7, 8]]

    horizontal = F.concat([a, b], axis=1)
    # horizontal has shape (2, 4):
    # [[1, 2, 5, 6], [3, 4, 7, 8]]

Args:
    original_vals: The tensors to concatenate. They must have the same
        rank and size on every dimension except ``axis``.
    axis: The axis to concatenate along. Negative values count from the
        end. Defaults to ``0``.

Returns:
    A ``Tensor`` containing the concatenated inputs. Its size along
    ``axis`` is the sum of the inputs' sizes and every other axis is
    unchanged.

Raises:
    ValueError: If no tensors are provided, if the inputs don't all have
        the same rank, if they differ in size on any dimension other than
        ``axis``, or if they aren't all on the same device.
    IndexError: If ``axis`` is out of range.
"""

stack = functional(ops.stack, rule=stack_rule)
stack.__doc__ = """Stacks tensors along a new axis.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    # Stack the two (2, 2) tensors into one (2, 2, 2) tensor
    result = F.stack([a, b], axis=0)
    # result has shape (2, 2, 2)

Args:
    values: The tensors to stack. Each must have the same dtype, rank,
        shape, and device.
    axis: The position of the new axis. Negative values count from the
        end, where ``-1`` inserts the new axis as the last dimension.
        Defaults to ``0``.

Returns:
    A ``Tensor`` containing the stacked inputs. It has one more dimension
    than the inputs, and the new dimension has size ``len(values)``.

Raises:
    ValueError: If ``values`` is empty, or if the tensors don't all have
        the same dtype, rank, shape, and device.
    IndexError: If ``axis`` is out of range.
"""

argsort = functional(ops.argsort, rule=argsort_rule)
argsort.__doc__ = """Returns the indices that would sort a rank-1 tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([3.0, 1.0, 2.0])
    # Ascending order visits 1, 2, 3, so the indices are [1, 2, 0].
    result = F.argsort(x, ascending=True)
    # result is [1, 2, 0]

Args:
    x: The input tensor to sort. Must be rank 1.
    ascending: Whether to sort in ascending order. If ``False``, sorts in
        descending order. Defaults to ``True``.

Returns:
    A ``Tensor`` containing the sorting indices, with the same shape as
    ``x`` and ``int64`` dtype.

Raises:
    ValueError: If ``x`` is not rank 1.
"""

nonzero = functional(ops.nonzero, rule=nonzero_rule)
nonzero.__doc__ = """Returns the indices of all nonzero elements of a tensor.

Each row is the multi-index of one nonzero element, and the rows are
generated in row-major order.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[0, 1], [2, 0]])
    # Nonzero elements at (0, 1) and (1, 0) produce [[0, 1], [1, 0]].
    result = F.nonzero(x, out_dim="nonzero")
    # result is [[0, 1], [1, 0]]

Args:
    x: The input tensor.
    out_dim: The new data-dependent dimension for the number of nonzero
        elements.

Returns:
    A ``Tensor`` containing the indices of the nonzero elements of ``x``,
    with shape ``[out_dim, x.rank]`` and ``int64`` dtype.

Raises:
    ValueError: If ``x`` is scalar, or if ``x`` is on a non-CPU device and
        ``strict_device_placement=DevicePlacementPolicy.Error``.
"""

gather = functional(ops.gather, rule=gather_rule)
gather.__doc__ = """Selects elements out of an input tensor by index.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    x = Tensor([[1, 2], [3, 4], [5, 6]], dtype=DType.int32)
    indices = Tensor([0, 2], dtype=DType.int64)
    # Select rows 0 and 2, producing [[1, 2], [5, 6]].
    result = F.gather(x, indices, axis=0)
    # result is [[1, 2], [5, 6]]

.. note::

    When the gather axis is :class:`~max.experimental.sharding.Sharded`, the
    dispatcher first calls :func:`allgather` to make the input
    :class:`~max.experimental.sharding.Replicated`. It doesn't emit an
    expert-parallel ``(Sharded(a_axis), R) → Partial(SUM)`` row, because that's
    only correct when the caller masks indices per rank. Models that want
    expert-parallel semantics override ``gather.rule`` with their own rule.

Args:
    input: The input tensor to select elements from.
    indices: A tensor of ``int32`` or ``int64`` index values on the same
        device as ``input``.
    axis: The dimension that ``indices`` indexes into ``input``. If
        negative, indexes relative to the end of the input tensor. For
        example, ``gather(input, indices, axis=-1)`` indexes against the
        last dimension of ``input``.

Returns:
    A ``Tensor`` containing the selected elements. Its shape is
    ``input.shape`` with the dimension at ``axis`` replaced by
    ``indices.shape``.

Raises:
    IndexError: If ``axis`` is out of range for ``input``.
    ValueError: If ``indices`` isn't integral or isn't on the same device
        as ``input``.
"""

scatter = functional(ops.scatter, rule=scatter_rule)
scatter.__doc__ = """Writes ``updates`` into a copy of ``input`` at positions given by ``indices``.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.driver import CPU
    from max.dtype import DType

    x = Tensor([1, 2, 3, 4, 5], dtype=DType.int32, device=CPU())
    updates = Tensor([10, 20], dtype=DType.int32, device=CPU())
    indices = Tensor([0, 3], dtype=DType.int64, device=CPU())
    # Overwrite positions 0 and 3, producing [10, 2, 3, 20, 5].
    result = F.scatter(x, updates, indices, axis=0)
    # result is [10, 2, 3, 20, 5]

.. note::

    When the scatter axis is :class:`~max.experimental.sharding.Sharded`, the
    dispatcher first calls :func:`allgather` to make the input
    :class:`~max.experimental.sharding.Replicated`. It doesn't emit a
    per-rank-local ``(Sharded(a_axis), R, R) → Sharded(a_axis)`` row, because
    that's only correct when the caller masks indices and updates per rank.
    Models that want expert-parallel semantics override ``scatter.rule`` with
    their own rule.

Args:
    input: The input tensor to write elements to.
    updates: A tensor of elements to write to ``input``.
    indices: The positions in ``input`` to update.
    axis: The axis along which ``indices`` indexes. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing ``input`` with ``updates`` written at
    ``indices``. It has the same shape and dtype as ``input``.

Raises:
    ValueError: If ``axis`` is out of range, if the input and updates
        dtypes mismatch, if ``indices`` dtype is not int32/int64, if the
        inputs aren't all on the same device, or if any input is on a
        non-CPU device and
        ``strict_device_placement=DevicePlacementPolicy.Error``.
    Error: If ``input``, ``updates``, and ``indices`` don't share the same
        rank, if ``updates`` and ``indices`` don't have the same shape, or
        if any ``indices`` dimension exceeds the matching ``input``
        dimension.
"""

scatter_add = functional(ops.scatter_add, rule=scatter_add_rule)
scatter_add.__doc__ = """Creates a new tensor by accumulating ``updates`` into ``input`` at ``indices``.

Produces an output tensor by scattering elements from ``updates`` into
``input`` according to ``indices``, summing values at duplicate indices. For
a 2-D input with ``axis=0`` the update rule is:

.. code-block:: text

    output[indices[i][j]][j] += updates[i][j]

and with ``axis=1``:

.. code-block:: text

    output[i][indices[i][j]] += updates[i][j]

Args:
    input: The input tensor to accumulate into.
    updates: A tensor of values to add.
    indices: The positions in ``input`` to update.
    axis: The axis along which ``indices`` indexes into. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``axis`` is out of range, if the input and updates
        dtypes mismatch, if ``indices`` dtype is not int32/int64, if the
        inputs aren't all on the same device, or if any input is on a
        non-CPU device and
        ``strict_device_placement=DevicePlacementPolicy.Error``.
    Error: If ``input``, ``updates``, and ``indices`` don't share the same
        rank, if ``updates`` and ``indices`` don't have the same shape, or
        if any ``indices`` dimension exceeds the matching ``input``
        dimension.
"""

scatter_max = functional(ops.scatter_max, rule=scatter_add_rule)
scatter_max.__doc__ = """Creates a new tensor by scattering the maximum of ``updates`` into ``input``.

Produces an output tensor by scattering elements from ``updates`` into
``input`` according to ``indices``, keeping the maximum at duplicate indices.
For a 2-D input with ``axis=0`` the update rule is:

.. code-block:: text

    output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j])

and with ``axis=1``:

.. code-block:: text

    output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j])

Args:
    input: The input tensor to scatter into.
    updates: A tensor of values to compare.
    indices: The positions in ``input`` to update.
    axis: The axis along which ``indices`` indexes into. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``axis`` is out of range, if the input and updates
        dtypes mismatch, if ``indices`` dtype is not int32/int64, if the
        inputs aren't all on the same device, or if any input is on a
        non-CPU device and
        ``strict_device_placement=DevicePlacementPolicy.Error``.
    Error: If ``input``, ``updates``, and ``indices`` don't share the same
        rank, if ``updates`` and ``indices`` don't have the same shape, or
        if any ``indices`` dimension exceeds the matching ``input``
        dimension.
"""

scatter_min = functional(ops.scatter_min, rule=scatter_add_rule)
scatter_min.__doc__ = """Creates a new tensor by scattering the minimum of ``updates`` into ``input``.

Produces an output tensor by scattering elements from ``updates`` into
``input`` according to ``indices``, keeping the minimum at duplicate indices.
For a 2-D input with ``axis=0`` the update rule is:

.. code-block:: text

    output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j])

and with ``axis=1``:

.. code-block:: text

    output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j])

Args:
    input: The input tensor to scatter into.
    updates: A tensor of values to compare.
    indices: The positions in ``input`` to update.
    axis: The axis along which ``indices`` indexes into. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``axis`` is out of range, if the input and updates
        dtypes mismatch, if ``indices`` dtype is not int32/int64, if the
        inputs aren't all on the same device, or if any input is on a
        non-CPU device and
        ``strict_device_placement=DevicePlacementPolicy.Error``.
    Error: If ``input``, ``updates``, and ``indices`` don't share the same
        rank, if ``updates`` and ``indices`` don't have the same shape, or
        if any ``indices`` dimension exceeds the matching ``input``
        dimension.
"""

scatter_mul = functional(ops.scatter_mul, rule=scatter_add_rule)
scatter_mul.__doc__ = """Creates a new tensor by scattering the product of ``updates`` into ``input``.

Produces an output tensor by scattering elements from ``updates`` into
``input`` according to ``indices``, multiplying values at duplicate indices.
For a 2-D input with ``axis=0`` the update rule is:

.. code-block:: text

    output[indices[i][j]][j] *= updates[i][j]

and with ``axis=1``:

.. code-block:: text

    output[i][indices[i][j]] *= updates[i][j]

Args:
    input: The input tensor to scatter into.
    updates: A tensor of values to multiply.
    indices: The positions in ``input`` to update.
    axis: The axis along which ``indices`` indexes into. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``axis`` is out of range, if the input and updates
        dtypes mismatch, if ``indices`` dtype is not int32/int64, if the
        inputs aren't all on the same device, or if any input is on a
        non-CPU device and
        ``strict_device_placement=DevicePlacementPolicy.Error``.
    Error: If ``input``, ``updates``, and ``indices`` don't share the same
        rank, if ``updates`` and ``indices`` don't have the same shape, or
        if any ``indices`` dimension exceeds the matching ``input``
        dimension.
"""

scatter_nd = functional(ops.scatter_nd, rule=scatter_nd_rule)
scatter_nd.__doc__ = """Scatters slices from ``updates`` into a copy of ``input`` at N-dimensional indices.

The last dimension of ``indices`` is the index vector. Its values select a
slice (or scalar) in ``input``. When the index vector length ``k`` is less
than ``input.rank``, each update writes a whole slice of the trailing
``input.rank - k`` dimensions.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.driver import CPU
    from max.dtype import DType

    x = Tensor(
        [[1, 2], [3, 4], [5, 6]], dtype=DType.int32, device=CPU()
    )
    updates = Tensor(
        [[10, 20], [50, 60]], dtype=DType.int32, device=CPU()
    )
    indices = Tensor([[0], [2]], dtype=DType.int64, device=CPU())
    # Overwrite rows 0 and 2, producing [[10, 20], [3, 4], [50, 60]].
    result = F.scatter_nd(x, updates, indices)
    # result is [[10, 20], [3, 4], [50, 60]]

Args:
    input: The input tensor to write elements to.
    updates: A tensor of elements to write to ``input``, with shape
        ``indices.shape[:-1] + input.shape[k:]``.
    indices: An ``int32`` or ``int64`` tensor specifying where to write
        ``updates``. Its last dimension ``k`` is the index vector length
        (``k <= input.rank``) and its leading dimensions may take any
        shape. Full indexing uses ``k = input.rank`` and partial indexing
        uses ``k < input.rank``.

Returns:
    A ``Tensor`` containing ``input`` with ``updates`` scattered in. It has
    the same shape and dtype as ``input``.

Raises:
    ValueError: If dtypes, devices, ranks, or shapes are incompatible, or
        if ``indices`` isn't an integral tensor.
"""

scatter_nd_add = functional(ops.scatter_nd_add, rule=scatter_nd_add_rule)
scatter_nd_add.__doc__ = """Creates a new tensor by accumulating ``updates`` into ``input`` at N-D indices.

Produces an output tensor by scattering slices from ``updates`` into a copy
of ``input`` according to N-dimensional index vectors, summing values at
duplicate index positions. Each index vector is the last dimension of
``indices`` and selects a slice (or scalar) in ``input``.

Example for ``input.shape = [4, 2]``, ``indices.shape = [3, 1]``
(1-D partial indexing, writes whole rows):

.. code-block:: text

    output[indices[i, 0], :] += updates[i, :]

Args:
    input: The input tensor to accumulate into.
    updates: A tensor of values to add.
    indices: An index tensor whose last dimension is the index vector length
        ``k`` (``k <= input.rank``).

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``input`` and ``updates`` dtypes mismatch, if
        ``indices`` dtype isn't int32 or int64, or if the inputs aren't all
        on the same device.
"""

scatter_nd_max = functional(ops.scatter_nd_max, rule=scatter_nd_add_rule)
scatter_nd_max.__doc__ = """Creates a new tensor by scattering the maximum of ``updates`` into ``input`` at N-D indices.

Produces an output tensor by scattering slices from ``updates`` into a copy
of ``input`` according to N-dimensional index vectors, keeping the maximum at
duplicate index positions. Each index vector is the last dimension of
``indices`` and selects a slice (or scalar) in ``input``.

Example for ``input.shape = [4, 2]``, ``indices.shape = [3, 1]``
(1-D partial indexing, writes whole rows):

.. code-block:: text

    output[indices[i, 0], :] = max(output[indices[i, 0], :], updates[i, :])

Args:
    input: The input tensor to scatter into.
    updates: A tensor of values to compare.
    indices: An index tensor whose last dimension is the index vector length
        ``k`` (``k <= input.rank``).

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``input`` and ``updates`` dtypes mismatch, if
        ``indices`` dtype isn't int32 or int64, or if the inputs aren't all
        on the same device.
"""

scatter_nd_min = functional(ops.scatter_nd_min, rule=scatter_nd_add_rule)
scatter_nd_min.__doc__ = """Creates a new tensor by scattering the minimum of ``updates`` into ``input`` at N-D indices.

Produces an output tensor by scattering slices from ``updates`` into a copy
of ``input`` according to N-dimensional index vectors, keeping the minimum at
duplicate index positions. Each index vector is the last dimension of
``indices`` and selects a slice (or scalar) in ``input``.

Example for ``input.shape = [4, 2]``, ``indices.shape = [3, 1]``
(1-D partial indexing, writes whole rows):

.. code-block:: text

    output[indices[i, 0], :] = min(output[indices[i, 0], :], updates[i, :])

Args:
    input: The input tensor to scatter into.
    updates: A tensor of values to compare.
    indices: An index tensor whose last dimension is the index vector length
        ``k`` (``k <= input.rank``).

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``input`` and ``updates`` dtypes mismatch, if
        ``indices`` dtype isn't int32 or int64, or if the inputs aren't all
        on the same device.
"""

scatter_nd_mul = functional(ops.scatter_nd_mul, rule=scatter_nd_add_rule)
scatter_nd_mul.__doc__ = """Creates a new tensor by scattering the product of ``updates`` into ``input`` at N-D indices.

Produces an output tensor by scattering slices from ``updates`` into a copy
of ``input`` according to N-dimensional index vectors, multiplying values at
duplicate index positions. Each index vector is the last dimension of
``indices`` and selects a slice (or scalar) in ``input``.

Example for ``input.shape = [4, 2]``, ``indices.shape = [3, 1]``
(1-D partial indexing, writes whole rows):

.. code-block:: text

    output[indices[i, 0], :] *= updates[i, :]

Args:
    input: The input tensor to scatter into.
    updates: A tensor of values to multiply.
    indices: An index tensor whose last dimension is the index vector length
        ``k`` (``k <= input.rank``).

Returns:
    A ``Tensor`` containing the updated tensor. It has the same shape and
    dtype as ``input``.

Raises:
    ValueError: If ``input`` and ``updates`` dtypes mismatch, if
        ``indices`` dtype isn't int32 or int64, or if the inputs aren't all
        on the same device.
"""

gather_nd = functional(ops.gather_nd, rule=gather_nd_rule)
gather_nd.__doc__ = """Selects elements from a tensor by N-dimensional index.

Unlike :func:`gather()`, which indexes along a single axis,
``gather_nd()`` indexes along multiple dimensions at once. The last
dimension of ``indices`` is the index vector: its values select
elements from ``input`` immediately after any ``batch_dims`` leading
dimensions. Any remaining trailing dimensions of ``input`` are sliced
into the output as features.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    input = Tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    indices = Tensor([[0, 1], [1, 0]], dtype=DType.int64)
    gathered = F.gather_nd(input, indices)
    # gathered is [[3.0, 4.0], [5.0, 6.0]]

Each row of ``indices`` selects one row from the first two dimensions of
``input``. The trailing dimension is copied into the output.

Args:
    input: The input tensor to gather from.
    indices: An integer tensor of multi-dimensional indices. Its last
        dimension must be static and gives the size of the index
        vector.
    batch_dims: The number of leading batch dimensions shared by
        ``input`` and ``indices``. The shapes must match exactly along
        these leading dimensions. This function does not broadcast.
        Defaults to ``0``.

Returns:
    A ``Tensor`` containing the gathered elements, with the same dtype as
    ``input``. Its shape is the concatenation of:

    - ``input.shape[:batch_dims]`` — the leading batch dimensions.
    - ``indices.shape[batch_dims:-1]`` — the gather dimensions.
    - ``input.shape[batch_dims + indices.shape[-1]:]`` — the trailing
      sliced dimensions.

Raises:
    ValueError: If any input is invalid. This includes when ``indices``'s
        last dimension is not static, ``indices`` is not an integer tensor,
        ``batch_dims`` is negative or greater than ``indices.rank - 1``,
        ``batch_dims + indices.shape[-1]`` exceeds ``input.rank``, or the
        leading ``batch_dims`` of ``input`` and ``indices`` don't match.
"""

masked_scatter = functional(ops.masked_scatter, rule=masked_scatter_rule)
masked_scatter.__doc__ = """Updates tensor values at positions where ``mask`` is true.

Positions are filled in row-major order, so the first ``True`` position in
``mask`` takes the first element of ``updates``, and so on.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    x = Tensor([[1, 2], [3, 4]], dtype=DType.int32)
    mask = Tensor([[True, False], [False, True]], dtype=DType.bool)
    updates = Tensor([10, 20], dtype=DType.int32)
    # Write into the True positions, producing [[10, 2], [3, 20]].
    result = F.masked_scatter(x, mask, updates, out_dim="num_updates")
    # result is [[10, 2], [3, 20]]

Args:
    input: The input tensor to write elements to.
    mask: A tensor selecting the positions to write, broadcast to the shape
        of ``input``. Pass a boolean tensor. A weak Python value is
        converted to boolean, but an existing tensor is used unchanged.
    updates: A tensor of elements to write to ``input``.
    out_dim: The new data-dependent dimension for the number of ``True``
        positions in ``mask``.

Returns:
    A ``Tensor`` containing ``input`` with ``updates`` written where
    ``mask`` is true. It has the same shape and dtype as ``input``.

Raises:
    ValueError: If ``input`` and ``updates`` have mismatched dtypes, or if
        the inputs aren't all on the same device.
"""

outer = functional(ops.outer, rule=outer_rule)
outer.__doc__ = """Computes the outer product of two vectors.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    lhs = Tensor([1.0, 2.0, 3.0])
    rhs = Tensor([4.0, 5.0])
    # Outer product, producing [[4, 5], [8, 10], [12, 15]].
    result = F.outer(lhs, rhs)
    # result has shape (3, 2)

Args:
    lhs: The left side of the product. Must be rank 1.
    rhs: The right side of the product. Must be rank 1.

Returns:
    A ``Tensor`` containing the
    `outer product <https://en.wikipedia.org/wiki/Outer_product>`_ of the
    two input vectors. It has rank 2, with dimension sizes equal to the
    number of elements of ``lhs`` and ``rhs`` respectively.

Raises:
    ValueError: If ``lhs`` or ``rhs`` is not rank 1.
"""

_split_impl = functional(ops.split, rule=split_rule)


def split(
    x: Tensor,
    split_size_or_sections: int | Sequence[DimLike],
    axis: int = 0,
) -> list[Tensor]:
    """Splits a tensor into chunks along an axis.

    An ``int`` ``split_size_or_sections`` produces equal chunks (the
    last may be smaller); a sequence specifies per-chunk sizes.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        first, second = F.split(x, [2, 4], axis=0)
        # first is [1.0, 2.0]
        # second is [3.0, 4.0, 5.0, 6.0]

    Args:
        x: The tensor to split.
        split_size_or_sections: Either a positive chunk size or a sequence
            giving the exact size of each output section.
        axis: The axis to split. Negative values count from the end.
            Defaults to ``0``.

    Returns:
        A list of tensors in their original order along ``axis``.

    Raises:
        TypeError: If an integer chunk size is used for a non-static axis.
        ValueError: If a section size is negative or explicit section sizes
            don't sum to the input size.
        IndexError: If ``axis`` is out of range.
    """
    if isinstance(split_size_or_sections, int):
        # On a sharded axis ``x.shape[axis]`` is a PerShardDim carrying the
        # global size; ``global_dim`` recovers that static global (and is a
        # no-op on a plain dim).
        dim = global_dim(Dim(x.shape[axis]))
        if not isinstance(dim, StaticDim):
            raise TypeError(
                f"split(x, chunk_size={split_size_or_sections}, axis={axis}): "
                f"non-static dim {x.shape[axis]!r}; pass an explicit "
                "split_sizes list."
            )
        dim_size = dim.dim
        chunk_size = split_size_or_sections
        num_full, remainder = divmod(dim_size, chunk_size)
        split_sizes: list[DimLike] = [chunk_size] * num_full
        if remainder > 0:
            split_sizes.append(remainder)
    else:
        split_sizes = list(split_size_or_sections)
    return _split_impl(x, split_sizes, axis)


top_k = functional(ops.top_k, rule=top_k_rule)
top_k.__doc__ = """Returns the ``k`` largest values along an axis with their indices.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    values, indices = F.top_k(x, k=2, axis=-1)
    # values is [5, 4] and indices is [3, 4]

Args:
    input: The input tensor from which to select the top ``k``.
    k: The number of values to select from ``input``. Must be in the range
        ``[0, input.shape[axis]]``.
    axis: The axis along which to select the top ``k``. Defaults to ``-1``.
        On a GPU input, only the last axis is supported.

Returns:
    A tuple of two ``Tensor`` objects. The first holds the top ``k`` values
    along ``axis``, and the second holds their ``int64`` indices in
    ``input``. Both tensors have the shape of ``input`` with the ``axis``
    dimension reduced to size ``k``.
"""

bottom_k = functional(ops.bottom_k, rule=top_k_rule)
bottom_k.__doc__ = """Returns the ``k`` smallest values along an axis with their indices.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    # The two smallest values are 1 and 2 at indices 0 and 2.
    values, indices = F.bottom_k(x, k=2, axis=-1)
    # values is [1, 2]
    # indices is [0, 2]

Args:
    input: The input tensor from which to select the bottom ``k``.
    k: The number of values to select from ``input``. Must be in the range
        ``[0, input.shape[axis]]``.
    axis: The axis along which to select the bottom ``k``. Defaults to
        ``-1``. On a GPU input, only the last axis is supported.

Returns:
    A tuple of two ``Tensor`` objects. The first holds the bottom ``k``
    values along ``axis`` in ascending order, and the second holds their
    ``int64`` indices in ``input``. Both tensors have the shape of
    ``input`` with the ``axis`` dimension reduced to size ``k``.
"""

chunk = functional(ops.chunk, rule=chunk_rule)
chunk.__doc__ = """Splits a tensor into equal-sized chunks along an axis.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1, 2, 3, 4, 5, 6])

    # Split into three equal chunks along axis 0
    parts = F.chunk(x, 3, axis=0)
    # parts[0] is [1, 2]
    # parts[1] is [3, 4]
    # parts[2] is [5, 6]

Args:
    x: The tensor to chunk.
    chunks: The number of chunks. Must be positive and evenly divide the
        size of ``x`` along ``axis``.
    axis: The axis to split along. Defaults to ``0``.

Returns:
    A list of ``Tensor`` objects (chunks), each the same size along
    ``axis``.

Raises:
    ValueError: If ``chunks`` does not evenly divide the size of ``x``
        along ``axis``, or if ``x`` is a scalar and ``chunks`` is greater
        than ``1``.
    IndexError: If ``axis`` is out of range for ``x``.
"""


def _reduce_op(
    graph_op: Callable[..., object],
    rule: Callable[..., ActionSet],
) -> Callable[..., Tensor]:
    """Builds a reduction wrapper.

    An integer ``axis`` delegates to the single-axis graph op;
    ``axis=None`` flattens to 1-D first.
    """
    single_axis = functional(graph_op, rule)

    def fn(
        x: Tensor,
        axis: int | None = -1,
    ) -> Tensor:
        assert isinstance(x, tensor.Tensor)
        if axis is None:
            x = reshape(x, [-1])
            axis = 0
        return single_axis(x, axis)

    return fn


def _reduce_elementwise_op(
    graph_op: Callable[..., object],
    rule: Callable[..., ActionSet],
    elementwise_fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[..., Tensor]:
    """Builds a function that reduces (1 arg) or runs elementwise (2 args)."""
    reduce_fn = _reduce_op(graph_op, rule)

    def fn(
        x: Tensor,
        y: Tensor | None = None,
        /,
        axis: int | None = -1,
    ) -> Tensor:
        if y is not None:
            return elementwise_fn(x, y)
        return reduce_fn(x, axis=axis)

    return fn


#: Computes the sum along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.sum` for details.
sum = _reduce_op(ops.sum, rule=linear_reduce_rule)
#: Computes the mean along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.mean` for details.
mean = _reduce_op(ops.mean, rule=mean_rule)
#: Computes the product along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.prod` for details.
prod = _reduce_op(ops.prod, rule=reduce_rule)
prod.__doc__ = """Computes the product of elements along a specified axis.

Args:
    x: The input tensor.
    axis: The axis along which to reduce. When ``None``, the tensor is
        flattened to 1-D and reduced. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the product along ``axis``. For an integer
    ``axis``, it has the same rank as ``x`` with the ``axis`` dimension
    reduced to size ``1``. When ``axis`` is ``None``, the result has shape
    ``(1,)``.

Raises:
    ValueError: If ``axis`` is out of range for ``x``.
"""

_argmax_impl = _reduce_op(ops.argmax, rule=reduce_rule)
_argmin_impl = _reduce_op(ops.argmin, rule=reduce_rule)


def argmax(
    x: Tensor,
    axis: int | None = -1,
) -> Tensor:
    """Returns the indices of the maximum values along an axis.

    It's useful for finding the position of the largest element along a
    given dimension, such as determining predicted classes in
    classification.

    When the input contains ties (identical maximum values), behavior
    depends on the device: CPU returns the first matching index, while
    GPU may return any of them.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]])
        indices = F.argmax(x, axis=-1)
        # indices has shape (2, 1): [[1], [2]]

        # Or flatten before reducing:
        flat_index = F.argmax(x, axis=None)
        # flat_index has shape (1,): [6] (flattened index of max value 4.2)

    Args:
        x: The input tensor.
        axis: The axis along which to compute the argmax. Negative values
            index from the last dimension. When ``None``, the tensor is
            flattened to 1-D first. Defaults to ``-1``.

    Returns:
        A ``Tensor`` with ``int64`` dtype containing the indices of the
        maximum values along ``axis``. For an integer ``axis``, the result
        has the same rank as ``x`` with the ``axis`` dimension reduced to
        size ``1``. When ``axis`` is ``None``, the result has shape ``(1,)``.

    Raises:
        ValueError: If ``axis`` is out of range for ``x``.
    """
    return _argmax_impl(x, axis=axis)


def argmin(
    x: Tensor,
    axis: int | None = -1,
) -> Tensor:
    """Returns the indices of the minimum values along an axis.

    When the input contains ties (identical minimum values), behavior
    depends on the device: CPU returns the first matching index, while
    GPU may return any of them.

    .. code-block:: python

        from max.experimental import Tensor
        from max.experimental import functional as F

        x = Tensor([[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]])
        indices = F.argmin(x, axis=-1)
        # indices has shape (2, 1): [[3], [1]]

    Args:
        x: The input tensor.
        axis: The axis along which to compute the argmin. Negative values
            index from the last dimension. When ``None``, the tensor is
            flattened to 1-D first. Defaults to ``-1``.

    Returns:
        A ``Tensor`` with ``int64`` dtype containing the indices of the
        minimum values along ``axis``. For an integer ``axis``, the result
        has the same rank as ``x`` with the ``axis`` dimension reduced to
        size ``1``. When ``axis`` is ``None``, the result has shape ``(1,)``.

    Raises:
        ValueError: If ``axis`` is out of range for ``x``.
    """
    return _argmin_impl(x, axis=axis)


max = _reduce_elementwise_op(
    ops.reduction.max,
    rule=reduce_rule,
    elementwise_fn=elementwise_max,
)
max.__doc__ = """Computes the maximum of a tensor, or the element-wise maximum of two tensors.

Called with one argument, reduces ``x`` along ``axis``. Called with two
tensor arguments, returns their element-wise maximum.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]])

    row_max = F.max(x, axis=-1)
    # row_max has shape (2, 1): [[3.5], [4.2]]

    col_max = F.max(x, axis=0)
    # col_max has shape (1, 4): [[2.3, 3.5, 4.2, 3.1]]

    y = Tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    element_wise = F.max(x, y)
    # element_wise: [[2.0, 3.5, 2.1, 2.0], [2.3, 2.0, 4.2, 3.1]]

Args:
    x: The input tensor.
    y: Optional second tensor. When provided, the result is the
        element-wise maximum of ``x`` and ``y``.
    axis: The axis to reduce along when ``y`` is omitted. When ``None``,
        the tensor is flattened to 1-D first. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing either the reduced maximum along ``axis`` or the
    element-wise maximum with the broadcast shape of the inputs.

Raises:
    ValueError: If ``axis`` is out of range for ``x`` when reducing.
"""

min = _reduce_elementwise_op(
    ops.reduction.min,
    rule=reduce_rule,
    elementwise_fn=elementwise_min,
)
min.__doc__ = """Computes the minimum of a tensor, or the element-wise minimum of two tensors.

Called with one argument, reduces ``x`` along ``axis``. Called with two
tensor arguments, returns their element-wise minimum.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]])

    row_min = F.min(x, axis=-1)
    # row_min has shape (2, 1): [[0.8], [1.9]]

    col_min = F.min(x, axis=0)
    # col_min has shape (1, 4): [[1.2, 1.9, 2.1, 0.8]]

    y = Tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    element_wise = F.min(x, y)
    # element_wise: [[1.2, 2.0, 2.0, 0.8], [2.0, 1.9, 2.0, 2.0]]

Args:
    x: The input tensor.
    y: Optional second tensor. When provided, the result is the
        element-wise minimum of ``x`` and ``y``.
    axis: The axis to reduce along when ``y`` is omitted. When ``None``,
        the tensor is flattened to 1-D first. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing either the reduced minimum along ``axis`` or the
    element-wise minimum with the broadcast shape of the inputs.

Raises:
    ValueError: If ``axis`` is out of range for ``x`` when reducing.
"""

#: Applies the softmax function along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.softmax` for details.
softmax = functional(ops.softmax, rule=softmax_rule)
#: Applies the log softmax function along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.logsoftmax` for details.
logsoftmax = functional(ops.logsoftmax, rule=softmax_rule)
#: Computes the cumulative sum along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.cumsum` for details.
cumsum = functional(ops.cumsum, rule=linear_reduce_rule)
cumsum.__doc__ = """Computes the cumulative sum of a tensor along an axis.

Args:
    x: The input tensor.
    axis: The axis along which to compute the cumulative sum. Defaults to
        ``-1``.
    exclusive: When ``True``, the first output value is ``0`` and the
        final input element is excluded from the sum. Defaults to
        ``False``.
    reverse: When ``True``, computes the sum starting from the end of the
        axis. Defaults to ``False``.

Returns:
    A tensor of the same shape and dtype where each element is the sum of
    the corresponding input elements up to that position along ``axis``.
"""


#: Applies 2D convolution. Distributed via SPMD.
#: See :func:`max.graph.ops.conv2d` for details.
conv2d = functional(ops.conv2d, rule=conv2d_rule)
conv2d.__doc__ = """Computes the 2-D convolution product of the input with the given filter, bias, strides, dilations, paddings, and groups.

This uses the following layout assumptions:

- The input has channels-last (NHWC) layout, meaning
  ``(batch_size, height, width, in_channels)``.
- The filter has RSCF layout, meaning
  ``(height, width, in_channels / num_groups, out_channels)``.
- The bias has shape ``(out_channels,)``.

The padding values are expected to take the form (pad_dim1_before,
pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
0's before and after the indicated *spatial* dimensions in the input. In
2-D convolution, dim1 here represents H and dim2 represents W. In
Python-like syntax, padding a 2x3 spatial input with [0, 1, 2, 1] would
yield:

.. code-block:: text

    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0],
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6

This op currently only supports strides and padding on the input.

Convolving a 2x2 input with an all-ones 2x2 filter sums the window:

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # NHWC input: batch 1, 2x2 spatial, 1 channel.
    x = Tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    # RSCF filter: 2x2, 1 in-channel, 1 out-channel, all ones.
    filter = Tensor([[[[1.0]], [[1.0]]], [[[1.0]], [[1.0]]]])
    result = F.conv2d(x, filter)
    # result is [[[[10]]]] with shape (1, 1, 1, 1)

Args:
    x: An NHWC input tensor to perform the convolution upon.
    filter: The convolution filter in RSCF layout,
        ``(height, width, in_channels / num_groups, out_channels)``.
    stride: The stride of the convolution operation.
    dilation: The spacing between the kernel points.
    padding: The amount of padding applied to the input.
    groups: When greater than 1, divides the convolution into multiple
        parallel convolutions. The number of input and output channels
        must both be divisible by the number of groups.
    bias: An optional 1-D bias of shape ``(out_channels,)``.
    input_layout: The layout of the input tensor. Defaults to NHWC.
    filter_layout: The layout of the filter tensor. Defaults to RSCF.

Returns:
    A ``Tensor`` containing the result of the convolution, with
    shape ``(batch_size, height_out, width_out, out_channels)``.

Raises:
    ValueError: If ``x`` isn't rank 4, ``filter`` isn't rank 4, ``bias`` is
        given and isn't rank 1, or ``x`` and ``filter`` aren't on the same
        device.
"""

conv3d = functional(ops.conv3d, rule=conv3d_rule)
conv3d.__doc__ = """Computes the 3-D convolution product of the input with the given filter, bias, strides, dilations, paddings, and groups.

This uses the following layout assumptions:

- The input has channels-last (NDHWC) layout, meaning
  ``(batch_size, depth, height, width, in_channels)``.
- The filter has QRSCF layout, meaning
  ``(depth, height, width, in_channels / num_groups, out_channels)``.

The padding values are expected to take the form (pad_dim1_before,
pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
0's before and after the indicated *spatial* dimensions in the input. In
3-D convolution, dim1 here represents D, dim2 represents H and dim3
represents W. In Python-like syntax, padding a 2x3 spatial input with
[0, 1, 2, 1] would yield:

.. code-block:: text

    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0],
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6

This op currently only supports strides and padding on the input.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    # NDHWC input: batch 1, 4x4x4 spatial, 1 channel.
    x = Tensor.ones((1, 4, 4, 4, 1), dtype=DType.float32)
    # QRSCF filter: 2x2x2, 1 in-channel, 1 out-channel.
    filter = Tensor.ones((2, 2, 2, 1, 1), dtype=DType.float32)
    result = F.conv3d(x, filter)
    # result has shape (1, 3, 3, 3, 1)

Args:
    x: An NDHWC input tensor to perform the convolution upon.
    filter: The convolution filter in QRSCF layout,
        ``(depth, height, width, in_channels / num_groups, out_channels)``.
    stride: The stride of the convolution operation.
    dilation: The spacing between the kernel points.
    padding: The amount of padding applied to the input.
    groups: When greater than 1, divides the convolution into multiple
        parallel convolutions. The number of input and output channels
        must both be divisible by the number of groups.
    bias: An optional 1-D bias of shape ``(out_channels,)``.
    input_layout: The layout of the input tensor. Defaults to NDHWC.
    filter_layout: The layout of the filter tensor. Defaults to QRSCF.

Returns:
    A ``Tensor`` containing the result of the convolution, with
    shape ``(batch_size, depth_out, height_out, width_out, out_channels)``.

Raises:
    ValueError: If ``x`` isn't rank 5, ``filter`` isn't rank 5, or ``bias``
        is given and isn't rank 1.
"""

conv2d_transpose = functional(ops.conv2d_transpose, rule=conv2d_transpose_rule)
conv2d_transpose.__doc__ = """Computes the 2-D deconvolution of the input with the given filter, strides, dilations, and paddings.

This computes the transpose (gradient) of convolution, with the following
layout assumptions (where ``out_channels`` is with respect to the original
convolution):

- The input ``x`` has channels-last (NHWC) layout, meaning
  ``(batch_size, height, width, in_channels)``.
- The filter has RSCF layout, meaning
  ``(kernel_height, kernel_width, out_channels, in_channels)``.
- The bias has shape ``(out_channels,)``.

This op effectively computes the gradient of a convolution with respect to
its input, as if the original convolution had the same filter and
hyperparameters as this op. For a visualization of the computation, see
`Transposed Convolution
<https://d2l.ai/chapter_computer-vision/transposed-conv.html>`_.

The padding values take the form ``(pad_dim1_before, pad_dim1_after,
pad_dim2_before, pad_dim2_after, ...)`` and are cropped (removed) from the
indicated *spatial* dimensions of the output. In 2-D transposed
convolution, ``dim1`` represents ``H_out`` and ``dim2`` represents
``W_out``. In Python-like syntax, cropping a 2x4 spatial output with
``[0, 1, 2, 1]`` would yield:

.. code-block:: text

    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    cropped_output = [
      [3],
    ]
    # Shape is 1x1

Deconvolving a 1x1 input with an all-ones 2x2 filter (filter is RSCF, with
``out_channels`` and ``in_channels`` with respect to the original
convolution):

.. code-block:: python

    from max.driver import CPU
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.experimental.tensor import default_device

    with default_device(CPU()):
        # NHWC input: batch 1, 1x1 spatial, 1 channel.
        x = Tensor([[[[3.0]]]])
        # RSCF filter: 2x2 kernel, 1 out-channel, 1 in-channel, all ones.
        filter = Tensor([[[[1.0]], [[1.0]]], [[[1.0]], [[1.0]]]])
        result = F.conv2d_transpose(x, filter)

Args:
    x: An NHWC input tensor to perform the deconvolution upon.
    filter: The convolution filter in RSCF layout,
        ``(height, width, out_channels, in_channels)``.
    stride: The stride of the sliding window as a tuple
        ``(stride_h, stride_w)``. Defaults to ``(1, 1)``.
    dilation: The spacing between the kernel points.
    padding: The amount cropped from each spatial dimension of the output.
    output_paddings: The number of zeros added at the end of each output
        spatial axis. This resolves the ambiguity between multiple output
        shapes when a stride is greater than 1. Only ``0`` is supported.
    bias: An optional tensor of shape ``(out_channels,)``.
    input_layout: The layout of the input tensor. Defaults to NHWC.
    filter_layout: The layout of the filter tensor. Defaults to RSCF.

Returns:
    A ``Tensor`` containing the result of the deconvolution, in
    channels-first (NCHW) layout
    ``(batch_size, out_channels, height_out, width_out)``. This differs from
    the channels-last (NHWC) input layout.

Raises:
    ValueError: If ``x`` isn't rank 4, ``filter`` isn't rank 4, ``bias`` is
        given and isn't rank 1, an output padding isn't smaller than its
        stride, or ``x`` and ``filter`` aren't on the same device.
"""


#: Copies a tensor setting everything outside a central band to zero. Distributed via SPMD.
#: See :func:`max.graph.ops.band_part` for details.
band_part = functional(ops.band_part, rule=band_part_rule)
band_part.__doc__ = """Set all values to zero except a diagonal band of an input matrix.

All but the last two axes are treated as batches, and
the last two axes define the matrices.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    # Keep the main diagonal and one sub-diagonal, producing
    # [[1, 0, 0], [1, 1, 0], [0, 1, 1]].
    result = F.band_part(x, num_lower=1, num_upper=0)
    # result is [[1, 0, 0], [1, 1, 0], [0, 1, 1]]

Args:
    x: The input tensor to mask.
    num_lower: The number of diagonal bands to include below the central
        diagonal. If ``None`` or ``-1``, includes the entire lower triangle.
        Defaults to ``None``.
    num_upper: The number of diagonal bands to include above the central
        diagonal. If ``None`` or ``-1``, includes the entire upper triangle.
        Defaults to ``None``.
    exclude: Whether to invert the selection, zeroing out the elements in
        the band instead. Defaults to ``False``.

Returns:
    A ``Tensor`` containing ``x`` with the masked-out elements set to zero
    and the remaining elements copied from ``x``. It has the same shape and
    dtype as ``x``.

Raises:
    ValueError: If the input tensor rank is less than 2, or if ``num_lower``
        or ``num_upper`` are out of bounds for statically known dimensions.
"""

fold = functional(ops.fold, rule=fold_rule)
fold.__doc__ = """Combines an array of sliding local blocks into a larger tensor.

``L``, the number of blocks, must equal ``prod((output_size[d] + 2 *
padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) // stride[d] + 1)``,
where ``d`` ranges over all spatial dimensions.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    # Shape (N, C * kernel_h * kernel_w, L) = (1, 1 * 2 * 2, 9).
    x = Tensor.ones((1, 4, 9), dtype=DType.float32)
    # Fold nine 2x2 blocks into a 4x4 image.
    result = F.fold(x, output_size=(4, 4), kernel_size=(2, 2))
    # result has shape (1, 1, 4, 4)

Args:
    input: The 3-D tensor to fold, with shape
        ``(N, C * kernel_sizes, L)``, where ``N`` is the batch dimension,
        ``C`` is the number of channels, ``kernel_sizes`` is the product of
        the kernel sizes, and ``L`` is the number of local blocks.
    output_size: The spatial dimensions of the output tensor, as a tuple
        of two ints.
    kernel_size: The size of the sliding blocks, as a tuple of two ints.
    stride: The stride of the sliding blocks. Either an int or a tuple of
        two ints. Defaults to ``1``.
    dilation: The spacing between kernel elements. Either an int or a
        tuple of two ints. Defaults to ``1``.
    padding: The zero-padding added on both sides of the input. Either an
        int or a tuple of two ints. Defaults to ``0``.

Returns:
    A ``Tensor`` containing the folded 4-D tensor, with shape
    ``(N, C, output_size[0], output_size[1])``.

Raises:
    ValueError: If the input's channel dimension isn't a multiple of the
        total kernel size, or if the number of blocks ``L`` doesn't match
        the value computed from the other arguments.
"""

as_interleaved_complex = functional(
    ops.complex.as_interleaved_complex,
    rule=as_interleaved_complex_rule,
)
as_interleaved_complex.__doc__ = """Reshapes a real tensor of alternating (real, imag) values into complex form.

Pulls each adjacent ``(real, imag)`` pair in the last dimension out into
a trailing pair of size 2.

Args:
    x: A real tensor representing complex numbers as alternating pairs of
        ``(real, imag)`` values. The last dimension must have an even
        size.

Returns:
    A tensor of shape ``(*x.shape[:-1], x.shape[-1] // 2, 2)``. All
    dimensions except the last are unchanged; the last dimension is
    halved, and a final dimension of size 2 is appended to hold the
    ``(real, imag)`` components.
"""

complex_mul = functional(ops.complex.mul, rule=binary_rule)
complex_mul.__doc__ = """Multiplies two complex-valued tensors element-wise.

Both inputs must use the interleaved complex representation (trailing
dimension of size 2).

Args:
    lhs: The left-hand side complex tensor.
    rhs: The right-hand side complex tensor.

Returns:
    A complex tensor with the broadcast shape containing element-wise
    products.
"""

resize = functional(ops.resize, rule=resize_rule)
resize.__doc__ = """Resizes a tensor to a given shape using a specified interpolation method.

.. code-block:: python

    from max.driver import CPU
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.experimental.tensor import default_device
    from max.graph.ops import InterpolationMode

    with default_device(CPU()):
        # NCHW input: batch 1, 1 channel, 2x2 spatial.
        x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Upscale the spatial dimensions to 4x4.
        result = F.resize(x, [1, 1, 4, 4], InterpolationMode.BILINEAR)
        # result has shape (1, 1, 4, 4)

Args:
    input: The input tensor to resize. Must be rank 4 in channels-first
        (NCHW) layout, ``(batch_size, channels, height, width)``.
    shape: The desired output shape, of length 4, layout
        ``(batch_size, channels, height, width)``.
    interpolation: The interpolation method, given as an
        :class:`~max.graph.ops.InterpolationMode`. Defaults to
        :attr:`~max.graph.ops.InterpolationMode.BILINEAR`.

Returns:
    A ``Tensor`` containing the resized tensor with the given ``shape``.

Raises:
    ValueError: If ``input`` doesn't have rank 4, or if ``shape`` has the
        wrong number of elements.
"""

resize_linear = functional(ops.resize_linear, rule=resize_linear_rule)
#: Resizes a tensor using nearest-neighbor interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_nearest` for details.
resize_nearest = functional(ops.resize_nearest, rule=resize_nearest_rule)
#: Resizes a tensor using bicubic interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_bicubic` for details.
resize_bicubic = functional(ops.resize_bicubic, rule=resize_bicubic_rule)
#: Computes the inverse real FFT. Distributed via SPMD.
#: See :func:`max.graph.ops.irfft` for details.
irfft = functional(ops.irfft, rule=irfft_rule)
irfft.__doc__ = """Computes the inverse of the real-input FFT.

Args:
    input_tensor: The input tensor to compute the inverse real FFT of.
    n: The size of the output tensor. The input tensor is padded or
        truncated to ``n // 2 + 1`` along ``axis``.
    axis: The axis along which to compute the inverse FFT. Defaults to
        ``-1``.
    normalization: The normalization to apply to the output tensor. One of
        ``"backward"``, ``"ortho"``, or ``"forward"``. When ``"backward"``,
        the output is divided by ``n``. When ``"ortho"``, the output is
        divided by ``sqrt(n)``. When ``"forward"``, no normalization is
        applied.
    input_is_complex: Whether the input tensor is already interleaved
        complex. When ``True``, the last dimension of the input tensor must
        be 2, and is excluded from the dimension referred to by ``axis``.
    buffer_size_mb: The estimated size of a persistent buffer to use for
        storage of intermediate results. Needs to be the same across
        multiple calls to ``irfft`` within the same graph.

Returns:
    A real tensor that is the inverse FFT of the complex input. The shape
    matches the input shape, except along ``axis``, which is replaced by
    ``n``.
"""


def _cond_graph(
    pred: TensorValueLike,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[[], Iterable[TensorValueLike] | TensorValueLike | None],
    else_fn: Callable[[], Iterable[TensorValueLike] | TensorValueLike | None],
) -> list[TensorValue]:
    """``ops.cond`` requires a CPU predicate — inserts a transfer when needed."""
    pred = TensorValue(pred)
    if not pred.device.is_cpu():
        pred = ops.transfer_to(pred, CPU())
    return ops.cond(pred, out_types, then_fn, else_fn)


cond = functional(_cond_graph, rule=cond_rule)
cond.__doc__ = """Conditionally executes one of two branches based on a boolean predicate.

Both branches must return the same number and types of values as
specified by ``out_types``. The predicate is evaluated at runtime to
determine which branch executes. If ``pred`` lives on a non-CPU device,
it is transferred to CPU automatically.

.. code-block:: python

    from max.driver import CPU
    from max.dtype import DType
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.graph import DeviceRef, TensorType

    def then_fn():
        return Tensor([1.0, 2.0], dtype=DType.float32, device=CPU())

    def else_fn():
        return Tensor([10.0, 20.0], dtype=DType.float32, device=CPU())

    pred = Tensor(True, dtype=DType.bool, device=CPU())
    out_types = [TensorType(DType.float32, [2], DeviceRef.CPU())]
    (result,) = F.cond(pred, out_types, then_fn, else_fn)
    # pred is True, so result is [1.0, 2.0]

Args:
    pred: A boolean scalar tensor of type :attr:`~max.dtype.DType.bool`
        determining which branch to execute.
    out_types: The expected output types for both branches. Use
        :obj:`None` for branches that don't return values (such as
        buffer mutations).
    then_fn: A callable executed when ``pred`` is ``True``.
    else_fn: A callable executed when ``pred`` is ``False``.

Returns:
    The output values from the executed branch, or an empty list when
    ``out_types`` is :obj:`None`.
"""


def _while_loop_graph(
    initial_values: Iterable[TensorValueLike] | TensorValueLike,
    predicate: Callable[..., Tensor],
    body: Callable[..., Tensor | Iterable[Tensor]],
) -> list[TensorValue]:
    """Wrap predicate/body so callbacks see :class:`Tensor`.

    ``ops.while_loop`` passes :class:`TensorValue` into its predicate/body
    and expects :class:`TensorValue` back. This wrapper wraps callback
    args as :class:`Tensor` and coerces callback returns back to
    :class:`TensorValue`. The outer ``functional()`` wrapper converts the
    returned :class:`TensorValue` list back to :class:`Tensor` for the
    public surface.
    """

    def _pred(*args: TensorValue) -> TensorValue:
        tensors = [Tensor.from_graph_value(a) for a in args]
        return TensorValue(predicate(*tensors))

    def _body(*args: TensorValue) -> list[TensorValue]:
        tensors = [Tensor.from_graph_value(a) for a in args]
        result = body(*tensors)
        if isinstance(result, Tensor):
            return [TensorValue(result)]
        return [TensorValue(t) for t in result]

    if isinstance(initial_values, Iterable):
        unwrapped = [TensorValue(v) for v in initial_values]
    else:
        unwrapped = [TensorValue(initial_values)]
    return ops.while_loop(unwrapped, _pred, _body)


while_loop = functional(_while_loop_graph, rule=while_loop_rule)
while_loop.__doc__ = """Repeatedly executes a body function while a predicate holds.

Both ``predicate`` and ``body`` receive and return :class:`Tensor`
values. They take the same number and types of arguments as the initial
values. The predicate must return a single boolean scalar tensor that
controls loop continuation, and that tensor must reside on CPU; the body
must return updated values matching the types of ``initial_values``.

.. code-block:: python

    from max.driver import CPU
    from max.dtype import DType
    from max.experimental import Tensor
    from max.experimental import functional as F

    def predicate(x):
        return x < 10

    def body(x):
        return x + 1

    x = Tensor(0, dtype=DType.int32, device=CPU())
    (result,) = F.while_loop(x, predicate, body)
    # Loop continues until ``x >= 10``; result is ``10``.

Args:
    initial_values: The initial values for the loop arguments. Must be
        non-empty.
    predicate: A callable that takes the loop arguments and returns a
        boolean scalar tensor of type :attr:`~max.dtype.DType.bool`.
    body: A callable that takes the loop arguments and returns updated
        values matching the types of ``initial_values``.

Returns:
    The output values from the final loop iteration.
"""


# Mutation ops: hand-rolled because they write in-place via __buffervalue__().


def _spmd_buffer_write(
    destination: Tensor,
    source: Tensor,
    write: Callable[[Any, Any], None],
) -> None:
    """Per-shard in-place write; flows each post-write value back to ``destination._state``."""
    shards = list(destination.local_shards)
    src_shards = list(source.local_shards) if source.is_distributed else None
    for i, dest_tensor in enumerate(shards):
        dest_shard = dest_tensor.__buffervalue__()
        src_shard = (
            src_shards[i].__tensorvalue__()
            if src_shards is not None
            else source.__tensorvalue__()
        )
        write(dest_shard, src_shard)
        if destination._state is not None and dest_tensor._state is not None:
            new_values = list(destination._state.values)
            new_values[i] = dest_tensor._state.value
            destination._state = type(destination._state)(
                tuple(new_values), destination._state.ctx
            )


def buffer_store(destination: Tensor, source: Tensor) -> None:
    """Stores values from a tensor into a tensor buffer.

    Args:
        destination: The destination buffer tensor.
        source: The source tensor whose values are written into
            ``destination``.
    """
    if destination.is_distributed:
        buffer_store_rule(
            tensor_to_layout(destination), tensor_to_layout(source)
        )

    with ensure_context():
        if destination.is_distributed:
            _spmd_buffer_write(destination, source, ops.buffer_store)
        else:
            ops.buffer_store(
                destination.__buffervalue__(), source.__tensorvalue__()
            )


def buffer_store_slice(
    destination: Tensor,
    source: Tensor,
    indices: SliceIndices,
) -> None:
    """Stores values into a slice of a tensor buffer.

    Args:
        destination: The destination buffer tensor.
        source: The source tensor whose values are written into the slice.
        indices: The slice specification within ``destination`` to write to.
    """
    if destination.is_distributed:
        buffer_store_slice_rule(
            tensor_to_layout(destination), tensor_to_layout(source), indices
        )

    with ensure_context():
        if destination.is_distributed:

            def _write(dest_buf: Any, src_tv: Any) -> None:
                dest_buf[indices] = src_tv

            _spmd_buffer_write(destination, source, _write)
        else:
            dest_buf = destination.__buffervalue__()
            source_tv = source.__tensorvalue__()
            dest_buf[indices] = source_tv


#: Applies group normalization.
#: See :func:`max.graph.ops.group_norm` for details.
group_norm = functional(ops.group_norm)
#: Applies RMS normalization.
#: See :func:`max.graph.ops.rms_norm` for details.
rms_norm = functional(ops.rms_norm, rule=rms_norm_rule)
#: Filters boxes with high intersection-over-union.
#: See :func:`max.graph.ops.non_maximum_suppression` for details.
non_maximum_suppression = functional(ops.non_maximum_suppression)
non_maximum_suppression.__doc__ = """Filters boxes with high intersection-over-union (IoU).

Applies greedy non-maximum suppression independently per (batch, class)
pair. For each pair, the algorithm:

1. Discards boxes whose score is at or below ``score_threshold``.
2. Sorts the remaining boxes by score in descending order.
3. Greedily selects boxes, suppressing any later candidate whose IoU with
   an already-selected box exceeds ``iou_threshold``.
4. Stops after ``max_output_boxes_per_class`` selections per pair.

Boxes use ``(y1, x1, y2, x2)`` corner format. Coordinates may be normalized
or absolute, since the op handles both. All inputs must be on CPU.

.. code-block:: python

    from max.driver import CPU
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.dtype import DType

    device = CPU()
    # boxes: (batch, num_boxes, 4); scores: (batch, num_classes, num_boxes).
    boxes = Tensor.ones((1, 3, 4), dtype=DType.float32, device=device)
    scores = Tensor.ones((1, 1, 3), dtype=DType.float32, device=device)
    # Each output row is (batch_index, class_index, box_index), with a
    # data-dependent number of rows.
    result = F.non_maximum_suppression(
        boxes,
        scores,
        max_output_boxes_per_class=Tensor(2, dtype=DType.int64, device=device),
        iou_threshold=Tensor(0.5, dtype=DType.float32, device=device),
        score_threshold=Tensor(0.0, dtype=DType.float32, device=device),
    )
    # result has shape (num_selected, 3)

Args:
    boxes: The input boxes tensor of shape
        ``(batch_size, num_boxes, 4)``, with a float dtype.
    scores: The per-class scores of shape
        ``(batch_size, num_classes, num_boxes)``, with the same dtype as
        ``boxes``.
    max_output_boxes_per_class: A scalar ``int64`` tensor giving the
        maximum number of boxes to select per (batch, class) pair.
    iou_threshold: A scalar float tensor giving the IoU suppression
        threshold.
    score_threshold: A scalar float tensor giving the minimum score to
        consider.
    out_dim: The name for the dynamic output dimension, which is the number
        of selected boxes. Defaults to ``"num_selected"``.

Returns:
    A ``Tensor`` containing the selected boxes, with shape
    ``(out_dim, 3)`` and ``int64`` dtype. Each row is
    ``(batch_index, class_index, box_index)``.
"""

roi_align = functional(ops.roi_align)
roi_align.__doc__ = """Applies ROI-align pooling.

Extracts fixed-size feature maps from regions of interest (ROIs) using
bilinear interpolation.

Args:
    input: The input tensor in channels-last (NHWC) layout,
        ``(batch_size, height, width, channels)``.
    rois: The regions of interest with shape ``(num_rois, 5)``, where each
        row is ``(batch_index, x1, y1, x2, y2)``.
    output_height: The height of each output feature map.
    output_width: The width of each output feature map.
    spatial_scale: The multiplicative factor mapping ROI coordinates to
        input spatial coordinates. Defaults to ``1.0``.
    sampling_ratio: The number of sampling points per bin in each
        direction. ``0`` means adaptive (``ceil(bin_size)``). Defaults to
        ``0.0``.
    aligned: When ``True``, applies a half-pixel offset to ROI
        coordinates for more precise alignment. Defaults to ``False``.
    mode: The pooling mode, either ``"AVG"`` or ``"MAX"``. Defaults to
        ``"AVG"``.

Returns:
    A ``Tensor`` containing the pooled values, with shape
    ``(num_rois, output_height, output_width, channels)``.

Raises:
    ValueError: If ``input`` isn't rank 4, ``rois`` isn't rank 2 with
        5 columns, or ``mode`` is invalid.
"""


def clamp(
    x: Tensor,
    lower_bound: TensorValueLike,
    upper_bound: TensorValueLike,
) -> Tensor:
    """Clamps tensor values to ``[lower_bound, upper_bound]``."""
    return max(min(x, upper_bound), lower_bound)


clip = clamp
rebind = functional(ops.rebind, rule=rebind_rule)
rebind.__doc__ = """Rebinds the symbolic shape of a tensor.

Asserts at runtime that the tensor's dimensions match the new shape.
Useful for narrowing dynamic dimensions to specific sizes when you have
external knowledge of their values.

Args:
    x: The input tensor.
    shape: The new symbolic shape.
    message: A message included in the runtime assertion if the shapes
        don't match. Defaults to ``""``.
    layout: An optional filter layout to attach to the result. Defaults
        to :obj:`None`.

Returns:
    A tensor with the same data and the new symbolic shape.
"""

group_norm.__doc__ = """Computes group normalization over the channel axis of ``input``.

Splits the channel axis (axis 1) of ``input`` into ``num_groups``
groups, computes the mean and variance within each group, and
normalizes. ``gamma`` and ``beta`` then apply a per-channel affine
transform. Useful when the batch axis is small enough that batch
normalization is unstable.

.. note::

    This op executes only on CUDA/HIP GPU targets.

Args:
    input: The tensor to normalize, of shape ``(batch, channels, ...)``.
    gamma: The per-channel scale applied after normalization. A 1-D
        tensor whose length matches the channel axis of ``input``.
    beta: The per-channel bias added after scaling. A 1-D tensor with
        the same shape as ``gamma``.
    num_groups: The number of groups to split the channel axis into.
        Must divide the channel size evenly.
    epsilon: A small positive constant added to the variance for
        numerical stability.

Returns:
    A ``Tensor`` with the same shape and dtype as ``input``.

Raises:
    ValueError: If ``input`` has fewer than 2 dimensions.
"""
layer_norm.__doc__ = """Computes layer normalization over the last dimension of ``input``.

The output is ``gamma * (input - mean) / sqrt(var + epsilon) + beta``,
where ``mean`` and ``var`` are reduced over the last axis of ``input``
and broadcast back across the leading axes.

Reduction is performed in the dtype of ``input``. For numerically stable
normalization on float16 or bfloat16 inputs, cast to float32 before
calling this op and cast the result back.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1.0, 3.0]])
    gamma = Tensor([1.0, 1.0])
    beta = Tensor([0.0, 0.0])
    result = F.layer_norm(x, gamma, beta, epsilon=1e-5)
    # Each row is normalized to zero mean and approximately unit variance.

Args:
    input: The tensor to normalize. Reduction runs over the last axis.
    gamma: The scale applied after normalization. A 1-D tensor whose
        length matches the last dimension of ``input``.
    beta: The bias added after scaling. A 1-D tensor with the same
        shape as ``gamma``.
    epsilon: A small positive constant added to the variance for
        numerical stability.

Returns:
    A ``Tensor`` with the same shape and dtype as ``input``.

Raises:
    ValueError: If ``gamma`` or ``beta`` does not match the last
        dimension of ``input``, or if ``epsilon`` is not positive.
"""
logsoftmax.__doc__ = """Computes the log-softmax of a tensor along an axis.

Args:
    value: The input to the log-softmax computation. Must have a
        floating-point dtype.
    axis: The axis along which to compute the log-softmax. Defaults to the
        final axis (``-1``).

Returns:
    A ``Tensor`` of the same shape and dtype as ``value`` containing the
    log-softmax of ``value`` computed along ``axis``.
"""
mean.__doc__ = """Computes the mean of elements along a specified axis.

Args:
    x: The input tensor.
    axis: The axis along which to reduce. When ``None``, the tensor is
        flattened to 1-D and reduced. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the mean along ``axis``. For an integer ``axis``,
    it has the same rank as ``x`` with the ``axis`` dimension reduced to size
    ``1``. When ``axis`` is ``None``, the result has shape ``(1,)``.

Raises:
    ValueError: If ``axis`` is out of range for ``x``.
"""
pad.__doc__ = """Pads a tensor along every dimension.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1, 2], [3, 4]])

    # Pad one element before and after each dimension.
    result = F.pad(x, [1, 1, 1, 1])
    # [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]

Args:
    input: The tensor to pad.
    paddings: The amount to pad. For a tensor of rank ``N``, pass ``2*N``
        non-negative integers in the order ``[before_dim0, after_dim0,
        before_dim1, after_dim1, ...]``.
    mode: How to fill the padded cells. Supported values:

        * ``"constant"``: fill using ``value``.
        * ``"reflect"``: reflect the content across each edge, excluding
          the boundary element (like ``numpy.pad`` with ``mode='reflect'``).
        * ``"edge"``: repeat the nearest boundary element (like
          ``numpy.pad`` with ``mode='edge'``).
    value: The fill value for ``mode="constant"``. Defaults to ``0``.

Returns:
    A ``Tensor`` containing the padded input, with the same dtype as
    ``input``.

Raises:
    ValueError: If ``mode`` is unsupported, or any padding value is
        negative.
    AssertionError: If the number of padding values isn't twice the input
        rank.
"""
qmatmul.__doc__ = """Performs matrix multiplication between floating point and quantized tensors.

Quantizes the ``lhs`` floating point value to match the encoding of the
``rhs`` quantized value, performs the matmul, and then dequantizes the
result. Compared to a regular matmul op, this one expects the ``rhs`` value
to be transposed. For example, if the ``lhs`` shape is ``[32, 64]`` and the
quantized ``rhs`` shape is also ``[32, 64]``, then the output shape is
``[32, 32]``. That is, this function returns the result from:

.. code-block:: text

    dequantize(quantize(lhs) @ transpose(rhs))

The last two dimensions in ``lhs`` are treated as matrices and multiplied
by ``rhs`` (which must be a 2-D tensor). Any remaining dimensions in
``lhs`` are broadcast dimensions.

.. note::

    This currently supports ``Q4_0``, ``Q4_K``, ``Q6_K``, and supported
    ``GPTQ`` configurations.

Args:
    encoding: The quantization encoding to use.
    config: The quantization config. Pass ``None`` for Vroom encodings;
        a supported configuration is required for GPTQ.
    lhs: The non-quantized, left-hand side of the matmul.
    rhs: The transposed and quantized right-hand side tensor(s).

Returns:
    A ``Tensor`` containing the dequantized, floating point result.

Raises:
    ValueError: If ``encoding`` is not a supported quantization encoding.
    TypeError: If ``lhs`` or ``rhs`` has an unsupported dtype or rank.
    AssertionError: If GPTQ is selected without a configuration.
"""
resize_bicubic.__doc__ = """Resizes a tensor using bicubic interpolation.

Produces an output tensor whose dimensions are given by ``size`` using a
4x4-pixel Keys/PyTorch (``a = -0.75``) cubic convolution filter with
half-pixel coordinate mapping.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # NCHW input: batch 1, 1 channel, 2x2 spatial.
    x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    # Upscale the spatial dimensions to 4x4.
    result = F.resize_bicubic(x, [1, 1, 4, 4])
    # result has shape (1, 1, 4, 4)

Args:
    input: The input tensor to resize. Must be rank 4 in channels-first
        (NCHW) layout, ``(batch_size, channels, height, width)``.
    size: The desired output shape, of length 4,
        ``(batch_size, channels, height, width)``.

Returns:
    A ``Tensor`` containing the resized tensor, with shape ``size``
    and the same dtype as ``input``.

Raises:
    ValueError: If ``input`` doesn't have rank 4, or if ``size`` has a
        different length.
"""
resize_linear.__doc__ = """Resizes a tensor using linear (bilinear) interpolation.

Produces an output tensor whose shape is given by ``size`` using separable
1-D linear filters. It resizes any dimension whose size changes, including
the batch and channel dimensions. The operation maps output coordinates
back to input coordinates according to ``coordinate_transform_mode``.

.. code-block:: python

    from max.driver import CPU
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.experimental.tensor import default_device

    with default_device(CPU()):
        # NCHW input: batch 1, 1 channel, 2x2 spatial.
        x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Upscale the spatial dimensions to 4x4.
        result = F.resize_linear(x, [1, 1, 4, 4])
        # result has shape (1, 1, 4, 4)

Args:
    input: The input tensor to resize.
    size: The desired output shape. Must have the same rank as ``input``.
    coordinate_transform_mode: How to map an output coordinate to an input
        coordinate. Allowed values:

        - ``0`` (``half_pixel``): Default. Shifts by 0.5 before
          scaling, consistent with most deep learning frameworks.
        - ``1`` (``align_corners``): Aligns the corner pixels of the input
          and output so that the first and last coordinates are preserved
          exactly.
        - ``2`` (``asymmetric``): Applies no shift, mapping each output
          coordinate to ``coordinate / scale``.
        - ``3`` (``half_pixel_1D``): Like ``half_pixel``, except any axis
          whose output size is ``1`` maps to coordinate ``0``.
    antialias: When ``True``, applies an antialiasing filter when
        downscaling, which reduces aliasing artifacts by widening the tent
        filter support by ``1 / scale``. Has no effect when upscaling.
        Defaults to ``False``.

Returns:
    A ``Tensor`` containing the resized tensor, with shape ``size``
    and the same dtype as ``input``.

Raises:
    ValueError: If ``coordinate_transform_mode`` isn't 0-3, or if ``size``
        has a different rank than ``input``.
"""
resize_nearest.__doc__ = """Resizes a tensor using nearest-neighbor interpolation.

Produces an output tensor whose dimensions are given by ``size`` by
selecting the nearest input sample for each output coordinate.

.. code-block:: python

    from max.driver import CPU
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.experimental.tensor import default_device

    with default_device(CPU()):
        # NCHW input: batch 1, 1 channel, 2x2 spatial.
        x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Upscale the spatial dimensions to 4x4.
        result = F.resize_nearest(x, [1, 1, 4, 4])
        # result has shape (1, 1, 4, 4)

Args:
    input: The input tensor to resize.
    size: The desired output shape. Must have the same rank as ``input``.
    coordinate_transform_mode: How to map an output coordinate to an input
        coordinate. Allowed values:

        - ``0`` (``half_pixel``). Default.
        - ``1`` (``align_corners``).
        - ``2`` (``asymmetric``).
        - ``3`` (``half_pixel_1D``).

        See :func:`resize_linear` for a description of each mode.
    round_mode: How to round the mapped coordinate to select the nearest
        input sample. Allowed values:

        - ``0`` (``HalfDown``, the default): ``ceil(x - 0.5)``.
        - ``1`` (``HalfUp``): ``floor(x + 0.5)``.
        - ``2`` (``Floor``): ``floor(x)``.
        - ``3`` (``Ceil``): ``ceil(x)``.

Returns:
    A ``Tensor`` containing the resized tensor, with shape ``size``
    and the same dtype as ``input``.

Raises:
    ValueError: If ``coordinate_transform_mode`` isn't 0-3, ``round_mode``
        isn't 0-3, or ``size`` has a different rank than ``input``.
"""
rms_norm.__doc__ = """Computes root mean square normalization over the last dimension of ``input``.

The output is ``input / rms(input) * (weight + weight_offset)`` where
``rms(x) = sqrt(mean(x ** 2) + epsilon)``. Reduction runs over the last
axis of ``input`` and is broadcast back across the leading axes. See
`Root Mean Square Layer Normalization
<https://arxiv.org/abs/1910.07467>`_ for the original formulation.

Two variants are supported through ``weight_offset`` and
``multiply_before_cast``:

- **Llama-style** (default): ``weight_offset=0`` and
  ``multiply_before_cast=False``. The normalized input is cast to the
  output dtype before multiplication by the weight.
- **Gemma-style**: ``weight_offset=1`` and ``multiply_before_cast=True``.
  The weight is treated as ``1 + weight`` and multiplication runs in
  the reduction dtype before casting back.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[3.0, 4.0]])
    weight = Tensor([1.0, 1.0])
    # Llama-style (default).
    y_llama = F.rms_norm(x, weight, epsilon=1e-6)
    # Gemma-style treats the weight as 1 + weight.
    y_gemma = F.rms_norm(
        x, weight, epsilon=1e-6, weight_offset=1.0, multiply_before_cast=True
    )

Args:
    input: The tensor to normalize. Reduction runs over the last axis.
    weight: The scale applied after normalization. A 1-D tensor whose
        shape matches the last dimension of ``input``.
    epsilon: A small positive constant added to the mean of squares for
        numerical stability.
    weight_offset: A value added to ``weight`` before scaling. Use
        ``1.0`` for Gemma-style normalization and ``0.0`` otherwise.
        Defaults to ``0.0``.
    multiply_before_cast: Whether to multiply by the (offset) weight
        before casting the normalized input back to the output dtype.
        Llama-style sets this to ``False``. Defaults to ``False``.

Returns:
    A ``Tensor`` with the same shape and dtype as ``input``.

Raises:
    ValueError: If ``weight`` does not match the last dimension of
        ``input``.
"""
softmax.__doc__ = """Computes the softmax of a tensor along an axis.

Normalizes the values along ``axis`` so that they sum to ``1``, with each
output element representing the exponentiated input divided by the sum of
exponentiated values along that axis.

Args:
    value: The input to the softmax computation. Must have a floating-point
        dtype.
    axis: The axis along which to compute the softmax. Defaults to the
        final axis (``-1``).

Returns:
    A ``Tensor`` of the same shape and dtype as ``value`` containing the
    softmax of ``value`` computed along ``axis``.
"""
squeeze.__doc__ = """Removes a dimension of size ``1`` from a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    # x has shape (2, 1, 3).
    x = Tensor.ones([2, 1, 3])
    # Remove the size-1 dimension at axis 1, producing shape (2, 3).
    result = F.squeeze(x, 1)

Args:
    x: The input tensor to squeeze.
    axis: The dimension to remove from the input's shape. If negative, this
        indexes from the end of the tensor. For example, a value of ``-1``
        removes the last dimension.

Returns:
    A ``Tensor`` containing ``x`` with the dimension at ``axis`` removed.
    That dimension size must equal ``1``, so the result holds the same
    elements as ``x`` with one fewer dimension.

Raises:
    ValueError: If the dimension at ``axis`` does not have size ``1``.
    IndexError: If ``axis`` is out of range, including for a rank-zero input.
"""
sum.__doc__ = """Computes the sum of elements along a specified axis.

Args:
    x: The input tensor.
    axis: The axis along which to reduce. When ``None``, the tensor is
        flattened to 1-D and reduced. Defaults to ``-1``.

Returns:
    A ``Tensor`` containing the sum along ``axis``. For an integer ``axis``,
    it has the same rank as ``x`` with the ``axis`` dimension reduced to size
    ``1``. When ``axis`` is ``None``, the result has shape ``(1,)``.

Raises:
    ValueError: If ``axis`` is out of range for ``x``.
"""
