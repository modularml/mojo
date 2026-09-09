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
"""Ops for generating random numbers."""

from __future__ import annotations

import weakref
from collections.abc import MutableMapping
from dataclasses import replace

import numpy as np
from max._core.dialects import kgen, rmo
from max.dtype import DType

from .. import dtype_promotion
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .elementwise import _accum_type

SEEDS: MutableMapping[Graph, TensorValue] = weakref.WeakKeyDictionary()


def SeedType(device: DeviceRef) -> TensorType:
    """Graph-input tensor type for an RNG seed: rank-1 ``[1]`` ``uint64`` on ``device``."""
    return TensorType(DType.uint64, [1], device=device)


def _rotate_seed(seed: TensorValue):  # noqa: ANN202
    # Let's just get some different random numbers
    # from the initial seed for now.
    return seed + 1


def assert_scalar(value: TensorValueLike) -> None:
    """Raises :class:`ValueError` if value is not a scalar (has non-empty ``shape``)."""
    if isinstance(value, np.ndarray | TensorValue) and value.shape:
        raise ValueError("Expected a scalar value")


def _peek_seed():  # noqa: ANN202
    graph = Graph.current
    try:
        return SEEDS[graph]
    except LookupError:
        raise RuntimeError("No seed set! Set with `ops.random.set_seed`.")  # noqa: B904


def _next_seed_for(device: DeviceRef) -> TensorValue:
    graph = Graph.current
    seed = _peek_seed()
    SEEDS[graph] = _rotate_seed(seed)
    if seed.device != device:
        seed = seed.to(device)
    return seed


def _normalize_seed(seed: TensorValueLike | int) -> TensorValue:
    if isinstance(seed, int):
        seed = dtype_promotion._promote_to_strong(
            seed, DType.uint64, DeviceRef.CPU()
        )
    seed_tv = TensorValue(seed)
    if seed_tv.dtype != DType.uint64:
        seed_tv = seed_tv.cast(DType.uint64)
    if seed_tv.rank == 0:
        seed_tv = seed_tv.reshape([1])
    if seed_tv.shape != [1]:
        raise ValueError(
            "Seed must be a scalar or rank-1 tensor with shape [1],"
            f" got shape {seed_tv.shape}."
        )
    return seed_tv


def set_seed(seed: TensorValueLike | int = 0) -> None:
    """Sets the seed for random numbers generated in the graph.

    Call this once per graph. Subsequent random ops rotate from this seed.
    For per-execution seeds, pass a :func:`SeedType` graph input.

    Args:
        seed: A Python int, a rank-0 ``uint64`` scalar
            :class:`~max.graph.TensorValue`, or a rank-1 ``[1]`` ``uint64``
            :class:`~max.graph.TensorValue`.
    """
    SEEDS[Graph.current] = _normalize_seed(seed)


def gaussian(
    like: TensorType,
    mean: TensorValueLike = 0,
    std: TensorValueLike = 1,
) -> TensorValue:
    """Samples from a Gaussian (normal) distribution with the given mean and standard deviation.

    A seed must be set on the current graph with :func:`set_seed`.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        with Graph("gaussian_example") as graph:
            ops.random.set_seed(0)
            like = TensorType(DType.float32, [2, 3], device=device)
            graph.output(ops.random.gaussian(like, mean=0.0, std=1.0))

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # result is a (2, 3) tensor sampled from a standard normal distribution.

    Args:
        like: A :class:`~max.graph.TensorType` whose shape, dtype, and device
            determine the output tensor.
        mean: The mean of the Gaussian distribution. Must be a scalar.
            Defaults to ``0``.
        std: The standard deviation of the Gaussian distribution. Must be a
            scalar. Defaults to ``1``.

    Returns:
        A ``TensorValue`` with the same shape and dtype as ``like``, representing
        values sampled from the specified Gaussian distribution.

    Raises:
        RuntimeError: If no seed has been set with :func:`set_seed`.
        ValueError: If ``mean`` or ``std`` are not scalar values.
    """
    assert_scalar(mean)
    assert_scalar(std)
    # Check whether we have a seed before we add other constants to the graph.
    seed = _next_seed_for(like.device)
    accum_type = _accum_type(like) if like.dtype.is_float() else DType.float32
    random_accum = Graph.current._add_op_generated(
        rmo.MoRandomNormalOp,
        result=replace(like, dtype=accum_type),
        shape=TensorValue(like.shape),
        mean=dtype_promotion._promote_to_strong(
            mean, DType.float32, DeviceRef.CPU()
        ),
        variance=dtype_promotion._promote_to_strong(
            std, DType.float32, DeviceRef.CPU()
        ),
        seed=seed,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if not like.dtype.is_float():
        random_accum = round(random_accum)
    return random_accum.cast(like.dtype)


# Alias normal <-> gaussian
normal = gaussian


def uniform(
    like: TensorType,
    range: tuple[TensorValueLike, TensorValueLike] = (0, 1),
) -> TensorValue:
    """Samples uniformly from the half-open interval ``[lower, upper)``.

    Values satisfy ``lower ≤ x < upper``. A seed must be set on the current graph with
    :func:`set_seed`.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        with Graph("uniform_example") as graph:
            ops.random.set_seed(0)
            like = TensorType(DType.float32, [2, 3], device=device)
            graph.output(ops.random.uniform(like, range=(0.0, 1.0)))

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # result is a (2, 3) tensor sampled uniformly from [0.0, 1.0).

    Args:
        like: A :class:`~max.graph.TensorType` whose shape, dtype, and device
            determine the output tensor.
        range: A tuple ``(lower, upper)`` specifying the half-open interval to
            sample from. Both bounds must be scalars. Defaults to ``(0, 1)``.

    Returns:
        A ``TensorValue`` with the same shape and dtype as ``like``,
        representing values sampled uniformly from ``[lower, upper)``.

    Raises:
        RuntimeError: If no seed has been set with :func:`set_seed`.
        ValueError: If ``lower`` or ``upper`` are not scalar values.
    """
    lower, upper = range

    assert_scalar(lower)
    assert_scalar(upper)
    # Check whether we have a seed before we add other constants to the graph.
    seed = _next_seed_for(like.device)
    return Graph.current._add_op_generated(
        rmo.MoRandomUniformOp,
        result=like,
        shape=TensorValue(like.shape),
        lower_bound=dtype_promotion._promote_to_strong(
            lower, like.dtype, DeviceRef.CPU()
        ),
        upper_bound=dtype_promotion._promote_to_strong(
            upper, like.dtype, DeviceRef.CPU()
        ),
        seed=seed,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
