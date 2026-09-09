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
"""Correctness tests for the row-wise reduction ops (Row API).

Covers every non-composite reduction routed through MAX's own Mojo Row API:
the pure reductions (``reduce_sum/max/min/mean/product``, ``argmax``,
``argmin``, ``reduce_min_and_max``) and the last-axis norm-type ops
(``softmax``, ``logsoftmax``, ``layer_norm``, ``rms_norm``,
``row_mean_of_squares``). Each op is checked against a float32 torch reference
across a few small column counts (even + odd), on the inner axis and -- for the
pure arbitrary-axis reductions -- on a non-inner axis, in bfloat16, float16 and
float32 where precision makes it meaningful, plus a CPU-only 8-byte-element
group (see ``test_rowwise_wide_element_cpu``) for what only the widest element
can reach.

Every case runs on both the CPU and, when an accelerator is present, the GPU --
the GPU is where the cooperative tiers, the cross-thread combine and the split-K
path live, so it is the priority target rather than an afterthought. One further
inner-axis group is GPU-only: the ops whose monoid state is narrower than the
4-byte word that combine exchanges (see
``test_rowwise_inner_gpu_subword_state``), which no CPU case can reach.

These are small, fast CI shapes. The bandwidth-oriented perf grid for the same
ops lives in the manual benchmark at
``//utils/benchmarking/kepler/graph:reductions`` and is not run here.

Each parametrization builds one symbolic-dimension graph and loads it once, then
feeds several concrete shapes as data (per the guidance against per-case graph
recompilation) -- so the compile count equals the number of parametrizations,
not the number of shapes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from test_common.reduction_graphs import (
    LAYER_NORM_EPS,
    PURE_REDUCTIONS,
    RMS_NORM_EPS,
    build_reduction,
)

# GPU is the priority target: every cooperative tier (warp / block), the
# cross-thread combine and the split-K path exist only there, while the CPU path
# is a serial per-row walk. So the whole matrix below runs on both devices, with
# the GPU half skipped when no accelerator is present.
_DEVICES = [DeviceRef.CPU()] + (
    [DeviceRef.GPU()] if accelerator_count() > 0 else []
)
_DEV_IDS = ["cpu", "gpu"][: len(_DEVICES)]

# Ops whose output is integer indices.
_INT_OUT = {"argmax", "argmin"}

# Small inner-axis column counts: even + odd, small + a wider row.
_INNER_COLS = [32, 127, 128, 512]
_INNER_ROWS = 8
# Non-inner (reduce over axis 0): (reduce_len, cols). Even + odd reduce length,
# even + odd cols, and short reduce axes crossed with SIMD-divisible and
# non-SIMD-divisible cols at small and large output counts.
_NONINNER_SHAPES = [(33, 16), (64, 33), (8, 8192), (8, 8193), (8, 9)]


def _torch_dtype(dtype: DType) -> torch.dtype:
    return {
        DType.bfloat16: torch.bfloat16,
        DType.float16: torch.float16,
        DType.bool: torch.bool,
        DType.int8: torch.int8,
        DType.int16: torch.int16,
        DType.int64: torch.int64,
        DType.float64: torch.float64,
    }.get(dtype, torch.float32)


def _build_graph(op: str, dtype: DType, axis: int, dev: DeviceRef) -> Graph:
    """Build a single-reduction graph over `axis` with symbolic dimensions."""
    x_t = TensorType(dtype, ["r", "c"], device=dev)
    w_t = TensorType(dtype, ["c"], device=dev)
    if op == "layer_norm":
        input_types = [x_t, w_t, w_t]  # x, gamma, beta
    elif op == "rms_norm":
        input_types = [x_t, w_t]
    else:
        input_types = [x_t]

    with Graph(f"{op}_ax{axis}", input_types=input_types) as graph:
        x = graph.inputs[0].tensor
        weights = [inp.tensor for inp in graph.inputs[1:]]
        graph.output(build_reduction(op, x, axis, weights=weights))
    return graph


def _make_input(op: str, rows: int, cols: int, dtype: DType) -> torch.Tensor:
    """Random [rows, cols] input; near-1 for product to avoid under/overflow."""
    torch.manual_seed(0)
    if dtype == DType.bool:
        # Sparse enough that a dropped element flips the row's max/min: most
        # rows are all-False with a single True (and vice versa for min).
        f = torch.rand(rows, cols, dtype=torch.float32)
        return (f > 1.0 - 2.0 / cols) | (f < 1.0 / cols)
    if dtype in (DType.int8, DType.int16, DType.int64):
        return torch.randint(-100, 100, (rows, cols), dtype=torch.int32).to(
            _torch_dtype(dtype)
        )
    if op == "reduce_product":
        f = 1.0 + 0.02 * torch.randn(rows, cols, dtype=torch.float32)
    else:
        f = torch.randn(rows, cols, dtype=torch.float32)
    return f.to(_torch_dtype(dtype))


def _make_weight(rows_seed: int, cols: int, dtype: DType) -> torch.Tensor:
    torch.manual_seed(rows_seed)
    w = 0.1 * torch.randn(cols, dtype=torch.float32) + 1.0
    return w.to(_torch_dtype(dtype))


def _feed(model: Model, tensors: list[torch.Tensor]) -> list[Buffer]:
    bufs = [
        Buffer.from_dlpack(t).to(model.input_devices[i])
        for i, t in enumerate(tensors)
    ]
    return model.execute(*bufs)


def _read_f32(buf: Buffer) -> np.ndarray:
    """Device buffer -> float32 numpy (bf16 read via torch, numpy lacks bf16)."""
    b = buf if buf.device.is_host else buf.to(CPU())
    if b.dtype == DType.bfloat16:
        return torch.from_dlpack(b).to(torch.float32).numpy()
    return b.to_numpy().astype(np.float32)


def _tol(op: str, dtype: DType) -> tuple[float, float, float]:
    """(atol, rtol, frac_allowed) comparing in float32; half looser than fp32.

    Tolerances follow the validated benchmark harness: bf16 reductions
    accumulate in-dtype, so a small fraction of large-magnitude rows can exceed
    a per-term tolerance (a broken kernel fails ~all elements, not a few). fp16
    shares the bf16 budget -- it accumulates in-dtype just the same, with a
    wider mantissa, so the bf16 numbers bound it.
    """
    half = dtype in (DType.bfloat16, DType.float16)
    if op in (
        "reduce_max",
        "reduce_min",
        "argmax",
        "argmin",
        "reduce_min_and_max",
    ):
        # max/min and the selected argmax/argmin value are representable
        # exactly, so both dtypes match to a tight absolute tolerance.
        return 1e-3, 0.0, 0.0
    if op == "reduce_sum":
        return (0.15, 5e-2, 2e-2) if half else (1e-2, 1e-3, 0.0)
    if op in ("reduce_mean", "row_mean_of_squares"):
        return (2e-2, 3e-2, 2e-2) if half else (1e-3, 1e-3, 0.0)
    if op == "softmax":
        return (3e-3, 6e-2, 5e-3) if half else (1e-4, 1e-3, 0.0)
    if op == "logsoftmax":
        return (5e-2, 6e-2, 5e-3) if half else (1e-3, 1e-3, 0.0)
    if op in ("layer_norm", "rms_norm"):
        return (4e-2, 6e-2, 1e-2) if half else (2e-3, 2e-3, 0.0)
    raise ValueError(f"no tolerance for {op!r}")


def _assert_close(
    got: np.ndarray, ref: np.ndarray, op: str, dtype: DType, label: str
) -> None:
    atol, rtol, frac = _tol(op, dtype)
    got = got.reshape(-1)
    ref = ref.reshape(-1)
    abs_err = np.abs(got - ref)
    over = abs_err > (atol + rtol * np.abs(ref))
    n_bad = int(over.sum())
    allowed = math.ceil(frac * got.size)
    assert n_bad <= allowed, (
        f"{op} {label}: {n_bad}/{got.size} over tol "
        f"(allowed {allowed}); max_abs_err={abs_err.max():.4g}"
    )


def _reference_and_check(
    op: str,
    dtype: DType,
    model: Model,
    x: torch.Tensor,
    axis: int,
    weights: list[torch.Tensor],
    label: str,
) -> None:
    """Run the model and compare to a float32 torch reference for `op`."""
    xf = x.to(torch.float32)
    outs = _feed(model, [x, *weights])

    if op in _INT_OUT:
        idx = (
            outs[0] if outs[0].device.is_host else outs[0].to(CPU())
        ).to_numpy()
        idx = idx.astype(np.int64)
        got_val = np.take_along_axis(xf.numpy(), idx, axis=axis)
        ref = (
            xf.amax(dim=axis, keepdim=True)
            if op == "argmax"
            else xf.amin(dim=axis, keepdim=True)
        ).numpy()
        # Tie-safe: the value at the chosen index must equal the true extremum.
        assert np.array_equal(got_val, ref), (
            f"{op} {label}: index selected a non-extreme value"
        )
        return

    if op == "reduce_min_and_max":
        got = _read_f32(outs[0])
        norm_axis = axis + xf.dim() if axis < 0 else axis
        got_min = np.take(got, 0, axis=norm_axis)
        got_max = np.take(got, 1, axis=norm_axis)
        _assert_close(
            got_min, xf.amin(dim=axis).numpy(), op, dtype, f"{label}/min"
        )
        _assert_close(
            got_max, xf.amax(dim=axis).numpy(), op, dtype, f"{label}/max"
        )
        return

    if op == "reduce_product":
        got = _read_f32(outs[0]).reshape(-1)
        ref = xf.prod(dim=axis).numpy().reshape(-1)
        assert np.isfinite(got).all(), f"{op} {label}: non-finite output"
        denom = np.maximum(np.abs(ref), 1e-3)
        med_rel = float(np.median(np.abs(got - ref) / denom))
        # bf16 in-dtype accumulation vs the fp32 reference differs by tree order;
        # a directional check (finite, right order of magnitude) is enough.
        assert med_rel < 0.5, f"{op} {label}: median rel err {med_rel:.3f}"
        return

    got = _read_f32(outs[0])
    if op == "reduce_sum":
        ref = xf.sum(dim=axis).numpy()
    elif op == "reduce_max":
        ref = xf.amax(dim=axis).numpy()
    elif op == "reduce_min":
        ref = xf.amin(dim=axis).numpy()
    elif op == "reduce_mean":
        ref = xf.mean(dim=axis).numpy()
    elif op == "softmax":
        ref = torch.softmax(xf, dim=axis).numpy()
    elif op == "logsoftmax":
        ref = torch.log_softmax(xf, dim=axis).numpy()
    elif op == "row_mean_of_squares":
        ref = (xf**2).mean(dim=-1).numpy()
    elif op == "layer_norm":
        gamma, beta = weights[0].to(torch.float32), weights[1].to(torch.float32)
        ref = torch.nn.functional.layer_norm(
            xf, (xf.shape[-1],), gamma, beta, eps=LAYER_NORM_EPS
        ).numpy()
    elif op == "rms_norm":
        weight = weights[0].to(torch.float32)
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        ref = (xf * torch.rsqrt(ms + RMS_NORM_EPS) * weight).numpy()
    else:
        raise ValueError(f"no reference for {op!r}")
    _assert_close(got, ref, op, dtype, label)


# Inner-axis matrix: precision-sensitive ops in both dtypes; exact
# selection/compare ops (max/min_and_max/arg) in bfloat16 only.
#
# reduce_min/reduce_mean/logsoftmax/argmin are deliberately absent: each is
# redundant with a sibling already in this matrix (reduce_max, reduce_sum,
# softmax, argmax respectively) -- same monoid/combine mechanics, differing
# only in comparison direction or a trivial final step (a divide, or the
# elementwise map after the shared OnlineLogSumExp reduce). A bug in the
# shared scaffolder path shows up in the sibling that's still tested; the two
# confirmed bugs that genuinely needed both directions tested (the bool
# subword-state crash, the split-K join_parallel bug) have their own narrow
# regression coverage below (`_GPU_SUBWORD_CASES`,
# `test_rowwise_inner_gpu_splitk`) that still exercises both regardless of
# this trim.
_INNER_BOTH = [
    "reduce_sum",
    "reduce_product",
    "softmax",
    "layer_norm",
    "rms_norm",
    "row_mean_of_squares",
]
_INNER_BF16_ONLY = [
    "reduce_max",
    "argmax",
    "reduce_min_and_max",
]
_INNER_CASES = (
    [(op, DType.bfloat16) for op in _INNER_BOTH + _INNER_BF16_ONLY]
    + [(op, DType.float32) for op in _INNER_BOTH]
    + [(op, DType.float16) for op in _INNER_BOTH]
)


@pytest.mark.parametrize("dev", _DEVICES, ids=_DEV_IDS)
@pytest.mark.parametrize(
    "op,dtype", _INNER_CASES, ids=lambda v: v if isinstance(v, str) else str(v)
)
def test_rowwise_inner(
    session: InferenceSession, op: str, dtype: DType, dev: DeviceRef
) -> None:
    model = session.load(_build_graph(op, dtype, axis=-1, dev=dev))
    for cols in _INNER_COLS:
        x = _make_input(op, _INNER_ROWS, cols, dtype)
        weights: list[torch.Tensor] = []
        if op == "layer_norm":
            weights = [
                _make_weight(1, cols, dtype),
                (0.1 * torch.randn(cols)).to(_torch_dtype(dtype)),
            ]
        elif op == "rms_norm":
            weights = [_make_weight(1, cols, dtype)]
        _reference_and_check(op, dtype, model, x, -1, weights, f"cols={cols}")


# Non-inner (reduce over axis 0): the pure arbitrary-axis reductions. Precision-
# sensitive sum in both dtypes; the rest in bfloat16 only. See the inner-axis
# matrix above for why reduce_min/reduce_mean/argmin are absent (redundant
# with reduce_max/reduce_sum/argmax).
_NONINNER_CASES = [
    (op, DType.bfloat16)
    for op in [
        "reduce_sum",
        "reduce_max",
        "reduce_product",
        "argmax",
        "reduce_min_and_max",
    ]
] + [(op, DType.float32) for op in ["reduce_sum"]]


@pytest.mark.parametrize("dev", _DEVICES, ids=_DEV_IDS)
@pytest.mark.parametrize(
    "op,dtype",
    _NONINNER_CASES,
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_rowwise_noninner(
    session: InferenceSession, op: str, dtype: DType, dev: DeviceRef
) -> None:
    model = session.load(_build_graph(op, dtype, axis=0, dev=dev))
    for rows, cols in _NONINNER_SHAPES:
        x = _make_input(op, rows, cols, dtype)
        _reference_and_check(op, dtype, model, x, 0, [], f"shape={rows}x{cols}")


# 8-byte elements, both axes -- CPU only (MAX's reduce dtype set has no 64-bit
# element on GPU). The widest element is what turns a wrong tile alignment into a
# crash rather than a slow path: the alignment a body claims per tile is scaled
# by the element size before it reaches the backend, so only on the widest
# element can an off-by-a-factor claim reach a whole SIMD register and select an
# alignment-checked instruction. Nothing narrower can catch that, which is why
# every dtype above passed while float64 segfaulted. Both axes, since the claim
# rides every load, and both 8-byte dtypes, since the claim is a function of the
# element's width and not of its kind -- int64 faulted on the same ops.
#
# int64 skips reduce_mean and reduce_product: integer division and integer
# overflow make them a comparison against the float32 reference rather than
# against the alignment claim, which every other op here already pins.
_WIDE_CASES = [(op, DType.float64) for op in PURE_REDUCTIONS] + [
    (op, DType.int64)
    for op in PURE_REDUCTIONS
    if op not in ("reduce_mean", "reduce_product")
]


@pytest.mark.parametrize("axis", [-1, 0], ids=["inner", "noninner"])
@pytest.mark.parametrize(
    "op,dtype", _WIDE_CASES, ids=lambda v: v if isinstance(v, str) else str(v)
)
def test_rowwise_wide_element_cpu(
    session: InferenceSession, op: str, dtype: DType, axis: int
) -> None:
    model = session.load(
        _build_graph(op, dtype, axis=axis, dev=DeviceRef.CPU())
    )
    shapes = (
        [(_INNER_ROWS, cols) for cols in _INNER_COLS]
        if axis == -1
        else _NONINNER_SHAPES
    )
    for rows, cols in shapes:
        x = _make_input(op, rows, cols, dtype)
        _reference_and_check(
            op, dtype, model, x, axis, [], f"shape={rows}x{cols}"
        )


# Sub-word monoid state, on an accelerator. The Row scaffolder's cross-thread
# combine (`Reducer.generic`, max/kernels/src/algorithm/gpu/rowwise.mojo) exists
# only on GPU and exchanges the monoid's width-1 state in whole 4-byte words, so
# a monoid whose state is narrower than one word is reachable only here -- the
# CPU cases above cannot see it, and it is silent memory corruption rather than a
# crash. Only the inner axis reaches the combine: a non-inner reduction lands on
# the tiled tier, where `rowwise.pjoin` no-ops because each thread owns its own
# output column.
#
# Which (op, dtype) pairs have a sub-word state, per each monoid's
# `join_parallel` in max/kernels/src/algorithm/reduce_op.mojo:
#   reduce_product  -- no hardware-fast scalar product exists, so every dtype
#                      takes the combine; fp16/bf16 make the state 2 bytes.
#   reduce_max/min  -- the fast-reducer dtype list covers only the 32/64-bit
#                      dtypes, so `bool` (1 byte) falls through.
#
# The sub-word integers are the same bug class -- `int8`/`int16` are legal graph
# op inputs for `reduce_max`/`reduce_min`/`reduce_min_and_max` and sit outside
# every fast-reducer dtype list -- but the legacy kernels this commit still
# precedes cannot compile them, so those cases belong with the commit that
# routes each op through the Row API (`bool` is not legal for `min_and_max`).
_GPU_SUBWORD_CASES = [
    ("reduce_product", DType.float16),
    ("reduce_product", DType.bfloat16),
    ("reduce_max", DType.bool),
    ("reduce_min", DType.bool),
]


@pytest.mark.skipif(
    accelerator_count() == 0, reason="the GPU cross-thread combine needs a GPU"
)
@pytest.mark.parametrize(
    "op,dtype",
    _GPU_SUBWORD_CASES,
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_rowwise_inner_gpu_subword_state(
    session: InferenceSession, op: str, dtype: DType
) -> None:
    model = session.load(_build_graph(op, dtype, axis=-1, dev=DeviceRef.GPU()))
    for cols in _INNER_COLS:
        x = _make_input(op, _INNER_ROWS, cols, dtype)
        _reference_and_check(op, dtype, model, x, -1, [], f"cols={cols}")


# Split-K cross-block join, on an accelerator. A few-row, long-row shape routes
# the inner-axis reduction through the split-K tier (`num_rows` under the SM
# count and at least `_SPLITK_MIN_ROW` = 32768 elements per row at
# `simd_width <= 4`, i.e. float32 here), whose cross-block finish in
# `rowwise.pjoin` is the only place the scaffolder combines whole per-block
# monoid states. Every other case in this file is orders of magnitude below that
# element floor, so nothing else reaches it.
#
# argmax/argmin are what catch a mistake there: their cross-thread step ends by
# publishing the winning index into the field the emit reads, so a finish that
# skips it returns a real-but-not-extreme index -- a wrong answer that still
# looks like a plausible one. reduce_max rides the same call with an exactly
# representable result, so it pins the non-arg monoids on that path too.
_SPLITK_ROWS = 8
_SPLITK_COLS = 40960


@pytest.mark.skipif(
    accelerator_count() == 0, reason="the split-K tier is GPU-only"
)
@pytest.mark.parametrize("op", ["argmax", "argmin", "reduce_max"])
def test_rowwise_inner_gpu_splitk(session: InferenceSession, op: str) -> None:
    dtype = DType.float32
    model = session.load(_build_graph(op, dtype, axis=-1, dev=DeviceRef.GPU()))
    x = _make_input(op, _SPLITK_ROWS, _SPLITK_COLS, dtype)
    _reference_and_check(op, dtype, model, x, -1, [], f"cols={_SPLITK_COLS}")
