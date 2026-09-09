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

"""Multi-GPU execution test for the fused reduce-scatter + RMSNorm graph op.

Exercises both dispatch branches (the fused kernel below the fuse threshold and
the two-launch fallback above it) and, in the grouped case, the device grouping
the handler applies: per-group peer/signal slicing, the group-local rank, and
per-group ragged row binning. This is the only test that runs the op through the
graph compiler and its Mojo handler -- the kernel-level test
(`max/kernels/test/gpu/comm/test_reducescatter_rmsnorm.mojo`) calls the `comm`
API directly and re-implements the grouping, so it cannot catch a handler bug.

Each device gets a DISTINCT gamma: with a shared gamma, indexing gammas by the
group-local rank instead of the global device index is undetectable.
"""

from __future__ import annotations

from typing import Any, cast

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.nn import Signals

COLS = 6144  # M3 hidden size; the fuse threshold is bytes = rows*COLS*2.
EPS = 1e-6
WEIGHT_OFFSET = 1.0  # M3 Gemma-style: gamma_eff = gamma + 1.0.

# `_dispatch_rs_norm` fuses at/below 128 rows * COLS * 2 B on group-rank 0.
FUSE_THRESHOLD_ROWS_PER_RANK = 128


def _to_device_bf16(arr: np.ndarray, device: Accelerator) -> Buffer:
    """Host float array -> bf16 device buffer (uint16 view: DLPack-safe)."""
    bits = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _from_device_bf16(buf: Buffer) -> np.ndarray:
    """bf16 device buffer -> host float32 (via the uint16 view)."""
    out = buf.copy(device=CPU())
    return (
        out.view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def _bf16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(ml_dtypes.bfloat16).astype(np.float32)


def _rs_rms_norm_graph(
    signals: Signals,
    rows_per_device: list[int],
    group_size: int | None,
    with_residual: bool = True,
) -> Graph:
    devices = signals.devices
    num_devices = len(devices)
    if not with_residual:
        return _rs_rms_norm_graph_no_residual(
            signals, rows_per_device, group_size
        )
    with Graph(
        "reduce_scatter_rms_norm",
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(
                    dtype=DType.bfloat16,
                    shape=[rows_per_device[i], COLS],
                    device=device,
                )
                for i, device in enumerate(devices)
            ]
            + [
                TensorType(dtype=DType.bfloat16, shape=[COLS], device=device)
                for device in devices
            ]
            + [
                TensorType(
                    dtype=DType.bfloat16,
                    shape=[rows_per_device[i], COLS],
                    device=device,
                )
                for i, device in enumerate(devices)
            ]
            + signals.input_types(),
        ),
    ) as graph:
        inputs = [v.tensor for v in graph.inputs[:num_devices]]
        gammas = [v.tensor for v in graph.inputs[num_devices : 2 * num_devices]]
        residuals = [
            v.tensor for v in graph.inputs[2 * num_devices : 3 * num_devices]
        ]
        sigs = [v.buffer for v in graph.inputs[3 * num_devices :]]
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=inputs,
            signal_buffers=sigs,
            gammas=gammas,
            epsilon=EPS,
            residuals=residuals,
            weight_offset=WEIGHT_OFFSET,
            group_size=group_size,
        )
        graph.output(*normed, *residual)
        return graph


def _rs_rms_norm_graph_no_residual(
    signals: Signals,
    rows_per_device: list[int],
    group_size: int | None,
) -> Graph:
    """The op with `residuals` omitted: a plain reduce-scatter + norm.

    No residual input block at all, so the operand slots the op still requires
    are the ones the builder fills with the inputs. That is what makes the
    zero-residual oracle in `_check` load-bearing here: if `has_residual` were
    not threaded, the kernel would add each device's own activations to the
    sum and the oracle would miss by a whole input.
    """
    devices = signals.devices
    num_devices = len(devices)
    with Graph(
        "reduce_scatter_rms_norm_no_residual",
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(
                    dtype=DType.bfloat16,
                    shape=[rows_per_device[i], COLS],
                    device=device,
                )
                for i, device in enumerate(devices)
            ]
            + [
                TensorType(dtype=DType.bfloat16, shape=[COLS], device=device)
                for device in devices
            ]
            + signals.input_types(),
        ),
    ) as graph:
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:num_devices]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * num_devices :]],
            gammas=[
                v.tensor for v in graph.inputs[num_devices : 2 * num_devices]
            ],
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
            group_size=group_size,
        )
        graph.output(*normed, *residual)
        return graph


def _shard_rows(rows: int, group_size: int, local_rank: int) -> int:
    """Ragged binning: remainder rows go to the low group-local ranks."""
    return (rows + (group_size - local_rank - 1)) // group_size


def _host_rmsnorm(shard_f32: np.ndarray, gamma_f32: np.ndarray) -> np.ndarray:
    """Reference norm of the bf16-rounded shard, gamma folded in f32 (mbc)."""
    m2 = np.mean(shard_f32**2, axis=-1, keepdims=True)
    nf = 1.0 / np.sqrt(m2 + EPS)
    return _bf16(shard_f32 * nf * (gamma_f32 + WEIGHT_OFFSET))


def _inputs_and_gammas(
    rows_per_device: list[int],
    devices: list[Accelerator],
    group_size: int,
) -> tuple[
    list[Buffer],
    list[Buffer],
    list[Buffer],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Positive varied per-device activations + a DISTINCT gamma per device.

    The residual is REPLICATED within each group, which the op requires: every
    rank adds its own shard of it, so ranks disagreeing on the residual would
    each fold a different value into the same group sum.
    """
    tensor_inputs: list[Buffer] = []
    gamma_inputs: list[Buffer] = []
    residual_inputs: list[Buffer] = []
    host_acts: list[np.ndarray] = []
    host_gammas: list[np.ndarray] = []
    host_residuals: list[np.ndarray] = []
    offset = 0
    for i, rows in enumerate(rows_per_device):
        size = rows * COLS
        arr = (((np.arange(size) + offset) % 251) + 1).reshape(rows, COLS)
        arr = arr.astype(np.float32)
        host_acts.append(_bf16(arr))
        tensor_inputs.append(_to_device_bf16(arr, devices[i]))
        offset += size

        # Keyed on the GROUP, so it is bit-identical within one and differs
        # across them: folding a sibling group's residual would show up.
        group_id = i // group_size
        res = (
            (((np.arange(size) + 13 * group_id) % 17) - 8)
            .reshape(rows, COLS)
            .astype(np.float32)
        )
        host_residuals.append(_bf16(res))
        residual_inputs.append(_to_device_bf16(res, devices[i]))

        # Per-device gamma spaced 0.5 apart, WELL outside the 2e-2 tolerance
        # below: at 0.01 the tolerance swallows the difference and a
        # group-local-vs-device gamma index swap goes undetected.
        gamma = ((np.arange(COLS) % 7) * 0.01) + (i * 0.5)
        gamma = gamma.astype(np.float32)
        host_gammas.append(_bf16(gamma))
        gamma_inputs.append(_to_device_bf16(gamma, devices[i]))
    return (
        tensor_inputs,
        gamma_inputs,
        residual_inputs,
        host_acts,
        host_gammas,
        host_residuals,
    )


def _check(
    outputs: list[Any],
    num_gpus: int,
    group_size: int,
    rows_per_device: list[int],
    host_acts: list[np.ndarray],
    host_gammas: list[np.ndarray],
    host_residuals: list[np.ndarray],
    label: str,
) -> None:
    normed_out = outputs[:num_gpus]
    residual_out = outputs[num_gpus : 2 * num_gpus]

    for group_start in range(0, num_gpus, group_size):
        group = list(range(group_start, group_start + group_size))
        group_rows = rows_per_device[group_start]
        # The reduce-scatter sums ONLY this group's devices; a handler that
        # slices the wrong window reduces a different (or a cross-group) set.
        # Peer sum rounded to bf16 first, then the residual added and rounded
        # again -- the order both arms use. It enters the total exactly once.
        group_sum = _bf16(
            _bf16(np.sum([host_acts[d] for d in group], axis=0))
            + host_residuals[group_start]
        )

        row = 0
        for local_rank, dev_idx in enumerate(group):
            units = _shard_rows(group_rows, group_size, local_rank)
            shard = group_sum[row : row + units]
            row += units

            res = _from_device_bf16(cast(Buffer, residual_out[dev_idx]))
            assert res.shape == (units, COLS), (
                f"{label}: residual shape on GPU {dev_idx}: {res.shape} != "
                f"{(units, COLS)}"
            )
            # The residual is the plain reduce-scatter shard, rounded to bf16 --
            # exactly representable, so this is a bit-identity gate.
            assert np.array_equal(res, shard), (
                f"{label}: residual mismatch on GPU {dev_idx} "
                f"(local_rank={local_rank})"
            )

            nrm = _from_device_bf16(cast(Buffer, normed_out[dev_idx]))
            ref = _host_rmsnorm(shard, host_gammas[dev_idx])
            max_abs = float(np.max(np.abs(nrm - ref))) if units else 0.0
            # Loose bf16 reduction-order tolerance; the tight ULP gate lives
            # in the kernel test. Any wrong gamma/eps/divisor/window is bigger.
            assert np.allclose(nrm, ref, rtol=2e-2, atol=2e-2), (
                f"{label}: normed mismatch on GPU {dev_idx} "
                f"(local_rank={local_rank}): max_abs={max_abs}"
            )


@pytest.mark.parametrize(
    "num_gpus, rows, regime",
    [
        (4, 8, "fused"),  # 2 rows/rank -> below threshold -> fused kernel
        (4, 1024, "two_launch"),  # 256 rows/rank -> above -> two-launch
    ],
)
def test_reduce_scatter_rms_norm_no_residual_execution(
    num_gpus: int, rows: int, regime: str
) -> None:
    """Omitting the residual gives a plain reduce-scatter + norm, both arms.

    Both arms, because the residual is folded in two structurally different
    places -- inside the fused kernel, and in the two-launch arm's
    reduce-scatter epilogue -- so a flag threaded to only one of them passes a
    single-arm test.
    """
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for {regime} reduce_scatter_rms_norm."
        )

    rows_per_device = [rows] * num_gpus
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _rs_rms_norm_graph(
        signals, rows_per_device, group_size=None, with_residual=False
    )
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, _, host_acts, host_gammas, _ = (
        _inputs_and_gammas(rows_per_device, devices, num_gpus)
    )
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *signals.buffers()
    )
    # A zero residual is the exact oracle: adding 0.0 then rounding to bf16 is
    # the identity, so `_check` reduces to reduce-scatter + norm.
    zero_residuals = [np.zeros_like(a) for a in host_acts]
    _check(
        list(outputs),
        num_gpus,
        num_gpus,
        rows_per_device,
        host_acts,
        host_gammas,
        zero_residuals,
        f"{regime}-no-residual",
    )


@pytest.mark.parametrize(
    "num_gpus, rows, regime",
    [
        (4, 8, "fused"),  # 2 rows/rank -> below threshold -> fused kernel
        (4, 1024, "two_launch"),  # 256 rows/rank -> above -> two-launch
    ],
)
def test_reduce_scatter_rms_norm_execution(
    num_gpus: int, rows: int, regime: str
) -> None:
    """Full-world op: both dispatch branches, per-device gamma."""
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for {regime} reduce_scatter_rms_norm."
        )

    rows_per_device = [rows] * num_gpus
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _rs_rms_norm_graph(signals, rows_per_device, group_size=None)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    (
        tensor_inputs,
        gamma_inputs,
        residual_inputs,
        host_acts,
        host_gammas,
        host_residuals,
    ) = _inputs_and_gammas(rows_per_device, devices, num_gpus)
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *residual_inputs, *signals.buffers()
    )
    _check(
        list(outputs),
        num_gpus,
        num_gpus,
        rows_per_device,
        host_acts,
        host_gammas,
        host_residuals,
        regime,
    )


@pytest.mark.parametrize("group_size", [None, 2])
def test_reduce_scatter_rms_norm_empty_batch(group_size: int | None) -> None:
    """A data-parallel replica can legitimately be handed an empty batch.

    Zero rows makes the fused launcher compute a zero grid, which
    `enqueue_function` rejects ("Dim value grid_dim.x must be a positive
    number"). That crashed the serving worker on the very first request under
    TP4xDP2, where one replica owns the single request and the other owns
    nothing. Driven through the graph because that is how production reaches
    it, and at both widths because the grouped op is what makes an empty
    replica reachable at all.
    """
    num_gpus = 4
    if num_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs ({num_gpus}) for empty-batch execution.")

    rows_per_device = [0] * num_gpus
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _rs_rms_norm_graph(signals, rows_per_device, group_size=group_size)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, residual_inputs, _, _, _ = _inputs_and_gammas(
        rows_per_device, devices, group_size or num_gpus
    )
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *residual_inputs, *signals.buffers()
    )

    # The contract is that it does not raise and returns empty shards. Every
    # rank skips together, so no peer is stranded at the start barrier.
    assert len(outputs) == 2 * num_gpus
    for out in outputs:
        assert _from_device_bf16(out).shape[0] == 0


def test_grouped_reduce_scatter_rms_norm_execution() -> None:
    """Grouped op (TP-within-DP): 4 GPUs as 2 groups of 2, per-group shapes.

    The two groups are deliberately sized on OPPOSITE sides of the fuse
    threshold, so one group runs the fused kernel while the other takes the
    two-launch fallback in the same execution. That divergence is what
    `_dispatch_rs_norm`'s per-group invariance comment asserts is safe (the
    groups' `rank_sigs` sets are disjoint); if it were not, this hangs.
    """
    num_gpus = 4
    group_size = 2
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for grouped reduce_scatter_rms_norm."
        )

    # Group 0: 8 rows -> 4 rows/rank -> fused. Group 1: 1024 rows -> 512
    # rows/rank -> two-launch. Different per-group row counts also cover the
    # per-device (not device-0) scatter-dim in the output-shape computation.
    group_rows = [8, 1024]
    assert (
        _shard_rows(group_rows[0], group_size, 0)
        <= FUSE_THRESHOLD_ROWS_PER_RANK
    ), "group 0 must be below the fuse threshold"
    assert (
        _shard_rows(group_rows[1], group_size, 0) > FUSE_THRESHOLD_ROWS_PER_RANK
    ), "group 1 must be above the fuse threshold"

    rows_per_device = [group_rows[i // group_size] for i in range(num_gpus)]
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _rs_rms_norm_graph(signals, rows_per_device, group_size=group_size)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    (
        tensor_inputs,
        gamma_inputs,
        residual_inputs,
        host_acts,
        host_gammas,
        host_residuals,
    ) = _inputs_and_gammas(rows_per_device, devices, group_size or num_gpus)
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *residual_inputs, *signals.buffers()
    )
    _check(
        list(outputs),
        num_gpus,
        group_size,
        rows_per_device,
        host_acts,
        host_gammas,
        host_residuals,
        "grouped-2xTP2",
    )


def test_grouped_reduce_scatter_rms_norm_ragged_execution() -> None:
    """Grouped op with a row count that does not divide the group size.

    Remainder rows go to the low group-local ranks, so the shards are unequal
    within each group -- the case where a global-rank row window silently
    reads the wrong rows.
    """
    num_gpus = 4
    group_size = 2
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for grouped reduce_scatter_rms_norm."
        )

    # 5 rows -> shards 3,2; 3 rows -> shards 2,1.
    rows_per_device = [5, 5, 3, 3]
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _rs_rms_norm_graph(signals, rows_per_device, group_size=group_size)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    (
        tensor_inputs,
        gamma_inputs,
        residual_inputs,
        host_acts,
        host_gammas,
        host_residuals,
    ) = _inputs_and_gammas(rows_per_device, devices, group_size or num_gpus)
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *residual_inputs, *signals.buffers()
    )
    _check(
        list(outputs),
        num_gpus,
        group_size,
        rows_per_device,
        host_acts,
        host_gammas,
        host_residuals,
        "grouped-ragged",
    )
