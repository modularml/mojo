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

"""Multi-GPU execution test for the fused all-gather + RMSNorm graph op.

The only test that runs the Mojo handler: the kernel-level test re-implements the
grouping itself, so it cannot catch a handler bug (wrong group window, wrong
gamma index, wrong output arity). Exercises both dispatch branches -- the fused
kernel (decode M, below the fuse threshold) and the two-launch fallback (prefill
M, standalone all-gather + rms_norm_gpu through the @__copy_capture closures) --
full-world and grouped, with a DISTINCT gamma per device so a gamma indexed by
the group-local rank instead of the device index fails here.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import ml_dtypes
import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_api,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.nn import Signals

COLS = 6144  # M3 hidden size; the fuse threshold is bytes = rows*COLS*2.
EPS = 1e-6
WEIGHT_OFFSET = 1.0  # M3 Gemma-style: gamma_eff = gamma + 1.0.
# The op fuses at/below this many GATHERED rows (the group's total, not per-rank
# as in reduce-scatter). Mirrors AG_NORM_FUSE_THRESHOLD at H=COLS.
FUSE_THRESHOLD_GATHERED_ROWS = 128


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


def _ag_rms_norm_graph(
    signals: Signals,
    rows_per_device: list[int],
    group_size: int | None,
) -> Graph:
    devices = signals.devices
    num_devices = len(devices)
    with Graph(
        "allgather_rms_norm",
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
        inputs = [v.tensor for v in graph.inputs[:num_devices]]
        gammas = [v.tensor for v in graph.inputs[num_devices : 2 * num_devices]]
        sigs = [v.buffer for v in graph.inputs[2 * num_devices :]]
        normed, residual = ops.allgather_rms_norm(
            inputs=inputs,
            signal_buffers=sigs,
            gammas=gammas,
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
            group_size=group_size,
        )
        graph.output(*normed, *residual)
        return graph


def _host_rmsnorm(
    gathered_f32: np.ndarray, gamma_f32: np.ndarray
) -> np.ndarray:
    """Reference norm of the gathered rows, gamma folded in f32 (mbc=True)."""
    m2 = np.mean(gathered_f32**2, axis=-1, keepdims=True)
    nf = 1.0 / np.sqrt(m2 + EPS)
    return _bf16(gathered_f32 * nf * (gamma_f32 + WEIGHT_OFFSET))


def _inputs_and_gammas(
    rows_per_device: list[int],
    devices: list[Accelerator],
    signed: bool = False,
) -> tuple[list[Buffer], list[Buffer], list[np.ndarray], list[np.ndarray]]:
    """Varied per-device shards + a DISTINCT gamma per device.

    `signed` flips the sign on a period coprime to the value period, so some
    blocks peak on a negative element. Gamma is positive, so without it a
    signed block max is indistinguishable from an absolute one.
    """
    tensor_inputs: list[Buffer] = []
    gamma_inputs: list[Buffer] = []
    host_shards: list[np.ndarray] = []
    host_gammas: list[np.ndarray] = []
    offset = 0
    for i, rows in enumerate(rows_per_device):
        size = rows * COLS
        arr = (((np.arange(size) + offset) % 251) + 1).reshape(rows, COLS)
        arr = arr.astype(np.float32)
        if signed:
            arr = np.where(
                np.arange(size).reshape(rows, COLS) % 3 == 0, -arr, arr
            )
        host_shards.append(_bf16(arr))
        tensor_inputs.append(_to_device_bf16(arr, devices[i]))
        # Distinct per device (the running offset), so gathering a sibling
        # group's shards cannot reproduce this group's tensor.
        offset += size

        # Per-device gamma spaced 0.5 apart, WELL outside the 2e-2 tolerance
        # below: at 0.01 the tolerance swallows the difference and a
        # group-local-vs-device gamma index swap goes undetected.
        gamma = ((np.arange(COLS) % 7) * 0.01) + (i * 0.5)
        gamma = gamma.astype(np.float32)
        host_gammas.append(_bf16(gamma))
        gamma_inputs.append(_to_device_bf16(gamma, devices[i]))
    return tensor_inputs, gamma_inputs, host_shards, host_gammas


def _check(
    outputs: list[Any],
    num_gpus: int,
    group_size: int,
    host_shards: list[np.ndarray],
    host_gammas: list[np.ndarray],
    label: str,
) -> None:
    normed_out = outputs[:num_gpus]
    residual_out = outputs[num_gpus : 2 * num_gpus]

    for group_start in range(0, num_gpus, group_size):
        group = list(range(group_start, group_start + group_size))
        # The all-gather concatenates ONLY this group's shards, in group-rank
        # order; a handler that slices the wrong window gathers a different set,
        # and one that gathers in global-device order permutes the rows.
        gathered = np.concatenate([host_shards[d] for d in group], axis=0)

        for local_rank, dev_idx in enumerate(group):
            res = _from_device_bf16(cast(Buffer, residual_out[dev_idx]))
            assert res.shape == gathered.shape, (
                f"{label}: residual shape on GPU {dev_idx}: {res.shape} != "
                f"{gathered.shape}"
            )
            # A gathered row is a verbatim copy, so this is a bit-identity gate.
            assert np.array_equal(res, gathered), (
                f"{label}: residual mismatch on GPU {dev_idx} "
                f"(local_rank={local_rank})"
            )

            nrm = _from_device_bf16(cast(Buffer, normed_out[dev_idx]))
            ref = _host_rmsnorm(gathered, host_gammas[dev_idx])
            max_abs = float(np.max(np.abs(nrm - ref)))
            # Loose bf16 reduction-order tolerance; the tight ULP gate and
            # bit-identity vs `rms_norm_gpu` live in
            # test/gpu/comm/test_allgather_rmsnorm.mojo. A wrong gamma index,
            # eps, divisor or group window is far larger than this.
            assert np.allclose(nrm, ref, rtol=2e-2, atol=2e-2), (
                f"{label}: normed mismatch on GPU {dev_idx} "
                f"(local_rank={local_rank}): max_abs={max_abs}"
            )

    # Groups must be SEPARATED, not merely self-consistent: a handler that
    # gathered the whole world, or a sibling group's window, still satisfies
    # every within-group gate above if both groups end up holding the same data.
    for a, b in itertools.combinations(range(0, num_gpus, group_size), 2):
        ra = _from_device_bf16(cast(Buffer, residual_out[a]))
        rb = _from_device_bf16(cast(Buffer, residual_out[b]))
        if ra.shape == rb.shape:
            assert not np.array_equal(ra, rb), (
                f"{label}: the groups led by GPU {a} and GPU {b} gathered "
                "identical data, so the per-group gates above cannot "
                "distinguish a cross-group read"
            )


@pytest.mark.parametrize(
    "num_gpus, shard_rows, regime",
    [
        (4, 2, "fused"),  # 8 gathered rows -> below threshold -> fused kernel
        (4, 128, "two_launch"),  # 512 gathered rows -> above -> two-launch
    ],
)
def test_allgather_rms_norm_execution(
    num_gpus: int, shard_rows: int, regime: str
) -> None:
    """Full-world op: both dispatch branches, per-device gamma."""
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for {regime} allgather_rms_norm."
        )

    rows_per_device = [shard_rows] * num_gpus
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _ag_rms_norm_graph(signals, rows_per_device, group_size=None)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, host_shards, host_gammas = _inputs_and_gammas(
        rows_per_device, devices
    )
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *signals.buffers()
    )
    _check(list(outputs), num_gpus, num_gpus, host_shards, host_gammas, regime)


@pytest.mark.parametrize("group_size", [None, 2])
def test_allgather_rms_norm_empty_batch(group_size: int | None) -> None:
    """A data-parallel replica can legitimately be handed an empty batch.

    Zero gathered rows makes the fused launcher compute a zero grid, which
    `enqueue_function` rejects ("Dim value grid_dim.x must be a positive
    number"). The reduce-scatter sibling crashed the serving worker this way on
    the very first request under TP4xDP2, where one replica owns the single
    request and the other owns nothing; this op has the same launcher shape.
    """
    num_gpus = 4
    if num_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs ({num_gpus}) for empty-batch execution.")

    rows_per_device = [0] * num_gpus
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _ag_rms_norm_graph(signals, rows_per_device, group_size=group_size)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, _, _ = _inputs_and_gammas(
        rows_per_device, devices
    )
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *signals.buffers()
    )

    # The contract is that it does not raise and returns empty tensors. Every
    # rank skips together, so no peer is stranded at the start barrier.
    assert len(outputs) == 2 * num_gpus
    for out in outputs:
        assert _from_device_bf16(cast(Buffer, out)).shape[0] == 0


def test_grouped_allgather_rms_norm_execution() -> None:
    """Grouped op (TP-within-DP): 4 GPUs as 2 groups of 2, per-group shapes.

    The two groups are deliberately sized on OPPOSITE sides of the fuse
    threshold, so one group runs the fused kernel while the other takes the
    two-launch fallback in the same execution. That divergence is what
    `_dispatch_ag_norm`'s per-group invariance comment asserts is safe (the
    groups' `rank_sigs` sets are disjoint); if it were not, this hangs.
    """
    num_gpus = 4
    group_size = 2
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for grouped allgather_rms_norm."
        )

    # Group 0 gathers 8 rows -> fused. Group 1 gathers 256 -> two-launch.
    # Different per-group totals also cover the per-device (not device-0)
    # gathered dim in the output-shape computation.
    shard_rows = [4, 128]
    assert shard_rows[0] * group_size <= FUSE_THRESHOLD_GATHERED_ROWS, (
        "group 0 must be below the fuse threshold"
    )
    assert shard_rows[1] * group_size > FUSE_THRESHOLD_GATHERED_ROWS, (
        "group 1 must be above the fuse threshold"
    )

    rows_per_device = [shard_rows[i // group_size] for i in range(num_gpus)]
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _ag_rms_norm_graph(signals, rows_per_device, group_size=group_size)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, host_shards, host_gammas = _inputs_and_gammas(
        rows_per_device, devices
    )
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *signals.buffers()
    )
    _check(
        list(outputs),
        num_gpus,
        group_size,
        host_shards,
        host_gammas,
        "grouped-2xTP2",
    )


def test_grouped_allgather_rms_norm_separation() -> None:
    """Grouped op where both groups gather the SAME shape but different data.

    This is the case that can actually catch a cross-group read: with matching
    shapes the separation gate in `_check` fires, so a handler that gathered the
    whole world -- or a sibling group's window -- produces two identical outputs
    and fails, where the shape-differing case above would still pass.
    """
    num_gpus = 4
    group_size = 2
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs ({num_gpus}) for grouped allgather_rms_norm."
        )

    rows_per_device = [4] * num_gpus  # both groups gather 8 rows
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _ag_rms_norm_graph(signals, rows_per_device, group_size=group_size)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, host_shards, host_gammas = _inputs_and_gammas(
        rows_per_device, devices
    )
    outputs = compiled.execute(
        *tensor_inputs, *gamma_inputs, *signals.buffers()
    )
    _check(
        list(outputs),
        num_gpus,
        group_size,
        host_shards,
        host_gammas,
        "grouped-separation",
    )


def _mxfp8_graph(
    signals: Signals, rows_per_device: list[int], group_size: int | None
) -> Graph:
    devices = signals.devices
    num_devices = len(devices)
    with Graph(
        "allgather_rms_norm_quant_mxfp8",
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
        inputs = [v.tensor for v in graph.inputs[:num_devices]]
        gammas = [v.tensor for v in graph.inputs[num_devices : 2 * num_devices]]
        sigs = [v.buffer for v in graph.inputs[2 * num_devices :]]
        normed, quant, scales, residual = ops.allgather_rms_norm_quant_mxfp8(
            inputs=inputs,
            signal_buffers=sigs,
            gammas=gammas,
            epsilon=EPS,
            weight_offset=WEIGHT_OFFSET,
            group_size=group_size,
        )
        # `normed, residual` first, in that order, so `_check` applies verbatim
        # to the output prefix.
        graph.output(*normed, *residual, *quant, *scales)
        return graph


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason=(
        "The MXFP8 quantize rides the CDNA4 path and emits its rank-2 scale "
        "layout; the target's own bazel rule is arch-agnostic, so skip here."
    ),
)
@pytest.mark.parametrize(
    "num_gpus, shard_rows, regime",
    [
        (4, 2, "fused"),
        (4, 128, "two_launch"),
    ],
)
def test_allgather_rms_norm_quant_mxfp8_execution(
    num_gpus: int, shard_rows: int, regime: str
) -> None:
    """The MXFP8 outputs must match a host quantize of the op's own norm.

    Two layers: the quantize oracle re-derives each E8M0 scale from the returned
    `normed`, so it is self-consistent by construction -- a wrong gamma index,
    gather window, epsilon or group would still pass it. `_check` is the
    independent half, rebuilding normed and residual from the INPUTS.
    """
    if num_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs ({num_gpus}) for {regime} mxfp8 AG norm.")

    rows_per_device = [shard_rows] * num_gpus
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _mxfp8_graph(signals, rows_per_device, group_size=None)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    tensor_inputs, gamma_inputs, host_shards, host_gammas = _inputs_and_gammas(
        rows_per_device, devices, signed=True
    )
    outputs = list(
        compiled.execute(*tensor_inputs, *gamma_inputs, *signals.buffers())
    )

    # The independent half. Ungrouped, so group_size == num_gpus.
    _check(
        outputs,
        num_gpus,
        num_gpus,
        host_shards,
        host_gammas,
        f"mxfp8 {regime}",
    )

    nd = num_gpus
    for dev in range(nd):
        normed = _from_device_bf16(cast(Buffer, outputs[dev]))
        quant = (
            cast(Buffer, outputs[2 * nd + dev])
            .copy(device=CPU())
            .view(DType.uint8)
            .to_numpy()
        )
        scales = (
            cast(Buffer, outputs[3 * nd + dev])
            .copy(device=CPU())
            .view(DType.uint8)
            .to_numpy()
        )
        assert quant.shape == normed.shape
        assert scales.shape == (normed.shape[0], COLS // 32)

        # Scale = ceil(log2(block_max / 448)), the smallest power of two that
        # keeps a block inside E4M3's +/-448. NOT ml_dtypes' float8_e8m0fnu
        # cast: it truncates, one exponent low, letting the max reach 896.
        blocks = normed.reshape(normed.shape[0], COLS // 32, 32)
        block_max = np.abs(blocks).max(axis=-1).astype(np.float32)
        assert (block_max > 0).all(), (
            "empty block: oracle assumes block_max > 0"
        )
        # frexp returns m in [0.5, 1), so ceil(log2(x)) is `e` except at the
        # exact power of two m == 0.5, where it is `e - 1`.
        mantissa, exponent = np.frexp(block_max / np.float32(448.0))
        want_scales = (
            np.where(mantissa == 0.5, exponent - 1, exponent) + 127
        ).astype(np.uint8)
        assert np.array_equal(scales, want_scales), (
            f"device {dev}: fused E8M0 scales disagree with a host quantize of "
            "the op's own normed output"
        )

        # Byte-exact on the payload too, against a host round-to-nearest-even
        # E4M3 encode -- non-vacuity alone would accept the wrong tensor.
        descaled = blocks / np.exp2(scales.astype(np.float32) - 127)[:, :, None]
        want_quant = (
            descaled.astype(ml_dtypes.float8_e4m3fn)
            .view(np.uint8)
            .reshape(quant.shape)
        )
        assert np.array_equal(quant, want_quant), (
            f"device {dev}: fused MXFP8 payload disagrees with a host quantize "
            "of the op's own normed output"
        )

        # Non-vacuity: an all-zero quant buffer would pass a shape check.
        assert np.count_nonzero(quant) > quant.size // 2
        assert len(np.unique(scales)) >= 4
