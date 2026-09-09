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
"""Subgraph correctness tests for the matmul + reduce_scatter fusion.

Constructs graphs whose pattern matches
FuseMatmulReduceScatterPattern so the resulting compiled graph
invokes the fused mo.composite.distributed.matmul_reduce_scatter.sum op on
B200/SM100.

Two patterns under test:
  - bare:     matmul(A_i, weight_i.T) -> reduce_scatter
  - residual: [add(residual, matmul_0), matmul_1, ...] -> reduce_scatter
    (mirrors the asymmetric DeepseekV3/KimiK2.5 pattern where the residual
    is added on a single peer before the cross-device sum).

The graph's M dimension is dynamic, so a single compiled model exercises
many concrete sequence lengths.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Device,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.nn import Module, Signals


def _bf16_tolerances(K: int) -> tuple[float, float]:
    """bf16 has ~3 decimal digits of precision; reductions over many K
    rows compound rounding, so atol scales with K."""
    rtol = 5e-2
    atol = max(1e-2, K * 1e-4)
    return rtol, atol


# Test combos: (N, K, ngpus, M_values, label).
# Includes ragged-prefill M values (rows_per_peer >= BM=128 but
# M % (ngpus*BM) != 0)
COMBOS: list[tuple[int, int, int, list[int], str]] = [
    # Tiny correctness checks.
    (256, 128, 2, [1, 32, 128, 384], "tiny-2gpu"),
    (256, 128, 4, [3, 32, 128, 768], "tiny-4gpu"),
    # Kimi K2.5 attention-output projection shape, inc. decode, prefill, ragged-prefill
    (7168, 1024, 8, [1, 7, 32, 512, 1272, 4096], "kimi-8gpu"),
]


def _bf16_weight(N: int, K: int, gen: torch.Generator) -> torch.Tensor:
    return (
        (torch.randn(N, K, generator=gen, dtype=torch.float32) * 0.1)
        .to(torch.bfloat16)
        .contiguous()
    )


class MatmulReduceScatter(Module):
    """matmul(A_i, weight_i.T) per-device, then reduce_scatter axis=0."""

    def __init__(
        self,
        N: int,
        K: int,
        with_residual: bool,
        num_devices: int,
        residual_peer: int = 0,
    ) -> None:
        super().__init__()
        self.with_residual = with_residual
        self.num_devices = num_devices
        self.residual_peer = residual_peer
        self.weights = [
            Weight(
                name=f"b_{i}",
                dtype=DType.bfloat16,
                shape=(N, K),
                device=DeviceRef.GPU(i),
            )
            for i in range(num_devices)
        ]

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        idx = 0
        a_inputs = [
            cast(TensorValue, args[idx + i]) for i in range(self.num_devices)
        ]
        idx += self.num_devices

        residual: TensorValue | None = None
        if self.with_residual:
            residual = cast(TensorValue, args[idx])
            idx += 1

        signal_buffers = [cast(BufferValue, arg) for arg in args[idx:]]

        per_device_outputs: list[TensorValue] = []
        for i in range(self.num_devices):
            w = self.weights[i].to(DeviceRef.GPU(i))
            per_device_outputs.append(a_inputs[i] @ w.T)

        rs_inputs: list[TensorValue]
        if residual is not None:
            p = self.residual_peer
            rs_inputs = [
                per_device_outputs[i] + residual
                if i == p
                else per_device_outputs[i]
                for i in range(self.num_devices)
            ]
        else:
            rs_inputs = per_device_outputs

        return ops.reducescatter.sum(rs_inputs, signal_buffers, axis=0)


def _compile_model(
    N: int,
    K: int,
    ngpus: int,
    with_residual: bool,
    residual_peer: int = 0,
) -> tuple[Model, list[torch.Tensor], list[Device], Signals]:
    """Compile the matmul+RS graph once with a symbolic M dimension"""
    gen = torch.Generator().manual_seed(0xCAFEF00D)
    b_per_dev = [_bf16_weight(N, K, gen) for _ in range(ngpus)]

    graph_devices = [DeviceRef.GPU(i) for i in range(ngpus)]
    signals = Signals(devices=graph_devices)

    a_input_types = [
        TensorType(DType.bfloat16, shape=["M", K], device=graph_devices[i])
        for i in range(ngpus)
    ]
    extra_input_types: list[TensorType] = []
    if with_residual:
        extra_input_types.append(
            TensorType(
                DType.bfloat16,
                shape=["M", N],
                device=graph_devices[residual_peer],
            )
        )

    model = MatmulReduceScatter(
        N=N,
        K=K,
        with_residual=with_residual,
        num_devices=ngpus,
        residual_peer=residual_peer,
    )

    # No custom_extensions / kernel-library wiring: the fused matmul+RS kernel
    # lives in the closed `internal_kernels` package, which the graph compiler
    # default-loads alongside builtin_kernels. A real model compile (Kimi
    # K2.5, etc.) goes through this same default-load path, so this test
    # compiles exactly like one -- the GC's fusion pass emits
    # mo.composite.distributed.matmul_reduce_scatter.sum and its kernel is already
    # registered.
    graph = Graph(
        f"MatmulReduceScatter_N{N}_K{K}_ngpus{ngpus}_res{with_residual}",
        forward=model,
        input_types=[
            *a_input_types,
            *extra_input_types,
            *signals.input_types(),
        ],
    )

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(ngpus)]
    session = InferenceSession(devices=[host] + devices)

    weights_registry = {f"b_{i}": b_per_dev[i] for i in range(ngpus)}
    compiled = session.load(graph, weights_registry=weights_registry)
    return compiled, b_per_dev, devices, signals


def _expected_output(
    a_per_dev: list[torch.Tensor],
    b_per_dev: list[torch.Tensor],
    residual: torch.Tensor | None,
    ngpus: int,
) -> list[np.ndarray]:
    """Reference: sum over peers of a_i @ b_i.T, plus residual once if
    present, then scatter rows along axis 0 with block-with-remainder
    distribution. b_per_dev[i] is shape (N, K).
    """
    M = a_per_dev[0].shape[0]
    N = b_per_dev[0].shape[0]
    summed = torch.zeros((M, N), dtype=torch.float32)
    for a, b in zip(a_per_dev, b_per_dev, strict=True):
        summed += a.to(torch.float32) @ b.to(torch.float32).T
    if residual is not None:
        summed += residual.to(torch.float32)

    summed_np = summed.numpy()

    q, rem = divmod(M, ngpus)
    expected: list[np.ndarray] = []
    row = 0
    for i in range(ngpus):
        rows_i = q + (1 if i < rem else 0)
        expected.append(summed_np[row : row + rows_i, :].copy())
        row += rows_i
    return expected


def _run_one_M(
    compiled: Model,
    b_per_dev: list[torch.Tensor],
    devices: list[Device],
    N: int,
    K: int,
    ngpus: int,
    with_residual: bool,
    M: int,
    signals: Signals,
    residual_peer: int = 0,
) -> None:
    """Build inputs at concrete M, execute, and verify against numpy."""
    host = CPU()
    gen = torch.Generator().manual_seed(0xC0FFEE ^ M)
    a_per_dev = [
        (torch.randn(M, K, generator=gen, dtype=torch.float32) * 0.1)
        .to(torch.bfloat16)
        .contiguous()
        for _ in range(ngpus)
    ]
    residual = (
        (torch.randn(M, N, generator=gen, dtype=torch.float32) * 0.1)
        .to(torch.bfloat16)
        .contiguous()
        if with_residual
        else None
    )

    input_buffers: list[Buffer] = []
    for i in range(ngpus):
        input_buffers.append(Buffer.from_dlpack(a_per_dev[i]).to(devices[i]))
    if residual is not None:
        input_buffers.append(
            Buffer.from_dlpack(residual).to(devices[residual_peer])
        )

    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*input_buffers, *signals.buffers())
    expected = _expected_output(a_per_dev, b_per_dev, residual, ngpus)

    rtol, atol = _bf16_tolerances(K)
    for i, (out, exp, dev) in enumerate(
        zip(outputs, expected, devices, strict=True)
    ):
        assert isinstance(out, Buffer)
        assert out.device == dev
        result = torch.from_dlpack(out.to(host)).to(torch.float32).numpy()
        if exp.shape[0] == 0:
            assert result.shape[0] == 0, (
                f"device {i}: expected empty rows, got shape {result.shape}"
            )
            continue
        np.testing.assert_allclose(
            result,
            exp,
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"device {i}: matmul+RS"
                f"{' (residual)' if with_residual else ''} "
                f"M={M} N={N} K={K} ngpus={ngpus}"
            ),
        )


def _run_combo(
    N: int,
    K: int,
    ngpus: int,
    M_values: list[int],
    with_residual: bool,
    residual_peer: int = 0,
) -> None:
    if (available := accelerator_count()) < ngpus:
        pytest.skip(f"need {ngpus} GPUs, have {available}")

    compiled, b_per_dev, devices, signals = _compile_model(
        N, K, ngpus, with_residual, residual_peer=residual_peer
    )

    for M in M_values:
        _run_one_M(
            compiled,
            b_per_dev,
            devices,
            N,
            K,
            ngpus,
            with_residual,
            M,
            signals,
            residual_peer=residual_peer,
        )


@pytest.mark.parametrize(
    "N,K,ngpus,M_values,label",
    COMBOS,
    ids=[c[4] for c in COMBOS],
)
def test_matmul_reducescatter_bare(
    N: int, K: int, ngpus: int, M_values: list[int], label: str
) -> None:
    """matmul -> reduce_scatter (no residual)."""
    _run_combo(N, K, ngpus, M_values, with_residual=False)


@pytest.mark.parametrize(
    "N,K,ngpus,M_values,label",
    COMBOS,
    ids=[c[4] for c in COMBOS],
)
def test_matmul_reducescatter_residual(
    N: int, K: int, ngpus: int, M_values: list[int], label: str
) -> None:
    """[add(residual, matmul_0), matmul_1, ...] -> reduce_scatter."""
    _run_combo(N, K, ngpus, M_values, with_residual=True)


# Non-0-peer residual combos: the residual add lands on a device other than
# peer 0. Before plumbing residual_peer through the op this mis-fused (peer 0's
# kernel read device-p memory); the reference sums the residual once regardless
# of peer, so these must match. (N, K, ngpus, M_values, residual_peer, label).
RESIDUAL_PEER_COMBOS: list[tuple[int, int, int, list[int], int, str]] = [
    (256, 128, 2, [1, 32, 128, 384], 1, "tiny-2gpu-peer1"),
    (256, 128, 4, [3, 32, 128, 768], 2, "tiny-4gpu-peer2"),
]


@pytest.mark.parametrize(
    "N,K,ngpus,M_values,residual_peer,label",
    RESIDUAL_PEER_COMBOS,
    ids=[c[5] for c in RESIDUAL_PEER_COMBOS],
)
def test_matmul_reducescatter_residual_nonzero_peer(
    N: int,
    K: int,
    ngpus: int,
    M_values: list[int],
    residual_peer: int,
    label: str,
) -> None:
    """[matmul_0, ..., add(residual, matmul_p), ...] -> reduce_scatter with p != 0."""
    _run_combo(
        N, K, ngpus, M_values, with_residual=True, residual_peer=residual_peer
    )
