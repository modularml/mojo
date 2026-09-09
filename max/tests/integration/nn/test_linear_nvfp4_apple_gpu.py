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
"""Graph-level numeric test for the Apple M5 NVFP4 (W4A16) ``Linear`` path.

Builds a single NVFP4 :class:`~max.nn.Linear` (a FLUX.2 transformer Linear
shape) and runs it through the graph on an Apple (Metal) GPU, then compares the
output to a reference graph branch that performs a plain bf16 dense matmul of
the *materialized* dequantized weight. Both outputs go through the same MAX
bf16 MMA, so they must agree to bf16-MMA tolerance.

This exercises the full Apple weight-only lowering chain:
``Linear.__call__ -> linear() -> quantized_matmul() -> _matmul_float4() ->
(Apple branch) _apple_weight_only_block_scaled_matmul() ->
mo.matmul.weight.only.block.scaled.apple -> enqueue_apple_fp4_matmul`` plus the
graph-level ``weight_scale_2`` fold.

Unlike the NVIDIA SM100 path, the activation stays in bf16 (it is *not*
dynamically quantized to FP4) and the weight block scales are plain rank-2
``[N, K // 16]`` (no rank-5 TCGEN05 interleave). The reference therefore
materializes the dequant exactly as the Apple kernel does:
``E2M1[nibble] * |block_scale| * weight_scale_2``.
"""

from __future__ import annotations

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
from max.graph import DeviceRef, Graph, Shape, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn import Linear
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

# E2M1 4-bit code -> float value (sign + magnitude baked in), matching
# `linalg/fp4_utils.mojo` E2M1_TO_FLOAT32. Index is the 4-bit nibble.
_E2M1_TO_FLOAT = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)

_SF_VECTOR_SIZE = 16


def _skip_if_not_apple() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for the Apple NVFP4 Linear test")
    if accelerator_api() != "metal":
        pytest.skip("Apple W4A16 NVFP4 Linear path requires a Metal GPU")


def _make_nvfp4_config() -> QuantConfig:
    """NVFP4 block-scaled config (block 16 on K), matching FLUX.2-NVFP4."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.NVFP4,
        scales_pre_interleaved=False,
    )


def _pack_fp4_weight(nibbles: np.ndarray) -> np.ndarray:
    """Pack ``[N, K]`` 4-bit codes into ``uint8 [N, K // 2]`` (low nibble first).

    Element ``2*j`` -> ``byte & 0xF``, element ``2*j+1`` -> ``byte >> 4``
    (the Apple kernel's lo-nibble-first convention).
    """
    lo = nibbles[:, 0::2].astype(np.uint8)
    hi = nibbles[:, 1::2].astype(np.uint8)
    return (lo | (hi << np.uint8(4))).astype(np.uint8)


def _materialize_dequant_weight(
    nibbles: np.ndarray,
    scales_fp32: np.ndarray,
    weight_scale_2: float,
) -> np.ndarray:
    """Dense dequantized weight ``[N, K]`` (fp32): ``E2M1 * |scale| * ws2``."""
    _, k = nibbles.shape
    vals = _E2M1_TO_FLOAT[nibbles]  # [N, K]
    block_scale = np.abs(scales_fp32)[:, : (k // _SF_VECTOR_SIZE)]
    block_scale_full = np.repeat(block_scale, _SF_VECTOR_SIZE, axis=1)[:, :k]
    return (vals * block_scale_full * np.float32(weight_scale_2)).astype(
        np.float32
    )


def _fp32_to_fp8_bytes(
    values_fp32: np.ndarray, device: Accelerator, device_ref: DeviceRef
) -> np.ndarray:
    """Round positive fp32 values to float8_e4m3fn and return the raw bytes.

    numpy has no fp8 dtype, so round-trip the values through a tiny cast graph
    on the device. The test scales are exactly fp8-representable, so this is a
    no-op rounding; it exists to produce the canonical fp8 byte encoding the
    graph const expects. (The cast must run on the accelerator -- fp8 is not a
    supported CPU dtype.)
    """
    flat = values_fp32.reshape(-1).astype(np.float32)
    sess = InferenceSession(devices=[device])
    with Graph(
        "fp32_to_fp8",
        input_types=[
            TensorType(DType.float32, (flat.shape[0],), device=device_ref)
        ],
    ) as g:
        (v,) = g.inputs
        assert isinstance(v, TensorValue)
        g.output(ops.cast(v, DType.float8_e4m3fn))
    out = sess.load(g).execute(Buffer.from_numpy(flat).to(device))[0]
    assert isinstance(out, Buffer)
    return (
        np.from_dlpack(out.to(CPU()).view(DType.uint8))
        .copy()
        .reshape(values_fp32.shape)
    )


def test_linear_nvfp4_apple() -> None:
    """Numeric check: Apple NVFP4 Linear == bf16 dense matmul of dequant weight."""
    _skip_if_not_apple()

    rng = np.random.default_rng(0)
    # A FLUX.2 transformer block dim: N=out, K=in. K must be a multiple of 16.
    M, N, K = 8, 256, 512

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)
    quant_config = _make_nvfp4_config()

    # Random 4-bit codes (full 0..15 range) + fp8-exact positive block scales.
    nibbles = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    packed = _pack_fp4_weight(nibbles)  # [N, K//2] uint8
    scale_k = K // _SF_VECTOR_SIZE
    # Scales in {0.5, 1.0, 1.5, 2.0} -> exactly fp8-e4m3 representable.
    scales_fp32 = rng.integers(1, 5, size=(N, scale_k)).astype(
        np.float32
    ) * np.float32(0.5)
    weight_scale_2 = np.float32(0.0125)
    input_scale = np.float32(
        1.0
    )  # cancels on the Apple path; value irrelevant.

    scales_fp8_bytes = _fp32_to_fp8_bytes(scales_fp32, device, device_ref)
    scales_fp8_buf = Buffer.from_numpy(scales_fp8_bytes).view(
        DType.float8_e4m3fn, (N, scale_k)
    )
    weight_scale_wd = WeightData(
        scales_fp8_buf, "weight_scale", DType.float8_e4m3fn, Shape((N, scale_k))
    )

    layer = Linear(
        in_dim=K,
        out_dim=N,
        dtype=DType.uint8,
        device=device_ref,
        has_bias=False,
        quant_config=quant_config,
    )
    layer.load_state_dict(
        {
            "weight": packed,  # uint8 [N, K//2]
            "weight_scale": weight_scale_wd,  # fp8 [N, K//16]
            "weight_scale_2": np.array(weight_scale_2, dtype=np.float32),
            "input_scale": np.array(input_scale, dtype=np.float32),
        },
        weight_alignment=1,
    )

    # Dense dequantized weight (fp32) for the reference matmul.
    w_dense = _materialize_dequant_weight(
        nibbles, scales_fp32, float(weight_scale_2)
    )  # [N, K]

    x_fp32 = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)

    session = InferenceSession(devices=[device])
    with Graph(
        "Linear_NVFP4_Apple_Test",
        input_types=[TensorType(DType.float32, (M, K), device=device_ref)],
    ) as graph:
        (x_in,) = graph.inputs
        assert isinstance(x_in, TensorValue)
        # Cast activation to bf16 in-graph (matches a real bf16 activation).
        x_bf16 = ops.cast(x_in, DType.bfloat16)

        # Path under test: the NVFP4 Linear (Apple W4A16 lowering).
        out_test = layer(x_bf16)

        # Reference: plain bf16 dense matmul of the materialized dequant weight.
        w_const = ops.constant(w_dense, DType.float32, device=device_ref)
        w_bf16 = ops.cast(w_const, DType.bfloat16)
        out_ref = ops.matmul(x_bf16, ops.transpose(w_bf16, 0, 1))

        graph.output(
            ops.cast(out_test, DType.float32),
            ops.cast(out_ref, DType.float32),
        )

    compiled = session.load(graph, weights_registry=layer.state_dict())

    x_dev = Buffer.from_numpy(x_fp32).to(device)
    got_buf, ref_buf = compiled.execute(x_dev)

    assert isinstance(got_buf, Buffer)
    assert isinstance(ref_buf, Buffer)
    got = np.from_dlpack(got_buf.to(CPU())).astype(np.float32)
    ref = np.from_dlpack(ref_buf.to(CPU())).astype(np.float32)

    assert got.shape == (M, N)
    assert np.isfinite(got).all(), "Apple NVFP4 Linear output has NaN/Inf"

    # Both go through the MAX bf16 MMA; the Apple in-register dequant is
    # bit-exact vs a materialized dequant (proven at the kernel level), so the
    # two outputs should agree to bf16-MMA tolerance.
    atol = 1e-2 + 1.6e-2 * np.abs(ref)
    max_err = float(np.max(np.abs(got - ref)))
    assert np.all(np.abs(got - ref) <= atol), (
        f"Apple NVFP4 Linear mismatch vs bf16 dequant reference: "
        f"max_err={max_err}"
    )
