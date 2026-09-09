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
"""Graph-level test for the Apple M5 int8 W8A8 fused matmul op.

Builds a one-op graph invoking ``mo.matmul.int8.w8a8.apple`` (via the
``_apple_int8_w8a8_matmul`` wrapper -- online activation quant + int8
widening-MMA GEMM + dequant, all internal to the op) and compares it to a bf16
reference matmul of the *dequantized* int8 weight, at a FLUX.2-Klein transformer
Linear shape. This exercises the full lowering:
``_apple_int8_w8a8_matmul -> ops.custom("mo.matmul.int8.w8a8.apple"[.bias]) ->
enqueue_apple_int8_quantize_activation + enqueue_apple_int8_matmul`` -- the
graph-op path the standalone kernel test cannot cover.

Both outputs go through the same bf16 MMA (the int8 GEMM dequants to bf16), so
they agree to int8-quant tolerance (cosine >= 0.999): the only difference is the
int8-vs-bf16 quantization of the *weight* (the reference uses the same absmax/127
quant, dequantized), plus the online int8 quant of the activation. Covers both
the no-bias op and the ``.bias`` sibling op.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_api,
    accelerator_architecture_name,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kernels import _apple_int8_w8a8_matmul
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.quant_ops import quantized_matmul


def _int8_w8a8_quant_config() -> QuantConfig:
    """A minimal symmetric int8 W8A8 ``QuantConfig`` (matches FLUX.2's)."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,
            dtype=DType.float32,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.INT8_W8A8,
    )


def _skip_if_not_apple() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for the Apple int8 W8A8 test")
    if accelerator_api() != "metal":
        pytest.skip("Apple int8 W8A8 path requires a Metal GPU")
    # `enqueue_apple_int8_matmul` requires Apple M5 (compute_capability == 5).
    # Metal reports the compute capability as the leading integer of the
    # architecture name (e.g. "2" on M2, "5-metal4" on M5).
    generation = int(accelerator_architecture_name().split("-", 1)[0])
    if generation != 5:
        pytest.skip(
            "Apple int8 W8A8 path requires Apple M5 "
            f"(compute_capability == 5); got compute_capability={generation}"
        )


def _quant_rows_absmax(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric absmax/127 int8 quant -> (int8 [R, C], fp32 scale [R]).

    Matches ``int8_matmul.mojo`` / ``_quantize_a_block``: ``scale = absmax/127``,
    ``q = round(x * 127 / absmax)``.
    """
    amax = np.abs(x).max(axis=1)
    scale = np.where(amax != 0, amax / 127.0, 0.0).astype(np.float32)
    mult = np.where(amax != 0, 127.0 / amax, 0.0).astype(np.float32)
    q = np.rint(x * mult[:, None]).astype(np.int8)
    return q, scale


@pytest.mark.parametrize("with_bias", [False, True])
def test_int8_w8a8_apple(with_bias: bool) -> None:
    """Numeric check: the fused int8 W8A8 op == bf16 matmul of the dequant weight."""
    _skip_if_not_apple()

    rng = np.random.default_rng(0)
    # A FLUX.2-Klein transformer Linear shape (attn projection).
    M, N, K = 4096, 3072, 3072

    device = Accelerator(0)
    dref = DeviceRef(device.label, device.id)

    a_f = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    w_f = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    bq, bsc = _quant_rows_absmax(
        w_f
    )  # int8 weight [N, K] + per-channel scale [N]
    w_dq = (bq.astype(np.float32) * bsc[:, None]).astype(np.float32)
    bias_f = (
        (rng.standard_normal((N,)) * 0.05).astype(np.float32)
        if with_bias
        else None
    )

    session = InferenceSession(devices=[device])
    with Graph(
        "int8_w8a8_apple",
        input_types=[TensorType(DType.float32, (M, K), device=dref)],
    ) as graph:
        (a_in,) = graph.inputs
        assert isinstance(a_in, TensorValue)
        a_bf16 = ops.cast(a_in, DType.bfloat16)

        w_i8 = ops.constant(bq, DType.int8, device=dref)
        # Pass the rowwise [N, 1] weight scale (the wrapper squeezes it).
        b_scale = ops.constant(bsc.reshape(N, 1), DType.float32, device=dref)
        bias_v = (
            ops.constant(bias_f, DType.float32, device=dref)
            if bias_f is not None
            else None
        )
        out_test = _apple_int8_w8a8_matmul(a_bf16, w_i8, b_scale, bias_v)

        # Reference: bf16 dense matmul of the dequantized int8 weight (+bias).
        w_ref = ops.cast(
            ops.constant(w_dq, DType.float32, device=dref), DType.bfloat16
        )
        out_ref = ops.matmul(a_bf16, ops.transpose(w_ref, 0, 1))
        if with_bias:
            assert bias_f is not None
            out_ref = out_ref + ops.cast(
                ops.constant(bias_f, DType.float32, device=dref),
                DType.bfloat16,
            )

        graph.output(
            ops.cast(out_test, DType.float32), ops.cast(out_ref, DType.float32)
        )

    compiled = session.load(graph)
    a_dev = Buffer.from_numpy(a_f).to(device)
    got_buf, ref_buf = compiled.execute(a_dev)
    assert isinstance(got_buf, Buffer)
    assert isinstance(ref_buf, Buffer)
    got = np.from_dlpack(got_buf.to(CPU())).astype(np.float32)
    ref = np.from_dlpack(ref_buf.to(CPU())).astype(np.float32)

    assert got.shape == (M, N)
    assert np.isfinite(got).all(), "int8 W8A8 op output has NaN/Inf"

    gf, rf = got.reshape(-1), ref.reshape(-1)
    cos = float(gf @ rf / (np.linalg.norm(gf) * np.linalg.norm(rf) + 1e-12))
    assert cos >= 0.999, (
        f"int8 W8A8 op vs bf16 dequant reference cosine {cos} < 0.999 "
        f"(with_bias={with_bias})"
    )


def test_int8_w8a8_fused_bias_equals_separate_add() -> None:
    """Fused bias (``.bias`` op) == bias-less matmul + separate add.

    Drives the ``quantized_matmul`` int8 path both ways in one graph: with a
    bias handed in (the fused ``.bias`` epilogue -- the wiring Fix 5 completes
    through ``Linear.__call__`` -> ``linear`` -> ``quantized_matmul`` ->
    ``_matmul_int8``) and without a bias followed by a separate ``+ bias`` add.
    The fused op adds ``bias[col]`` after dequant on the same int32 accumulator,
    so the two must agree to bf16 round-off (both add the same bf16 bias to the
    same dequant result). This guards the plumbing the standalone op test and
    the FLUX.2-klein path (``bias=False``) never exercise.
    """
    _skip_if_not_apple()

    rng = np.random.default_rng(1)
    M, N, K = 512, 768, 1024

    device = Accelerator(0)
    dref = DeviceRef(device.label, device.id)

    a_f = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    w_f = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    bq, bsc = _quant_rows_absmax(w_f)  # int8 weight [N, K] + scale [N]
    bias_f = (rng.standard_normal((N,)) * 0.05).astype(np.float32)

    qc = _int8_w8a8_quant_config()

    session = InferenceSession(devices=[device])
    with Graph(
        "int8_w8a8_fused_bias",
        input_types=[TensorType(DType.float32, (M, K), device=dref)],
    ) as graph:
        (a_in,) = graph.inputs
        assert isinstance(a_in, TensorValue)
        a_bf16 = ops.cast(a_in, DType.bfloat16)

        w_i8 = ops.constant(bq, DType.int8, device=dref)
        w_scale = ops.constant(bsc.reshape(N, 1), DType.float32, device=dref)
        bias_v = ops.cast(
            ops.constant(bias_f, DType.float32, device=dref), DType.bfloat16
        )

        # Fused: bias handed to quantized_matmul -> the .bias op fires.
        out_fused = quantized_matmul(
            a_bf16, w_i8, w_scale, None, qc, bias=bias_v
        )
        # Unfused: bias-less quantized_matmul + a separate add (the old path).
        out_unfused = quantized_matmul(a_bf16, w_i8, w_scale, None, qc)
        out_unfused = out_unfused + bias_v

        graph.output(
            ops.cast(out_fused, DType.float32),
            ops.cast(out_unfused, DType.float32),
        )

    compiled = session.load(graph)
    a_dev = Buffer.from_numpy(a_f).to(device)
    fused_buf, unfused_buf = compiled.execute(a_dev)
    assert isinstance(fused_buf, Buffer)
    assert isinstance(unfused_buf, Buffer)
    fused = np.from_dlpack(fused_buf.to(CPU())).astype(np.float32)
    unfused = np.from_dlpack(unfused_buf.to(CPU())).astype(np.float32)

    assert fused.shape == (M, N)
    assert np.isfinite(fused).all(), "fused-bias int8 output has NaN/Inf"
    # Both add the same bf16 bias to the same dequant result: bit-identical.
    np.testing.assert_array_equal(
        fused,
        unfused,
        err_msg="fused .bias epilogue != bias-less matmul + separate add",
    )


def _rtn_quantize_int8_rows(
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row RTN int8 quant matching ``weight_adapters._rtn_quantize_int8``.

    ``scale = absmax/127`` (1.0 where a row is all-zero), ``q =
    clip(round(w/scale), -127, 127)``. Returns ``(int8 [N, K], scale [N, 1])``.
    """
    absmax = np.abs(w).max(axis=1, keepdims=True)
    scale = np.where(absmax != 0.0, absmax / 127.0, np.float32(1.0)).astype(
        np.float32
    )
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def test_int8_w8a8_rtn_quality_vs_bf16() -> None:
    """RTN quantization *quality*: int8 W8A8 vs the ORIGINAL bf16 weight.

    The other test compares against a matmul of the *dequantized* int8 weight,
    so it only exercises the kernel's dequant -- it can't see a bad RTN
    quantization (a broken scale would dequant back to the same wrong number on
    both sides). Here the reference is a bf16 matmul of the *original* bf16
    weight, so the cosine gap is exactly the round-to-nearest int8 error the
    FLUX.2-klein weights actually incur at load. Measured ~0.99986-0.99991
    across seeds and the attn/MLP Linear shapes; 0.999 is the CI floor (a real
    regression -- e.g. per-tensor instead of per-channel scale -- drops it far
    below).
    """
    _skip_if_not_apple()

    rng = np.random.default_rng(0)
    # FLUX.2-klein MLP Linear shape (the widest -- lowest quality of the set).
    M, N, K = 4096, 12288, 3072

    device = Accelerator(0)
    dref = DeviceRef(device.label, device.id)

    a_f = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    w_f = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    bq, bsc = _rtn_quantize_int8_rows(w_f)  # int8 [N, K] + scale [N, 1]

    session = InferenceSession(devices=[device])
    with Graph(
        "int8_w8a8_rtn_quality",
        input_types=[TensorType(DType.float32, (M, K), device=dref)],
    ) as graph:
        (a_in,) = graph.inputs
        assert isinstance(a_in, TensorValue)
        a_bf16 = ops.cast(a_in, DType.bfloat16)

        w_i8 = ops.constant(bq, DType.int8, device=dref)
        w_scale = ops.constant(bsc, DType.float32, device=dref)
        out_test = _apple_int8_w8a8_matmul(a_bf16, w_i8, w_scale)

        # Reference: the ORIGINAL bf16 weight (not dequantized) -- so the gap
        # is the RTN quantization error, not just dequant round-off.
        w_ref = ops.cast(
            ops.constant(w_f, DType.float32, device=dref), DType.bfloat16
        )
        out_ref = ops.matmul(a_bf16, ops.transpose(w_ref, 0, 1))

        graph.output(
            ops.cast(out_test, DType.float32), ops.cast(out_ref, DType.float32)
        )

    compiled = session.load(graph)
    a_dev = Buffer.from_numpy(a_f).to(device)
    got_buf, ref_buf = compiled.execute(a_dev)
    assert isinstance(got_buf, Buffer)
    assert isinstance(ref_buf, Buffer)
    got = np.from_dlpack(got_buf.to(CPU())).astype(np.float32).reshape(-1)
    ref = np.from_dlpack(ref_buf.to(CPU())).astype(np.float32).reshape(-1)

    assert np.isfinite(got).all(), "int8 W8A8 output has NaN/Inf"
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
    assert cos >= 0.999, (
        f"int8 W8A8 RTN quantization-quality cosine {cos} < 0.999 vs the "
        "original bf16 weight -- RTN quant quality regressed"
    )
