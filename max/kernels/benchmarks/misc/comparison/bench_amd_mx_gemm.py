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

# AMD MXFP4/MXFP6/MXFP8 dense GEMM benchmark comparing MAX against aiter.
#
# Computes Y = A @ B^T where A is [M, K] and B is [N, K], both quantized to
# the same MX block-scaled format (one E8M0 scale per 32-element K block):
# MXFP4 packs 2 E2M1 elements per uint8 along K; MXFP6 packs 4 six-bit codes
# per 3 bytes; MXFP8 stores one E4M3 element per byte. Output is BF16.
#   * MAX path  -> `dynamic_block_scaled_matmul_amd` for MXFP4/MXFP8 (one
#                  entry point for both; it picks the packing from the input
#                  dtype) -> custom op `mo.matmul.dynamic.block.scaled.amd`
#                  -> the CDNA4 kernel `block_scaled_matmul_amd`.
#                  MXFP6 goes through `dynamic_block_scaled_matmul_mxfp6`
#                  -> `mo.matmul.dynamic.block.scaled.mxfp6` -> the CDNA4
#                  kernel `mxfp6_block_scaled_matmul_amd`: a separate op
#                  because both FP6 encodings put 24 bytes in a lane, so the
#                  byte count cannot choose between them.
#   * aiter path -> `aiter.ops.triton.gemm.basic.gemm_afp4wfp4` (the Triton
#                  `_gemm_afp4wfp4_kernel`), MXFP4 only. aiter has no native
#                  MX-format FP6 or FP8 kernel in this version, so those run
#                  MAX-only.
#
# Timing mirrors bench_amd_mla.py's chained-call strategy: ncopies distinct
# rotating weight buffers are chained into ONE device-graph (MAX) / CUDA-graph
# (aiter); a single replay sweeps all ncopies GEMMs back-to-back, free of
# per-replay launch gaps. Reported latency is whole-graph time / ncopies
# (per-op).
#
# On L2 residency: the copy count is capped (chained device-graph ops compile
# superlinearly -- 48 of them takes >10 minutes), so weights of a few MB cannot
# be pushed out of the 256 MB L2 here. A production model's tp=4 shapes are
# in that range, so they measure partly L2-warm. What this harness does
# guarantee is that the three formats are compared at the SAME working-set
# bytes rather than the same copy count (see `_equalized_working_set`) --
# otherwise the cap binds at different L2 pressure per format and flatters
# the narrower ones. Absolute cold-HBM MXFP6 numbers need a harness that
# reads through a cache-busting buffer rather than chained ops.
#
# Run via kbench: kbench bench_amd_mx_gemm.yaml

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable
from typing import Any

# Configure aiter JIT environment before any aiter imports. When run via
# kbench (file: mode), the Bazel env vars aren't applied, so set them as
# fallbacks.
if "AITER_JIT_DIR" not in os.environ:
    _ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    os.environ["AITER_JIT_DIR"] = os.path.join(
        _ws, ".derived", "aiter_jit_cache"
    )
if "/usr/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = (
        "/usr/bin:/bin:/usr/local/bin:/opt/rocm/bin:"
        + os.environ.get("PATH", "")
    )

import torch
from bencher_utils import Bench, ThroughputMeasure
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    dynamic_block_scaled_matmul_amd,
    dynamic_block_scaled_matmul_mxfp6,
)
from max.pipelines.weights.block_scaled_preshuffle import (
    shuffle_block_scaled_b_dense_arrays,
)

# aiter MXFP4 Triton GEMM (`_gemm_afp4wfp4_kernel`). JIT/compile on first use.
_aiter_gemm: Callable[..., torch.Tensor] | None
try:
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        gemm_afp4wfp4 as _aiter_gemm,
    )
except (ImportError, Exception) as e:
    print(f"Warning: aiter gemm_afp4wfp4 not available: {e}")
    _aiter_gemm = None

# MI355X L2 cache size (256 MB).
_L2_CACHE_SIZE_BYTES = int(256e6)
# Number of rotating weight copies chained into one graph. The total footprint
# (ncopies * B) must exceed the 256 MB L2 so each chained GEMM reads cold HBM.
# Overridable via `--ncopies`; 0 = auto.
_NCOPIES = 16
# Cap on the auto-chosen copy count. Raising it is not an option: chained MAX
# device-graph ops compile superlinearly, and 48 ops takes >10 minutes where 16
# takes seconds.
_NCOPIES_AUTO_CAP = 16
# MX micro-scaling block: 32 values share one E8M0 scale, every format.
_SCALE_BLOCK = 32
# FP6 packing: four 6-bit codes tile three bytes exactly.
_FP6_CODES_PER_GROUP = 4
_FP6_BYTES_PER_GROUP = 3
# (mantissa_width, exponent_width, exponent_bias) per OCP MX FP6 encoding.
_FP6_PARAMS = {"e2m3": (3, 2, 1), "e3m2": (2, 3, 3)}


def _k_bytes(k: int, dtype: str) -> int:
    """Packed byte extent of one K row: 2 elems/byte at MXFP4, 4-per-3 at
    MXFP6, 1 at MXFP8."""
    if dtype == "mxfp4":
        return k // 2
    if dtype == "mxfp6":
        return k * _FP6_BYTES_PER_GROUP // _FP6_CODES_PER_GROUP
    return k


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------


def _auto_ncopies(weight_bytes_per_copy: int, target_bytes: int = 0) -> int:
    """Rotating-copy count so the working set exceeds the L2 (=> cold HBM
    reads), capped at `_NCOPIES_AUTO_CAP`.

    For large weights one or two copies already exceed L2. For small weights the
    cap binds and the working set stays partly L2-resident; exceeding 256 MB
    there needs hundreds of chained ops, which does not compile in reasonable
    time. Those shapes are reported as WARM by the caller rather than passed off
    as cold.

    `target_bytes` overrides the 1.5x-L2 goal with an explicit working-set size.
    That is what keeps a cross-format comparison honest: the same (N, K) has a
    different footprint in each format, so matching the COPY COUNT would put the
    formats at different L2 pressure -- at a cap of 16 MXFP8's MLP weights
    cleared 256 MB while MXFP4's and MXFP6's did not, crediting the smaller
    formats with L2 hits the larger one paid HBM for. Matching working-set BYTES
    instead puts all three in the same regime.
    """
    goal = target_bytes if target_bytes > 0 else int(1.5 * _L2_CACHE_SIZE_BYTES)
    return max(
        2,
        min(
            _NCOPIES_AUTO_CAP,
            math.ceil(goal / max(1, weight_bytes_per_copy)),
        ),
    )


def _equalized_working_set(n: int, k: int) -> int:
    """Working-set target that every format can reach within the op cap.

    Set by the *smallest* footprint of the three (MXFP4, 2 elements per byte):
    whatever copy count that format needs at the cap is the most bytes any
    format can be asked for without pushing a wider one past
    `_NCOPIES_AUTO_CAP` and back into the compile-time wall.
    """
    return n * _k_bytes(k, "mxfp4") * _NCOPIES_AUTO_CAP


def _compute_flops(m: int, n: int, k: int) -> int:
    """Dense GEMM FLOPs: 2*M*N*K (one multiply + one add per MAC)."""
    return 2 * m * n * k


# E2M1 (FP4) value table indexed by the 4-bit nibble (sign bit = bit 3).
_E2M1_VALUES = [
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
]


def _dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize a [R, K//2] uint8 (2 FP4/byte) + [R, K//32] E8M0 uint8 scale
    tensor to a [R, K] float32 reference. Low nibble = even element, high
    nibble = odd element (matches the kernels)."""
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    rows, k_half = packed.shape
    out = torch.empty(
        rows, k_half * 2, dtype=torch.float32, device=packed.device
    )
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    # E8M0: value = 2^(byte - 127); each scale covers _SCALE_BLOCK K-elements.
    sc = torch.exp2(scales.float() - 127.0).repeat_interleave(
        _SCALE_BLOCK, dim=1
    )
    return out * sc


def _gen_mxfp4_inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random valid MXFP4 GEMM inputs on the GPU. Every uint8 is a valid pair
    of E2M1 nibbles (E2M1 has no inf/nan). Scale bytes stay in [124, 128) so
    the E8M0 scale ~= 1 and never hits the 0xFF NaN code."""
    if k % _SCALE_BLOCK != 0:
        raise ValueError(f"K={k} must be a multiple of {_SCALE_BLOCK}")
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    a_s = torch.randint(
        124, 128, (m, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    b_s = torch.randint(
        124, 128, (n, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    return a, b, a_s, b_s


def _dequant_mxfp8(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize a [R, K] uint8 (1 E4M3/byte) + [R, K//32] E8M0 uint8 scale
    tensor to a [R, K] float32 reference."""
    out = packed.view(torch.float8_e4m3fn).float()
    sc = torch.exp2(scales.float() - 127.0).repeat_interleave(
        _SCALE_BLOCK, dim=1
    )
    return out * sc


def _gen_mxfp8_inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random valid MXFP8 GEMM inputs on the GPU. Values are drawn from
    [-1, 1] and cast through float8_e4m3fn (rather than filled with raw
    random bytes, since E4M3 has NaN codes E2M1 doesn't). Scale bytes stay
    in [124, 128) so the E8M0 scale ~= 1 and never hits the 0xFF NaN code."""
    if k % _SCALE_BLOCK != 0:
        raise ValueError(f"K={k} must be a multiple of {_SCALE_BLOCK}")
    a = (
        (torch.rand(m, k, device="cuda") * 2 - 1)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    b = (
        (torch.rand(n, k, device="cuda") * 2 - 1)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    a_s = torch.randint(
        124, 128, (m, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    b_s = torch.randint(
        124, 128, (n, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    return a, b, a_s, b_s


def _fp6_decode_table(fp6_format: str) -> list[float]:
    """Builds the 64-entry FP6 code-to-value table for an OCP MX FP6 encoding.

    Computed from the format parameters rather than transcribed, so E2M3 and
    E3M2 share one derivation. Mirrors `fp6_decode_table` in
    `max/python/max/pipelines/weights/fp6_quantization.py`, which is itself
    pinned bit-for-bit against `fp6_utils.mojo`.
    """
    m, e_width, bias = _FP6_PARAMS[fp6_format]
    out = []
    for code in range(64):
        exponent = (code >> m) & ((1 << e_width) - 1)
        mantissa = code & ((1 << m) - 1)
        sign = -1.0 if code & 0x20 else 1.0
        if exponent == 0:
            mag = mantissa * 2.0 ** (1 - bias - m)
        else:
            mag = (1.0 + mantissa * 2.0**-m) * 2.0 ** (exponent - bias)
        out.append(sign * mag)
    return out


def _unpack_fp6(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks a [R, K*3//4] uint8 tensor to [R, K] FP6 codes.

    Element i of a group occupies bits [6i+5 : 6i] of a little-endian 24-bit
    word, i.e. a group is a contiguous 6-bit stream. Inverse of `pack_fp6`.
    """
    rows, nbytes = packed.shape
    groups = packed.view(rows, -1, _FP6_BYTES_PER_GROUP).int()
    word = groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16)
    codes = torch.stack([(word >> (6 * i)) & 0x3F for i in range(4)], dim=-1)
    return codes.reshape(rows, nbytes * 4 // 3)


def _dequant_mxfp6(
    packed: torch.Tensor,
    scales: torch.Tensor,
    fp6_format: str = "e2m3",
) -> torch.Tensor:
    """Dequantize a [R, K*3//4] uint8 (4 FP6 codes / 3 bytes) + [R, K//32] E8M0
    scale tensor to a [R, K] float32 reference."""
    lut = torch.tensor(
        _fp6_decode_table(fp6_format),
        dtype=torch.float32,
        device=packed.device,
    )
    out = lut[_unpack_fp6(packed).long()]
    sc = torch.exp2(scales.float() - 127.0).repeat_interleave(
        _SCALE_BLOCK, dim=1
    )
    return out * sc


def _gen_mxfp6_inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random valid MXFP6 GEMM inputs on the GPU. Every uint8 is valid: FP6
    has no Inf or NaN encoding, so all 64 codes decode, and a byte is just
    six bits of one code plus part of the next. Scale bytes stay in [124, 128)
    so the E8M0 scale ~= 1 and never hits the 0xFF NaN code."""
    if k % _SCALE_BLOCK != 0:
        raise ValueError(f"K={k} must be a multiple of {_SCALE_BLOCK}")
    k_bytes = _k_bytes(k, "mxfp6")
    a = torch.randint(0, 256, (m, k_bytes), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (n, k_bytes), dtype=torch.uint8, device="cuda")
    a_s = torch.randint(
        124, 128, (m, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    b_s = torch.randint(
        124, 128, (n, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    return a, b, a_s, b_s


_DEQUANT = {
    "mxfp4": _dequant_mxfp4,
    "mxfp6": _dequant_mxfp6,
    "mxfp8": _dequant_mxfp8,
}
_GEN_INPUTS = {
    "mxfp4": _gen_mxfp4_inputs,
    "mxfp6": _gen_mxfp6_inputs,
    "mxfp8": _gen_mxfp8_inputs,
}


def _check_close(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    label: str,
    dtype: str = "mxfp4",
    fp6_format: str = "e2m3",
) -> None:
    """Compare a kernel output against the float32 dequantized reference."""
    dequant = _DEQUANT[dtype]
    kw = {"fp6_format": fp6_format} if dtype == "mxfp6" else {}
    ref = dequant(a, a_s, **kw) @ dequant(b, b_s, **kw).T
    out_f = out.detach().to(torch.float32)
    denom = ref.abs().max().clamp_min(1e-6)
    max_rel = (out_f - ref).abs().max() / denom
    print(f"  [{label} check] max_rel_err={max_rel.item():.4e}")
    torch.testing.assert_close(
        out_f, ref, rtol=3e-2, atol=(denom * 3e-2).item()
    )


# ----------------------------------------------------------------------------
# MAX backend.
# ----------------------------------------------------------------------------


def bench_matmul_max(
    m: int,
    n: int,
    k: int,
    num_iters: int,
    check: bool = False,
    ncopies: int | None = None,
    dtype: str = "mxfp4",
    fp6_format: str = "e2m3",
    preshuffled_b: bool = False,
) -> tuple[float, int] | None:
    """MAX dense MX block-scaled GEMM.

    MXFP4 and MXFP8 share `dynamic_block_scaled_matmul_amd`, which keys the
    packing off the input dtype. MXFP6 has its own entry point
    (`dynamic_block_scaled_matmul_mxfp6`): both FP6 encodings put 24 bytes in
    a lane, so the byte count cannot choose between them and the encoding
    travels as a parameter.

    Builds ONE device-graph chaining `ncopies` GEMM ops, op i reading a
    distinct rotating weight buffer (total > L2). A single replay sweeps all
    `ncopies` cold-HBM GEMMs; per-op latency = whole-graph time / ncopies.

    `preshuffled_b` (MXFP6 only) preshuffles every rotating B weight copy and
    its scales on the CPU (numpy, via `shuffle_block_scaled_b_dense_arrays`)
    before the timed graph replay -- the one-time, load-time cost a real
    checkpoint pays once via `preshuffle_block_scaled_b_dense`, so it must
    stay out of the per-op measurement below.
    """
    if preshuffled_b and dtype != "mxfp6":
        raise ValueError("preshuffled_b is only implemented for mxfp6")

    ncopies = _NCOPIES if ncopies is None else ncopies
    k_bytes = _k_bytes(k, dtype)
    if ncopies <= 0:
        ncopies = _auto_ncopies(
            n * k_bytes, target_bytes=_equalized_working_set(n, k)
        )

    a_t, b_t, a_s_t, b_s_t = _GEN_INPUTS[dtype](m, n, k)
    # MXFP4 and MXFP6 both travel as packed uint8; only MXFP8 has a real
    # element dtype.
    elem_dtype = DType.float8_e4m3fn if dtype == "mxfp8" else DType.uint8

    # `b_t`/`b_s_t` stay row-major throughout: `_check_close` below dequantizes
    # them with the row-major reference, so the correctness check must read
    # the un-shuffled bytes. `b_gpu_t`/`b_s_gpu_t` are what actually go on the
    # wire to the graph -- preshuffled copies when `preshuffled_b`.
    b_gpu_t, b_s_gpu_t = b_t, b_s_t
    if preshuffled_b:
        b_gpu_np, b_s_gpu_np = shuffle_block_scaled_b_dense_arrays(
            b_t.cpu().numpy(), b_s_t.cpu().numpy()
        )
        b_gpu_t = torch.from_numpy(b_gpu_np).to(b_t.device)
        b_s_gpu_t = torch.from_numpy(b_s_gpu_np).to(b_s_t.device)

    def _gen_weight_copy() -> torch.Tensor:
        if dtype == "mxfp8":
            bt = (
                (torch.rand(n, k_bytes, device="cuda") * 2 - 1)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        else:
            # Every byte is a valid MXFP4 nibble pair / MXFP6 code stream.
            bt = torch.randint(
                0, 256, (n, k_bytes), dtype=torch.uint8, device="cuda"
            )
        if preshuffled_b:
            # The B-scale argument only picks the shuffle's output shape
            # here; every rotating copy shares one already-shuffled
            # `b_s_gpu_t` graph input, so this result is discarded.
            bt_np, _ = shuffle_block_scaled_b_dense_arrays(
                bt.cpu().numpy(), b_s_t.cpu().numpy()
            )
            bt = torch.from_numpy(bt_np).cuda()
        return bt

    a_type = TensorType(elem_dtype, shape=[m, k_bytes], device=DeviceRef.GPU())
    b_type = TensorType(elem_dtype, shape=[n, k_bytes], device=DeviceRef.GPU())
    a_s_type = TensorType(
        DType.float8_e8m0fnu,
        shape=[m, k // _SCALE_BLOCK],
        device=DeviceRef.GPU(),
    )
    b_s_type = TensorType(
        DType.float8_e8m0fnu,
        shape=[n, k // _SCALE_BLOCK],
        device=DeviceRef.GPU(),
    )

    # Inputs are uint8-viewed so the dequant references can share one uint8
    # signature; reinterpret back to `elem_dtype` for the MAX buffer.
    def _as_buffer(t: torch.Tensor) -> Buffer:
        return Buffer.from_dlpack(t).view(elem_dtype)

    # Rotating weight copies (op i reads copy i in the chained graph below).
    keepalive: list[Any] = []
    b_bufs: list[Buffer] = [_as_buffer(b_gpu_t)]
    for _ in range(ncopies - 1):
        bt = _gen_weight_copy()
        keepalive.append(bt)
        b_bufs.append(_as_buffer(bt))

    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        f"{dtype}_matmul_max_chain",
        input_types=[
            a_type,
            a_s_type,
            b_s_type,
            *([b_type] * ncopies),
        ],
    ) as graph:
        ins = graph.inputs
        a, a_scales, b_scales = ins[0], ins[1], ins[2]
        b_in = ins[3 : 3 + ncopies]

        def _gemm(b: Any) -> Any:
            if dtype == "mxfp6":
                return dynamic_block_scaled_matmul_mxfp6(
                    a.tensor,
                    b.tensor,
                    a_scales.tensor,
                    b_scales.tensor,
                    fp6_format=fp6_format,
                    out_type=DType.bfloat16,
                    preshuffled_b=preshuffled_b,
                )
            return dynamic_block_scaled_matmul_amd(
                a.tensor,
                b.tensor,
                a_scales.tensor,
                b_scales.tensor,
                out_type=DType.bfloat16,
            )

        graph.output(*[_gemm(b) for b in b_in])

    model = session.load(graph)

    a_buf = _as_buffer(a_t)
    a_s_buf = Buffer.from_dlpack(a_s_t).view(DType.float8_e8m0fnu)
    b_s_buf = Buffer.from_dlpack(b_s_gpu_t).view(DType.float8_e8m0fnu)
    graph_inputs = (a_buf, a_s_buf, b_s_buf, *b_bufs)

    outs = model.capture(0, *graph_inputs)
    keepalive.append(outs)
    # Validate the captured graph matches eager execution.
    model.debug_verify_replay(0, *graph_inputs)

    if check:
        try:
            # Replay once so the captured output buffers hold live data, then
            # read op 0 (which uses weight copy 0 == b_t).
            model.replay(0, *graph_inputs)
            torch.cuda.synchronize()
            out0 = torch.from_dlpack(outs[0]).clone()
            _check_close(
                out0,
                a_t,
                b_t,
                a_s_t,
                b_s_t,
                "MAX",
                dtype=dtype,
                fp6_format=fp6_format,
            )
        except Exception as e:
            print(f"  [MAX check skipped: {e}]")

    nrun = max(num_iters, 200)
    torch.cuda.synchronize()
    for _ in range(50):
        model.replay(0, *graph_inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nrun):
        model.replay(0, *graph_inputs)
    end.record()
    torch.cuda.synchronize()
    per_op_s = start.elapsed_time(end) / 1e3 / nrun / ncopies

    weight_bytes_per_copy = n * k_bytes
    working_set_bytes = weight_bytes_per_copy * ncopies
    w_mb = weight_bytes_per_copy / (1024.0 * 1024.0)
    cold_state = (
        "cold"
        if working_set_bytes > _L2_CACHE_SIZE_BYTES
        else f"WARM: working set < {_L2_CACHE_SIZE_BYTES / 1e6:.0f}MB L2"
    )
    print(
        f"[MAX chained device-graph] chain={ncopies} "
        f"weight_per_copy~{w_mb:.1f}MB working_set~{w_mb * ncopies:.1f}MB ({cold_state})"
        f" | per-op {per_op_s * 1e6:.2f}us"
    )
    keepalive.clear()
    return per_op_s, _compute_flops(m, n, k)


# ----------------------------------------------------------------------------
# aiter backend.
# ----------------------------------------------------------------------------


def bench_matmul_aiter(
    m: int,
    n: int,
    k: int,
    num_iters: int,
    check: bool = False,
    ncopies: int | None = None,
    dtype: str = "mxfp4",
    fp6_format: str = "e2m3",
) -> tuple[float, int] | None:
    """aiter gemm_afp4wfp4 (Triton `_gemm_afp4wfp4_kernel`).

    Mirrors the MAX path: `ncopies` rotating weight buffers (total > L2)
    chained into ONE CUDA graph, one output buffer per chained call to avoid a
    false write-after-write dependency. per-op = whole-graph time / ncopies.
    """
    if dtype != "mxfp4":
        print(
            f"aiter has no native MX-format {dtype} kernel in this version, "
            "skipping bench_matmul_aiter"
        )
        return None
    if _aiter_gemm is None:
        print("aiter not available, skipping bench_matmul_aiter")
        return None
    ncopies = _NCOPIES if ncopies is None else ncopies
    if ncopies <= 0:
        ncopies = _auto_ncopies(
            n * (k // 2), target_bytes=_equalized_working_set(n, k)
        )

    a_t, b0_t, a_s_t, b_s_t = _gen_mxfp4_inputs(m, n, k)
    b_bufs = [b0_t] + [
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        for _ in range(ncopies - 1)
    ]
    # One output per chained call so consecutive calls don't serialize on a
    # shared output tensor.
    o_list = [
        torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        for _ in range(ncopies)
    ]

    def call(b_buf: torch.Tensor, out: torch.Tensor) -> None:
        _aiter_gemm(a_t, b_buf, a_s_t, b_s_t, dtype=torch.bfloat16, y=out)

    if check:
        call(b0_t, o_list[0])
        torch.cuda.synchronize()
        _check_close(o_list[0], a_t, b0_t, a_s_t, b_s_t, "aiter")

    nrun = max(num_iters, 200)
    # Capture one CUDA graph chaining `ncopies` GEMM calls over distinct cold
    # weight buffers. Side-stream warmup is mandatory before capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            for b in range(ncopies):
                call(b_bufs[b], o_list[b])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gchain = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gchain):
        for b in range(ncopies):
            call(b_bufs[b], o_list[b])
    torch.cuda.synchronize()
    for _ in range(50):
        gchain.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nrun):
        gchain.replay()
    end.record()
    torch.cuda.synchronize()
    per_op_s = start.elapsed_time(end) / 1e3 / nrun / ncopies

    weight_bytes_per_copy = n * (k // 2)
    working_set_bytes = weight_bytes_per_copy * ncopies
    w_mb = weight_bytes_per_copy / (1024.0 * 1024.0)
    cold_state = (
        "cold"
        if working_set_bytes > _L2_CACHE_SIZE_BYTES
        else f"WARM: working set < {_L2_CACHE_SIZE_BYTES / 1e6:.0f}MB L2"
    )
    print(
        f"[aiter chained CUDA-graph] chain={ncopies} "
        f"weight_per_copy~{w_mb:.1f}MB working_set~{w_mb * ncopies:.1f}MB ({cold_state})"
        f" | per-op {per_op_s * 1e6:.2f}us"
    )
    return per_op_s, _compute_flops(m, n, k)


# ----------------------------------------------------------------------------
# Dispatch and CLI.
# ----------------------------------------------------------------------------


_ENGINE_MAP: dict[str, Callable[..., tuple[float, int] | None]] = {
    "modular_max": bench_matmul_max,
    "aiter": bench_matmul_aiter,
}


def bench_matmul(
    engine: str,
    m: int,
    n: int,
    k: int,
    num_iters: int,
    check: bool,
    dtype: str = "mxfp4",
    fp6_format: str = "e2m3",
    preshuffled_b: bool = False,
) -> tuple[float, int] | None:
    print("=" * 80)
    label = f"{dtype.upper()}" + (
        f"({fp6_format.upper()})" if dtype == "mxfp6" else ""
    )
    label += "+preshuffled_b" if preshuffled_b else ""
    print(f"AMD {label} GEMM (M={m}, N={n}, K={k}, engine={engine})")
    print("=" * 80)

    fn = _ENGINE_MAP.get(engine)
    if fn is None:
        raise ValueError(
            f"Unknown engine '{engine}'. Available: {list(_ENGINE_MAP.keys())}"
        )
    if preshuffled_b and engine != "modular_max":
        raise ValueError("preshuffled_b is only implemented for modular_max")

    try:
        result = fn(
            m,
            n,
            k,
            num_iters,
            check=check,
            dtype=dtype,
            fp6_format=fp6_format,
            **(
                {"preshuffled_b": preshuffled_b}
                if engine == "modular_max"
                else {}
            ),
        )
    except Exception as e:
        print(f"{engine} benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    if result is not None:
        time_s, flops = result
        tflops = flops / time_s / 1e12
        print(f"  Time: {time_s * 1e6:.3f} us | {tflops:.2f} TFLOPS")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMD MXFP4/MXFP6/MXFP8 GEMM Benchmark"
    )
    parser.add_argument(
        "--engine",
        choices=list(_ENGINE_MAP.keys()),
        default="modular_max",
    )
    parser.add_argument(
        "--dtype",
        choices=["mxfp4", "mxfp6", "mxfp8"],
        default="mxfp4",
        help="Quantization format. mxfp6 and mxfp8 run MAX-only: aiter has "
        "no native MX-format FP6 or FP8 kernel in this version.",
    )
    parser.add_argument(
        "--fp6_format",
        "--fp6-format",
        choices=["e2m3", "e3m2"],
        default="e2m3",
        help="FP6 element encoding, mxfp6 only. e2m3 has 3 mantissa bits and "
        "is what the M3 MXFP6 checkpoints ship.",
    )
    parser.add_argument("--M", "--m", type=int, default=4096, help="GEMM M")
    parser.add_argument("--N", "--n", type=int, default=16384, help="GEMM N")
    parser.add_argument("--K", "--k", type=int, default=2048, help="GEMM K")
    parser.add_argument("--num_iters", "--num-iters", type=int, default=100)
    parser.add_argument("--output", "-o", type=str, default="output.csv")
    parser.add_argument(
        "--ncopies",
        type=int,
        default=0,
        help="Rotating weight copies chained per graph (0 = auto: footprint "
        f"> L2, capped at {_NCOPIES_AUTO_CAP}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the kernel against a float32 dequantized reference.",
    )
    parser.add_argument(
        "--preshuffled_b",
        "--preshuffled-b",
        action="store_true",
        help="mxfp6/modular_max only. Preshuffle B and its scales on the CPU "
        "before the timed replay and dispatch through "
        "`mxfp6_block_scaled_matmul_amd`'s preshuffled-B path -- the M > 64 "
        "load-path fix. See `preshuffle_block_scaled_b_dense`.",
    )
    args, _ = parser.parse_known_args()

    global _NCOPIES
    _NCOPIES = args.ncopies

    print(
        f"[bench_amd_mx_gemm] engine={args.engine} dtype={args.dtype} "
        f"fp6_format={args.fp6_format} M={args.M} N={args.N} K={args.K}",
        file=sys.stderr,
    )

    result = bench_matmul(
        args.engine,
        args.M,
        args.N,
        args.K,
        args.num_iters,
        args.check,
        dtype=args.dtype,
        fp6_format=args.fp6_format,
        preshuffled_b=args.preshuffled_b,
    )

    if result is None:
        sys.exit(1)

    time_s, flops = result
    metric = ThroughputMeasure(Bench.flops, flops)
    dtype_tag = args.dtype + (
        f"_{args.fp6_format}" if args.dtype == "mxfp6" else ""
    )
    dtype_tag += "_preb" if args.preshuffled_b else ""
    name = (
        f"Matmul_{dtype_tag.upper()}/M={args.M}/N={args.N}/K={args.K}/"
        f"dtype={dtype_tag}/engine={args.engine}/"
    )
    b = Bench(name, iters=1, met=time_s, metric_list=[metric])
    b.dump_report(output_path=args.output)


if __name__ == "__main__":
    main()
