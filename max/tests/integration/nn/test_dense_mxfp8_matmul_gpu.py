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
"""De-risk test for the dense MXFP8 block-scaled matmul path on SM100.

Dense MXFP8 linear layers (``float8_e4m3fn`` weights, ``float8_e8m0fnu`` E8M0
block scales, 32-element K blocks) currently fall back to the naive CUDA-core
``naive_blockwise_scaled_fp8_matmul`` kernel because the 1d2d blockwise-fp8
path only supports K-scale-granularity 128. The SM100 tensor-core block-scaled
MMA (``UMMAKind.KIND_MXF8F6F4``) does support 32-element MXFP8 scaling.

This test exercises that tensor-core path through the same graph ops a wired-up
``QuantFormat.MXFP8`` linear would use:

    quantize_dynamic_block_scaled  (mo.quantize.dynamic.block.scaled)
    dynamic_block_scaled_matmul    (mo.matmul.dynamic.block.scaled)

It quantizes random bf16 activations and weights to MXFP8 in-graph, runs the
block-scaled matmul, and checks the result against an fp32 reference of the
original (un-quantized) ``a @ b.T``. The tolerance is loose enough to absorb
MXFP8 round-trip error but tight enough to catch a wrong layout / wrong kernel
(which would destroy the cosine similarity).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    dynamic_block_scaled_matmul,
    quantize_dynamic_block_scaled,
)
from test_common.graph_utils import is_b100_b200


def _skip_if_not_supported() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available for MXFP8 matmul test")
    if accelerator_api() == "hip":
        pytest.skip("MXFP8 block-scaled MMA only supports NVIDIA GPUs")
    if not is_b100_b200():
        pytest.skip("MXFP8 block-scaled MMA requires B100 or B200 (SM100)")


# Representative MiniMax-M3 dense-layer shapes (M = batch*tokens, N = out, K = in):
#   q_proj   [8192, 6144]
#   gate/up  [12288, 6144]
#   down     [6144, 12288]
# K must be a multiple of 128 (the rank-5 SF K-group size); both 6144 and 12288
# qualify. M and N are tiled by 128 with partial-tile masking, so a decode-sized
# M = 1 is exercised too.
@pytest.mark.parametrize(
    "label,M,N,K",
    [
        ("q_proj_prefill", 128, 8192, 6144),
        ("gate_up_prefill", 256, 12288, 6144),
        ("down_prefill", 256, 6144, 12288),
        ("q_proj_decode", 1, 8192, 6144),
    ],
)
def test_dense_mxfp8_block_scaled_matmul(
    label: str, M: int, N: int, K: int
) -> None:
    _skip_if_not_supported()

    rng = np.random.default_rng(0)
    # Small-magnitude Gaussian inputs keep values comfortably inside the E4M3
    # dynamic range so block scaling is well-conditioned.
    a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    b_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    with Graph(
        f"dense_mxfp8_matmul_{label}",
        input_types=[
            TensorType(DType.bfloat16, shape=(M, K), device=device_ref),
            TensorType(DType.bfloat16, shape=(N, K), device=device_ref),
        ],
    ) as graph:
        a_bf16, b_bf16 = (inp.tensor for inp in graph.inputs)

        a_q, a_scales = quantize_dynamic_block_scaled(
            a_bf16,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        b_q, b_scales = quantize_dynamic_block_scaled(
            b_bf16,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )

        out = dynamic_block_scaled_matmul(
            a_q,
            b_q,
            a_scales,
            b_scales,
            sf_vector_size=32,
            out_type=DType.bfloat16,
        )
        graph.output(out)

    model = session.load(graph)

    a_buf = Buffer.from_dlpack(torch.from_numpy(a_np).to(torch.bfloat16)).to(
        device
    )
    b_buf = Buffer.from_dlpack(torch.from_numpy(b_np).to(torch.bfloat16)).to(
        device
    )

    (out_buf,) = model.execute(a_buf, b_buf)
    out = torch.from_dlpack(out_buf).to(torch.float32).cpu().numpy()

    ref = a_np @ b_np.T

    out_flat = out.reshape(-1)
    ref_flat = ref.reshape(-1)
    cos = float(
        np.dot(out_flat, ref_flat)
        / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-12)
    )
    rel_err = float(
        np.linalg.norm(out_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-12)
    )
    print(
        f"\n=== {label} (M={M}, N={N}, K={K}) ===\n"
        f"  cosine similarity : {cos:.5f}\n"
        f"  relative L2 error : {rel_err:.5f}",
        flush=True,
    )

    # MXFP8 round-trip on random Gaussian inputs averaged over K leaves high
    # cosine similarity; a wrong layout / wrong kernel collapses it. The rel-L2
    # bound mainly guards against a systematic scale error.
    assert cos > 0.99, f"{label}: cosine {cos:.5f} too low (wrong result?)"
    assert rel_err < 0.1, f"{label}: relative L2 error {rel_err:.5f} too high"
