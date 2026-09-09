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

"""End-to-end test for the float32-RMSNorm-sandwich + rotate-half RoPE fusion.

This is the JSC-32 query/key preprocess graph: a bfloat16 activation is upcast
to float32, RMS-normalized, cast back to bfloat16, then run through rotate-half
RoPE. The graph compiler must fuse this into a single ``rms_norm_rope`` GPU
kernel (input upcast absorbed by prologue fusion, output produced in bfloat16
via the kernel's decoupled output dtype). GPU-only: the composite has no CPU
kernel.
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

N, NHEADS, HEAD = 256, 6, 256
EPS = 1e-6


def _build_graph(dev: DeviceRef) -> Graph:
    with Graph(
        "qk_preprocess",
        input_types=[
            TensorType(DType.bfloat16, [N, NHEADS, HEAD], device=dev),
            TensorType(DType.bfloat16, [N, 1, HEAD], device=dev),  # cos
            TensorType(DType.bfloat16, [N, 1, HEAD], device=dev),  # sin
        ],
    ) as graph:
        q, cos, sin = (i.tensor for i in graph.inputs)

        # RMSNorm written "in float32": upcast -> normalize -> downcast.
        weight = ops.constant(1.0, DType.float32, dev).broadcast_to([HEAD])
        normed = ops.rms_norm(q.cast(DType.float32), weight, EPS).cast(
            DType.bfloat16
        )

        # Rotate-half RoPE.
        half = HEAD // 2
        x1, x2 = ops.split(normed, [half, half], axis=-1)
        rotated = ops.concat([-x2, x1], axis=-1)
        cos_b = cos.broadcast_to([N, NHEADS, HEAD])
        sin_b = sin.broadcast_to([N, NHEADS, HEAD])
        graph.output(normed * cos_b + rotated * sin_b)

    return graph


def _torch_ref(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    q_f32 = q.float()
    normed = q_f32 * torch.rsqrt(q_f32.pow(2).mean(-1, keepdim=True) + EPS)
    n = normed.to(torch.bfloat16)
    half = n.shape[-1] // 2
    x1, x2 = n[..., :half], n[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return n * cos + rotated * sin


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_rms_norm_rope_fusion(session: InferenceSession) -> None:
    dev = DeviceRef.from_device(session.devices[0])
    model = session.load(_build_graph(dev))

    # 1. The graph compiler fused the f32-sandwich RMSNorm + RoPE into the
    #    composite kernel (rather than several separate elementwise kernels).
    summaries = model.kernel_summaries
    combined = " ".join(summaries)
    assert "rms_norm_rope" in combined, (
        f"expected a fused rms_norm_rope kernel, got: {summaries}"
    )

    # 2. The fused kernel is numerically equivalent to the unfused reference.
    torch.manual_seed(0)
    q = torch.randn(N, NHEADS, HEAD, dtype=torch.bfloat16)
    cos = torch.randn(N, 1, HEAD, dtype=torch.bfloat16)
    sin = torch.randn(N, 1, HEAD, dtype=torch.bfloat16)

    out = model(
        Buffer.from_dlpack(q).to(model.input_devices[0]),
        Buffer.from_dlpack(cos).to(model.input_devices[1]),
        Buffer.from_dlpack(sin).to(model.input_devices[2]),
    )[0]

    # bfloat16 buffers can't go through ``Buffer.to_numpy`` (numpy has no
    # bfloat16); read back via DLPack into torch and upcast on host instead.
    max_np = torch.from_dlpack(out).to("cpu").to(torch.float32).numpy()
    ref_np = _torch_ref(q, cos, sin).float().numpy()

    cos_sim = _cosine(max_np, ref_np)
    assert cos_sim > 0.999, f"cosine similarity too low: {cos_sim}"
