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
"""Accuracy of the fused reduce_add + rms_norm path vs a torch reference.

Exercises the f32 cols=256 single-pass static-divisor rms_norm kernel that the
``reduce.add -> slice + add`` decompose fusion produces. The graph is:

    x[E, H, KB, V] --sum(axis=2)--> [E, H, 1, V]
                   --cast bf16--> --cast f32-->        (lossy round-trip, kept)
                   --rms_norm over V (weight = ones)--> --cast bf16--> out

The graph compiler decomposes the size-``KB`` reduction into elementwise + view
ops and folds them, plus both casts, into the rms_norm load. This test checks
that the fused MAX output matches a torch eager reference computing the
identical sequence (so the optimization does not change numerics). Both sides
accumulate in f32 and cast to bf16, so the comparison is at bf16 tolerance.

Cases vary the reduced axis (including a negative ``axis=-2``), its extent
across the decompose bound (KB in {2, 8, 9}), and static vs dynamic outer dims,
so both the fused decompose path and the fallback (gate-declined) path are
checked for numerical parity with torch.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

EPS = 1e-5
V = 256  # normalized axis -- the f32 cols==256 single-pass regime

# Both sides do f32 accumulation then a single bf16 store; the tolerance covers
# bf16 rounding plus any reduction-order difference between the kernels.
ATOL = 1e-2
RTOL = 1e-2

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0,
    reason="reduce_add + rms_norm fusion path is GPU-only",
)

CASES = [
    # static shape, KB=2 -> decompose fires
    ([4096, 6, 2, V], [4096, 6, 2, V], 2),
    # negative axis: -2 normalizes to the KB axis, so the decompose still fires
    ([4096, 6, 2, V], [4096, 6, 2, V], -2),
    # outer (non-last) axis: reduce axis=1 (extent 4) instead of the KB axis,
    # feeding rms_norm over V. The reduced axis collapses into the row dim and
    # the slice+add folds into the row load -- same fused single-pass kernel.
    ([1024, 4, 2, V], [1024, 4, 2, V], 1),
    # extent 8 == inclusive decompose upper bound -> decompose fires (smaller
    # outer dims keep the buffer in-pool; the single-pass regime is row-count
    # independent, so the same kernel still runs)
    ([1024, 6, 8, V], [1024, 6, 8, V], 2),
    # extent 9 == one past the bound -> gate declines, fallback reduce kernel
    ([1024, 6, 9, V], [1024, 6, 9, V], 2),
    # dynamic outer dims, static reduced axis -> decompose now fires (only the
    # reduced axis must be static); accuracy must still match torch
    (["e", "h", 2, V], [4096, 6, 2, V], 2),
    # dynamic reduced axis -> unroll count unknown at compile time, so the gate
    # declines (fallback reduce kernel); accuracy must still match torch
    (["e", "h", "kb", V], [4096, 6, 2, V], 2),
]
CASE_IDS = [
    "static_kb2_axis2_fused",
    "static_kb2_negaxis_fused",
    "static_axis1_outer_fused",
    "static_kb8_boundary_fused",
    "static_kb9_fallback",
    "dyn_outer_static_inner_fused",
    "dyn_reduced_axis_fallback",
]


@pytest.fixture(scope="module")
def gpu_session() -> tuple[InferenceSession, Accelerator]:
    device = Accelerator(0)
    return InferenceSession(devices=[device]), device


def _bf16_buffer_to_f32(buf: Buffer) -> np.ndarray:
    """Read a bf16 device buffer back to host as float32."""
    host = buf.copy(device=CPU())
    return (
        host.view(DType.uint16)
        .to_numpy()
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def _max_fused(
    session: InferenceSession,
    device: Accelerator,
    type_shape: list[int | str],
    axis: int,
    x_np: np.ndarray,
) -> np.ndarray:
    """Run the fused reduce_add + rms_norm graph; return the [rows, V] output."""
    gpu = DeviceRef.GPU()
    with Graph(
        "reduce_add_rms_norm",
        input_types=(
            TensorType(DType.float32, type_shape, device=gpu),
            TensorType(DType.float32, [V], device=gpu),
        ),
    ) as graph:
        x = graph.inputs[0].tensor
        weight = graph.inputs[1].tensor
        gla = ops.sum(x, axis=axis)  # [E, H, 1, V] keepdim, f32
        gla = ops.cast(gla, DType.bfloat16)  # lossy GLA round-trip (kept)
        x_fp32 = ops.cast(gla, DType.float32)
        normed = ops.rms_norm(x_fp32, weight, EPS)  # f32
        graph.output(ops.cast(normed, DType.bfloat16))

    compiled = session.load(graph)
    x_buf = Buffer.from_numpy(np.ascontiguousarray(x_np)).to(device)
    w_buf = Buffer.from_numpy(np.ones([V], dtype=np.float32)).to(device)
    (result,) = compiled.execute(x_buf, w_buf)
    return _bf16_buffer_to_f32(result).reshape(-1, V)


def _torch_ref(x_np: np.ndarray, axis: int) -> np.ndarray:
    """Torch eager reference for the identical op sequence -> [rows, V].

    Runs on CPU: it is a numerical reference, and MAX holds the GPU under the
    bazel test sandbox (torch's CUDA pool is capped there). The math is
    identical whether computed on host or device.
    """
    x = torch.from_numpy(x_np)
    gla = x.sum(dim=axis).to(torch.bfloat16)  # lossy round-trip, matching MAX
    x_fp32 = gla.float()
    inv_rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + EPS)
    out = (x_fp32 * inv_rms).to(torch.bfloat16)
    return out.float().numpy().reshape(-1, V)


@pytest.mark.parametrize("type_shape,concrete,axis", CASES, ids=CASE_IDS)
def test_reduce_add_rmsnorm_matches_torch(
    gpu_session: tuple[InferenceSession, Accelerator],
    type_shape: list[int | str],
    concrete: list[int],
    axis: int,
) -> None:
    session, device = gpu_session
    x = np.random.default_rng(0).standard_normal(concrete).astype(np.float32)
    got = _max_fused(session, device, type_shape, axis, x)
    ref = _torch_ref(x, axis)
    np.testing.assert_allclose(got, ref, atol=ATOL, rtol=RTOL)
