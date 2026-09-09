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
"""End-to-end correctness for the short-axis GPU softmax fast/safe arms.

The short-inner-axis softmax warp kernel picks its load arm from the
graph-compiler `has_prologue_fusion` flag:

- No real input fusion (`softmax(x)`) -> flat fast path.
- A fused elementwise producer (`softmax(x * scale)`) -> true-coordinate
  fusion-safe path; the per-row coordinate is decomposed using the output
  layout's dims (compile-time constants for a static graph shape).
"""

import ml_dtypes
import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

SCALE = 1.7

TOL = {DType.float32: 1e-5, DType.bfloat16: 1e-2}
DTYPES = [DType.float32, DType.bfloat16]
DTYPE_IDS = ["f32", "bf16"]

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0,
    reason="short-axis softmax fast/safe arms are GPU-only",
)

STATIC_SHAPES = [
    [1, 1, 1, 1],
    [4, 8, 8, 24],
    [4, 8, 16, 33],  # just over the boundary -> block
]

# [dynamic, dynamic, static, static]: outer dims symbolic, inner dims static.
# (type_shape feeds the graph; concrete is the buffer/reference shape.)
DYN_CASES = [
    (["b", "s", 16, 24], [4, 8, 16, 24]),
    (["b", "s", 4, 64], [2, 3, 4, 64]),  # block
]
DYN_IDS = ["dynxdynx16x24", "dynxdynx4x64"]


@pytest.fixture(scope="module")
def gpu_session() -> tuple[InferenceSession, Accelerator]:
    device = Accelerator(0)
    return InferenceSession(devices=[device]), device


def _np_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _round(x: np.ndarray, dtype: DType) -> np.ndarray:
    if dtype == DType.bfloat16:
        return x.astype(ml_dtypes.bfloat16).astype(np.float32)
    return x


def _to_buffer(np_f32: np.ndarray, dtype: DType, device: Accelerator) -> Buffer:
    if dtype == DType.bfloat16:
        bits = np.ascontiguousarray(
            np_f32.astype(ml_dtypes.bfloat16).view(np.uint16)
        )
        return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)
    return Buffer.from_numpy(np_f32).to(device)


def _run(
    session: InferenceSession,
    device: Accelerator,
    type_shape: list[int | str],  # ints (static) and/or symbolic-dim names
    np_in: np.ndarray,
    *,
    fused: bool,
    dtype: DType,
) -> np.ndarray:
    gpu = DeviceRef.GPU()
    with Graph(
        "softmax_fusion",
        input_types=(TensorType(dtype, type_shape, device=gpu),),
    ) as graph:
        x = graph.inputs[0].tensor
        if fused:
            x = ops.mul(x, ops.constant(SCALE, dtype, gpu))
        graph.output(ops.softmax(x, axis=-1))

    compiled = session.load(graph)
    (result,) = compiled.execute(_to_buffer(np_in, dtype, device))
    out = result.copy(device=CPU())
    if dtype == DType.bfloat16:
        return (
            out.view(DType.uint16)
            .to_numpy()
            .view(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
    return out.to_numpy()


def _check(
    gpu_session: tuple[InferenceSession, Accelerator],
    type_shape: list[int | str],
    concrete_shape: list[int],
    *,
    fused: bool,
    seed: int,
    dtype: DType,
) -> None:
    session, device = gpu_session
    x = np.random.default_rng(seed).standard_normal(concrete_shape)
    x = x.astype(np.float32)
    got = _run(session, device, type_shape, x, fused=fused, dtype=dtype)

    xin = _round(x, dtype)
    if fused:
        scale = (
            np.float32(ml_dtypes.bfloat16(SCALE))
            if dtype == DType.bfloat16
            else np.float32(SCALE)
        )
        xin = _round(xin * scale, dtype)
    ref = _round(_np_softmax(xin), dtype)
    tol = TOL[dtype]
    np.testing.assert_allclose(got, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize(
    "shape", STATIC_SHAPES, ids=lambda s: "x".join(str(d) for d in s)
)
def test_softmax_static_shapes(gpu_session, shape, fused, dtype) -> None:  # noqa: ANN001
    """Static shapes across the WARP_SIZE boundary, both load arms, fp32+bf16."""
    _check(gpu_session, shape, shape, fused=fused, seed=0, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("type_shape,concrete", DYN_CASES, ids=DYN_IDS)
def test_softmax_dynamic_outer_static_inner(
    gpu_session,  # noqa: ANN001
    type_shape,  # noqa: ANN001
    concrete,  # noqa: ANN001
    fused,  # noqa: ANN001
    dtype,  # noqa: ANN001
) -> None:
    """[dynamic, dynamic, static, static]: symbolic outer dims, static inner
    dims, in fp32 and bf16. The warp kernel decomposes the row index with mixed
    runtime/static dims; result must still match the reference."""
    _check(gpu_session, type_shape, concrete, fused=fused, seed=1, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_softmax_sliced_view(gpu_session, dtype) -> None:  # noqa: ANN001
    """Softmax over a non-contiguous fused VIEW (a slice).

    Regression: `x[:, :-1]` is a view whose outer stride reflects the original
    (un-sliced) size. A fused view producer sets `has_prologue_fusion == True`,
    so the warp kernel must use true coordinates; a flat fast path would assume
    contiguity and read the wrong elements for outer rows. `vocab = 5` keeps this
    on the warp path. Mirrors the rejection sampler's `[:, :-1]` bonus-token
    slice.
    """
    session, device = gpu_session
    batch, steps, vocab = 3, 4, 5
    x = np.random.default_rng(2).standard_normal((batch, steps + 1, vocab))
    x = x.astype(np.float32)

    gpu = DeviceRef.GPU()
    with Graph(
        "softmax_sliced",
        input_types=(TensorType(dtype, [batch, steps + 1, vocab], device=gpu),),
    ) as graph:
        # Drop the last step -> non-contiguous view feeding softmax.
        graph.output(ops.softmax(graph.inputs[0].tensor[:, :-1]))

    compiled = session.load(graph)
    (result,) = compiled.execute(_to_buffer(x, dtype, device))
    out = result.copy(device=CPU())
    if dtype == DType.bfloat16:
        got = (
            out.view(DType.uint16)
            .to_numpy()
            .view(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
    else:
        got = out.to_numpy()

    ref = _round(_np_softmax(_round(x, dtype)[:, :-1]), dtype)
    tol = TOL[dtype]
    np.testing.assert_allclose(got, ref, atol=tol, rtol=tol)
