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

"""End-to-end tests for reshape-into-epilogue fusion under advanced fusion.

Every graph here is compiled with ``MAX_GC_USE_ADV_FUSION`` set, exercising
``MOToMAP`` + ``ReshapeEpilogueFuser`` end to end: a reshape between a
fused-output producer and a downstream elementwise op folds into the
producer's epilogue as a ``mogg.index.reshape`` store-index transform (the
store lands at the reshaped position), so the elementwise op fuses too and the
reshape no longer materializes its own kernel. These prove numeric correctness
across the reshape kinds -- the store-index math itself is pinned by
``max/kernels/test/graph_compiler/test_index_reshape.mojo`` and the IR shape by
the mo-opt suite. Distinct element values catch a scrambled store index, not
just a wrong output shape.
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_matmul_reshape_add_static(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`reshape(matmul(a, b), [4]) + bias`, all static: the flatten folds into
    the matmul's epilogue store index. Mirrors `matmul-reshape-add-epilogue-fusion.mlir`.
    """
    with Graph(
        "matmul_reshape_add_static",
        input_types=[
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, bias = (v.tensor for v in graph.inputs)
        graph.output(ops.reshape(a @ b, [4]) + bias)

    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.eye(2, dtype=np.float32)
    bias_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, bias_np, fused=r"mo\.matmul.*mo\.add"
    )
    np.testing.assert_allclose(
        out, (a_np @ b_np).reshape(4) + bias_np, rtol=1e-5, atol=1e-5
    )


def test_matmul_reshape_add_dynamic_batch(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`reshape(matmul(a, b), [M, 2, 2]) + bias` with a dynamic OUTER (batch)
    dim `M`: it never enters a stride or a needed modulo, so the store-index
    transform needs no runtime shape. Mirrors
    `matmul-reshape-add-epilogue-fusion-dynamic.mlir`.
    """
    with Graph(
        "matmul_reshape_add_dynamic_batch",
        input_types=[
            TensorType(DType.float32, ["M", 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["M", 2, 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, bias = (v.tensor for v in graph.inputs)
        mm = a @ b  # [M, 4]
        graph.output(ops.reshape(mm, [mm.shape[0], 2, 2]) + bias)

    a_np = np.ones((2, 2), dtype=np.float32)
    b_np = np.ones((2, 4), dtype=np.float32)
    bias_np = np.ones((2, 2, 2), dtype=np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, bias_np, fused=r"mo\.matmul.*mo\.add"
    )
    # ones @ ones (K=2) = 2 everywhere; reshape to [M, 2, 2]; + ones = 3.
    np.testing.assert_allclose(
        out, np.full((2, 2, 2), 3.0), rtol=1e-5, atol=1e-5
    )


def test_matmul_reshape_add_inner_dynamic(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`reshape(matmul(a, b), [N, 2]) + bias` with an INNER dynamic dim `N`:
    `[2, N] -> [N, 2]` re-linearizes across `N`, so the transform reads the
    runtime shapes threaded into the epilogue functor. Mirrors
    `matmul-reshape-add-epilogue-fusion-inner-dynamic.mlir`.
    """
    with Graph(
        "matmul_reshape_add_inner_dynamic",
        input_types=[
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, "N"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["N", 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, bias = (v.tensor for v in graph.inputs)
        mm = a @ b  # [2, N]
        graph.output(ops.reshape(mm, [b.shape[1], 2]) + bias)

    a_np = np.ones((2, 2), dtype=np.float32)
    b_np = np.ones((2, 4), dtype=np.float32)
    bias_np = np.ones((4, 2), dtype=np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, bias_np, fused=r"mo\.matmul.*mo\.add"
    )
    # ones @ ones (K=2) = 2 everywhere; reshape [2, 4] -> [4, 2]; + ones = 3.
    np.testing.assert_allclose(out, np.full((4, 2), 3.0), rtol=1e-5, atol=1e-5)


def test_reduce_max_reshape_add_non_accumulating(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """Reshape fused into a NON-accumulating fused-output epilogue: `reduce.max`
    never touches the output pointer during its own compute, isolating the
    reshape store-index path from accumulator concerns. Stands in for
    `imposter-reshape-add-epilogue-fusion.mlir`'s signature-only imposter
    kernel, which has no real-op counterpart.
    """
    with Graph(
        "reduce_max_reshape_add",
        input_types=[
            TensorType(DType.float32, [2, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, bias = (v.tensor for v in graph.inputs)
        reduced = ops.max(x, axis=0)  # [1, 4]
        graph.output(ops.reshape(reduced, [4]) + bias)

    x_np = np.random.randn(2, 4).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, x_np, bias_np, fused=r"mo\.reduce\.max.*mo\.add"
    )
    ref = np.max(x_np, axis=0).reshape(4) + bias_np
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
