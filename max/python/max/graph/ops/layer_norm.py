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
"""Op implementation for layer_norm."""

from max._core.dialects import kgen, mo
from max.dtype import DType

from .. import dtype_promotion
from ..dim import StaticDim
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant


def layer_norm(
    input: TensorValue,
    gamma: TensorValueLike,
    beta: TensorValueLike,
    epsilon: float,
) -> TensorValue:
    """Computes layer normalization over the last dimension of ``input``.

    The output is ``gamma * (input - mean) / sqrt(var + epsilon) + beta``,
    where ``mean`` and ``var`` are reduced over the last axis of ``input``
    and broadcast back across the leading axes.

    Reduction is performed in the dtype of ``input``. For numerically stable
    normalization on float16 or bfloat16 inputs, cast to float32 before
    calling this op and cast the result back.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("layer_norm_example") as graph:
            x = ops.constant([[1.0, 3.0]], DType.float32, device=device)
            gamma = ops.constant([1.0, 1.0], DType.float32, device=device)
            beta = ops.constant([0.0, 0.0], DType.float32, device=device)
            graph.output(ops.layer_norm(x, gamma, beta, epsilon=1e-5))

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # Each row is normalized to zero mean and approximately unit variance.

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(result.to_numpy(), [[-1.0, 1.0]], atol=1e-3)

    Args:
        input: The tensor to normalize. Reduction runs over the last axis.
        gamma: The scale applied after normalization. A 1-D tensor whose
            length matches the last dimension of ``input``.
        beta: The bias added after scaling. A 1-D tensor with the same
            shape as ``gamma``.
        epsilon: A small positive constant added to the variance for
            numerical stability.

    Returns:
        A ``TensorValue`` with the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``gamma`` or ``beta`` does not match the last
            dimension of ``input``, or if ``epsilon`` is not positive.
    """
    if isinstance(gamma, TensorValue) and isinstance(
        input.shape[-1], StaticDim
    ):
        gamma_tensor = gamma

        # Check that gamma size matches the last dimension of input
        if gamma_tensor.shape[0] != input.shape[-1]:
            raise ValueError(
                f"Gamma size {gamma_tensor.shape[0]} does not match dimension"
                f" of reduction {input.shape[-1]}."
            )

    if isinstance(beta, TensorValue) and isinstance(input.shape[-1], StaticDim):
        beta_tensor = beta

        # Check that beta size matches the last dimension of input
        if beta_tensor.shape[0] != input.shape[-1]:
            raise ValueError(
                f"Beta size {beta_tensor.shape[0]} does not match dimension of"
                f" reduction {input.shape[-1]}."
            )

    # Check that epsilon is positive
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    input, gamma = dtype_promotion._promote_weak_dtypes(input, gamma)
    input, beta = dtype_promotion._promote_weak_dtypes(input, beta)
    return Graph.current._add_op_generated(
        mo.ReduceLayerNormOp,
        input._mlir_value.type,
        input,
        gamma,
        beta,
        constant(epsilon, DType.float32, DeviceRef.CPU()),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
