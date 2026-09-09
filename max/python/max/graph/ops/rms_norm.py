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
"""Op implementation for rms_norm."""

from max._core.dialects import builtin, kgen, mo
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def rms_norm(
    input: TensorValueLike,
    weight: TensorValueLike,
    epsilon: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
) -> TensorValue:
    """Computes root mean square normalization over the last dimension of ``input``.

    The output is ``input / rms(input) * (weight + weight_offset)`` where
    ``rms(x) = sqrt(mean(x ** 2) + epsilon)``. Reduction runs over the last
    axis of ``input`` and is broadcast back across the leading axes. See
    `Root Mean Square Layer Normalization
    <https://arxiv.org/abs/1910.07467>`_ for the original formulation.

    Two variants are supported through ``weight_offset`` and
    ``multiply_before_cast``:

    - **Llama-style** (default): ``weight_offset=0`` and
      ``multiply_before_cast=False``. The normalized input is cast to the
      output dtype before multiplication by the weight.
    - **Gemma-style**: ``weight_offset=1`` and ``multiply_before_cast=True``.
      The weight is treated as ``1 + weight`` and multiplication runs in
      the reduction dtype before casting back.

    .. code-block:: python

        import numpy as np

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("rms_norm_example") as graph:
            x = ops.constant([[3.0, 4.0]], DType.float32, device=device)
            weight = ops.constant([1.0, 1.0], DType.float32, device=device)
            y_llama = ops.rms_norm(x, weight, epsilon=1e-6)
            y_gemma = ops.rms_norm(
                x, weight, epsilon=1e-6,
                weight_offset=1.0, multiply_before_cast=True,
            )
            graph.output(y_llama, y_gemma)

        model = InferenceSession().load(graph)
        llama, gemma = model.execute()
        assert np.allclose(llama.to_numpy(), [[0.848528, 1.131371]], atol=1e-4)
        # weight_offset adds 1.0 to the weight, doubling the result here.

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(gemma.to_numpy(), [[1.697056, 2.262742]], atol=1e-4)

    Args:
        input: The tensor to normalize. Reduction runs over the last axis.
        weight: The scale applied after normalization. A 1-D tensor whose
            shape matches the last dimension of ``input``.
        epsilon: A small positive constant added to the mean of squares for
            numerical stability.
        weight_offset: A value added to ``weight`` before scaling. Use
            ``1.0`` for Gemma-style normalization and ``0.0`` otherwise.
            Defaults to ``0.0``.
        multiply_before_cast: Whether to multiply by the (offset) weight
            before casting the normalized input back to the output dtype.
            Llama-style sets this to ``False``. Defaults to ``False``.

    Returns:
        A ``TensorValue`` with the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``weight`` does not match the last dimension of
            ``input``.
    """
    input = TensorValue(input)
    weight = TensorValue(weight)

    if input.shape[-1:] != weight.shape:
        raise ValueError(
            f"RMSNorm: Could not apply weight shape {weight.shape} to input"
            f" shape {input.shape}, weight shape must match the final input"
            " dimension."
        )

    return Graph.current._add_op_generated(
        mo.ReduceRmsNormOp,
        result=TensorType(
            dtype=input.dtype, shape=input.shape, device=input.device
        ),
        input=input,
        weight=weight,
        epsilon=constant(epsilon, DType.float32, DeviceRef.CPU()),
        weight_offset=constant(weight_offset, input.dtype, DeviceRef.CPU()),
        multiply_before_cast=builtin.BoolAttr(multiply_before_cast),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
