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
"""Op implementation for group_norm."""

from max._core.dialects import kgen, mo
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def group_norm(
    input: TensorValueLike,
    gamma: TensorValueLike,
    beta: TensorValueLike,
    num_groups: int,
    epsilon: float,
) -> TensorValue:
    """Computes group normalization over the channel axis of ``input``.

    Splits the channel axis (axis 1) of ``input`` into ``num_groups``
    groups, computes the mean and variance within each group, and
    normalizes. ``gamma`` and ``beta`` then apply a per-channel affine
    transform. Useful when the batch axis is small enough that batch
    normalization is unstable.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("group_norm_example") as graph:
            # Shape (batch=1, channels=4, spatial=1, 1); 2 groups of 2 channels.
            x = ops.constant(
                [[[[1.0]], [[3.0]], [[1.0]], [[3.0]]]],
                DType.float32,
                device=device,
            )
            gamma = ops.constant([1.0, 1.0, 1.0, 1.0], DType.float32, device=device)
            beta = ops.constant([0.0, 0.0, 0.0, 0.0], DType.float32, device=device)
            graph.output(
                ops.group_norm(x, gamma, beta, num_groups=2, epsilon=1e-5)
            )

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # Each group (channels 0-1 and 2-3) is normalized to zero mean and
        # approximately unit variance.

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(
            result.to_numpy(), [[[[-1.0]], [[1.0]], [[-1.0]], [[1.0]]]], atol=1e-3
        )

    Args:
        input: The tensor to normalize, of shape
            ``(batch, channels, ...)``.
        gamma: The per-channel scale applied after normalization. A 1-D
            tensor whose length matches the channel axis of ``input``.
        beta: The per-channel bias added after scaling. A 1-D tensor with
            the same shape as ``gamma``.
        num_groups: The number of groups to split the channel axis into.
            Must divide the channel size evenly.
        epsilon: A small positive constant added to the variance for
            numerical stability.

    Returns:
        A ``TensorValue`` with the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``input`` has fewer than 2 dimensions.
    """
    input = TensorValue(input)
    gamma = TensorValue(gamma)
    beta = TensorValue(beta)

    if len(input.shape) < 2:
        raise ValueError(
            "Expected input tensor with >=2 dimensions, got shape"
            f" {input.shape}"
        )

    return Graph.current._add_op_generated(
        mo.ReduceGroupNormOp,
        result=TensorType(
            dtype=input.dtype, shape=input.shape, device=input.device
        ),
        input=input,
        gamma=gamma,
        beta=beta,
        epsilon=constant(epsilon, DType.float32, DeviceRef.CPU()),
        num_groups=constant(num_groups, DType.int32, DeviceRef.CPU()),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
