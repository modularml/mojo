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
"""Op implementation for argsort."""

from max.dtype import DType

from ..type import TensorType
from ..value import StrongTensorValueLike, TensorValue
from .custom import custom


def argsort(x: StrongTensorValueLike, ascending: bool = True) -> TensorValue:
    """Returns the indices that would sort a rank-1 tensor.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("argsort") as graph:
            x = ops.constant([3.0, 1.0, 2.0], DType.float32, device=device)
            # Ascending order visits 1, 2, 3, so the indices are [1, 2, 0].
            graph.output(ops.argsort(x, ascending=True))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor to sort. Must be rank 1.
        ascending: Whether to sort in ascending order. If ``False``, sorts in
            descending order. Defaults to ``True``.

    Returns:
        A ``TensorValue`` representing the sorting indices, with the same shape
        as ``x`` and ``int64`` dtype.

    Raises:
        ValueError: If ``x`` is not rank 1.
    """
    x = TensorValue(x)
    if x.rank != 1:
        raise ValueError("argsort only implemented for input tensors of rank 1")
    return custom(
        "mx.argsort",
        x.device,
        [x],
        out_types=[
            TensorType(dtype=DType.int64, shape=x.shape, device=x.device)
        ],
        parameters={
            "ascending": ascending,
        },
    )[0].tensor
