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
"""Op implementation for chunk."""

from ..value import TensorValue, TensorValueLike
from .slice_tensor import slice_tensor
from .validation import assert_valid_axis


def chunk(x: TensorValueLike, chunks: int, axis: int = 0) -> list[TensorValue]:
    """Splits a tensor into equal-sized chunks along an axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("chunk_example") as graph:
            x = ops.constant([1, 2, 3, 4, 5, 6], DType.int32, device=device)

            # Split into three equal chunks along axis 0
            chunks = ops.chunk(x, 3, axis=0)  # [1, 2], [3, 4], and [5, 6]
            graph.output(*chunks)

        model = InferenceSession().load(graph)
        a, b, c = model.execute()

    Args:
        x: The tensor to chunk.
        chunks: The number of chunks. Must be a positive integer that evenly
            divides the size of ``x`` along ``axis``.
        axis: The axis to split along. Defaults to ``0``.

    Returns:
        A list of ``TensorValue`` objects (chunks), each the same size along ``axis``.

    Raises:
        ValueError: If ``chunks`` does not evenly divide the size of ``x`` along
            ``axis``, or if ``x`` is a scalar and ``chunks`` is greater than ``1``.
        IndexError: If ``axis`` is out of range for the rank of ``x``.
    """
    # TODO(GEX-1943): once we have control flow in the graph, this can be updated to
    # dynamic chunk counts while still supporting algebraic dims. For now,
    # this will generate exactly chunks or fail.
    x = TensorValue(x)

    if x.rank == 0 and chunks > 1:
        raise ValueError(f"Cannot split scalar value into {chunks=}")

    assert_valid_axis(x, axis)

    if axis < 0:
        axis = x.rank + axis

    # Convert to a python bigint for int math
    n = int(x.shape[axis])

    if n % chunks != 0:
        raise ValueError(
            f"chunk: {chunks=} must statically divide {x.shape[axis]=}"
        )

    # Determine chunk size using ceiling division.
    chunk_size = n // chunks

    def slices(offset: int) -> list[slice]:
        slices = [slice(None)] * x.rank
        slices[axis] = slice(chunk_size * offset, chunk_size * (offset + 1))
        return slices

    return [slice_tensor(x, slices(offset)) for offset in range(chunks)]
