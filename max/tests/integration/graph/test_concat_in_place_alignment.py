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

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, ops

_POOL_ROWS = 64
_CTX_LEN = 2


@pytest.mark.skipif(accelerator_count() == 0, reason="requires an accelerator")
@pytest.mark.parametrize("n", [11, 12])
def test_concat_operand_producer_keeps_vector_alignment(
    session: InferenceSession, n: int
) -> None:
    """A concat operand produced by a vectorizing kernel stays 16-byte aligned.

    `ConcatInPlacePass` used to hand each operand a slice of the concat buffer
    at the running sum of the preceding operand sizes. With an int64 operand of
    odd length that offset is 8 mod 16, while the gather producing the second
    operand issues a 16-byte vector store against the alignment every kernel
    tensor is told its base has -- so every odd `n` faulted with
    CUDA_ERROR_MISALIGNED_ADDRESS and every even `n` passed.
    """
    device = DeviceRef.from_device(session.devices[0])
    with Graph(
        "concat_in_place_alignment",
        input_types=[
            TensorType(DType.int64, [_POOL_ROWS, _CTX_LEN], device=device),
            TensorType(DType.int32, [Dim("batch")], device=device),
            TensorType(DType.int32, [Dim("n")], device=device),
        ],
    ) as graph:
        pool, slots, raw_ids = graph.inputs
        context = ops.gather(pool.tensor, slots.tensor, axis=0)
        graph.output(
            ops.concat(
                [
                    ops.cast(raw_ids.tensor, DType.int64),
                    ops.reshape(context, (-1,)),
                ],
                axis=0,
            )
        )

    model = session.load(graph)
    pool_data = np.arange(_POOL_ROWS * _CTX_LEN, dtype=np.int64).reshape(
        _POOL_ROWS, _CTX_LEN
    )
    actual = model(
        Buffer.from_numpy(pool_data).to(model.input_devices[0]),
        Buffer.from_numpy(np.array([3], dtype=np.int32)).to(
            model.input_devices[1]
        ),
        Buffer.from_numpy(np.arange(n, dtype=np.int32)).to(
            model.input_devices[2]
        ),
    )[0]
    assert isinstance(actual, Buffer)
    session.devices[0].synchronize()

    expected = np.concatenate([np.arange(n, dtype=np.int64), pool_data[3]])
    np.testing.assert_equal(actual.to_numpy(), expected)
