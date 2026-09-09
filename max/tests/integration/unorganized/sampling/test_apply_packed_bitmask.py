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
"""End-to-end graph tests for the fused ``apply_packed_bitmask`` op.

Builds a tiny graph that calls :func:`apply_packed_bitmask` and runs it on the
GPU, comparing against a numpy unpack + ``where`` reference. Covers both the
rank-2 (token sampler) and rank-3 (speculative-decode acceptance sampler) paths.
"""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import apply_packed_bitmask
from max.support.math import ceildiv

_FILL = -10000.0  # Value used for masked-out logits


def _packed_to_bool(packed: np.ndarray, vocab_size: int) -> np.ndarray:
    """Reference unpack of a packed int32 bitmask to a bool mask."""
    bits = 2 ** np.arange(32, dtype=np.int32)
    unpacked = (packed[..., np.newaxis] & bits) != 0
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)
    return unpacked[..., :vocab_size]


def _random_inputs(
    shape: tuple[int, ...], vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(shape, dtype=np.float32)
    packed_vocab = ceildiv(vocab_size, 32)
    packed = rng.integers(
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        size=(*shape[:-1], packed_vocab),
        dtype=np.int32,
    )
    return logits, packed


@pytest.mark.parametrize(
    "shape",
    [
        (3, 40),  # rank-2: token sampler ([batch, vocab])
        (2, 3, 40),  # rank-3: acceptance sampler ([batch, num_pos, vocab])
        (2, 1, 64),  # vocab a multiple of 32 (no trailing padding)
    ],
)
def test_apply_packed_bitmask(
    session: InferenceSession, shape: tuple[int, ...]
) -> None:
    device = session.devices[0]
    gpu = DeviceRef.GPU()
    vocab_size = shape[-1]
    packed_vocab = ceildiv(vocab_size, 32)

    logits_np, packed_np = _random_inputs(shape, vocab_size)

    logits_type = TensorType(DType.float32, list(shape), device=gpu)
    packed_type = TensorType(
        DType.int32, [*shape[:-1], packed_vocab], device=gpu
    )
    with Graph(
        "apply_packed_bitmask", input_types=[logits_type, packed_type]
    ) as graph:
        logits, packed = (v.tensor for v in graph.inputs)
        graph.output(apply_packed_bitmask(logits, packed, fill_val=_FILL))

    model = session.load(graph)
    out = model(
        Buffer.from_numpy(logits_np).to(device),
        Buffer.from_numpy(packed_np).to(device),
    )[0]
    assert isinstance(out, Buffer)
    actual = out.to_numpy()

    keep = _packed_to_bool(packed_np, vocab_size)
    expected = np.where(keep, logits_np, np.float32(_FILL))
    np.testing.assert_array_equal(actual, expected)
