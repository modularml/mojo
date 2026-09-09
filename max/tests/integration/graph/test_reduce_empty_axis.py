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
"""Reductions over a zero-extent axis, end to end through the graph.

A reduce whose axis is empty still owns one output per row -- the monoid
identity -- and so does anything fused into its epilogue. Both used to be
skipped entirely, leaving whatever the output buffer already held: a stale
result from the previous execution on GPU, uninitialized memory on CPU.

The kernel-level regressions live in
``max/kernels/test/{algorithm,gpu/algorithm}/test_reduce_empty_axis.mojo``.
This covers the shape the defect was reported at: a real reduction, then an
empty one, alternating on the same compiled model so a stale value is as
visible as an unwritten one.
"""

import numpy as np
import pytest
import torch
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()

# A reduce over `dim` alternates between these; every empty run is preceded
# by a real one, so the output buffer always holds a distinguishable value
# that the empty run has to overwrite.
_DIMS = [3, 0, 3, 0]

# The fused comparison's threshold. `sum` over three ones clears it; the
# identity `0` an empty axis produces does not, so the fused output flips
# 1.0 -> 0.0 and a skipped epilogue shows up as a stuck 1.0.
_THRESHOLD = 0.5


@pytest.fixture(scope="module")
def reduce_model(session: InferenceSession) -> Model:
    """A sum over a symbolic axis, plus a comparison fused into its
    epilogue."""
    input_type = TensorType(DType.float32, ["batch", "dim"], device=device_ref)
    with Graph("reduce_empty_axis", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        reduced = ops.sum(x, axis=1)
        gated = (reduced > _THRESHOLD).cast(DType.float32)
        graph.output(ops.squeeze(reduced, axis=1), ops.squeeze(gated, axis=1))
    return session.load(graph)


def test_reduce_empty_axis_writes_identity(reduce_model: Model) -> None:
    batch = 2

    for dim in _DIMS:
        input_data = torch.ones((batch, dim), dtype=torch.float32)
        max_input = Buffer.from_dlpack(input_data).to(
            reduce_model.input_devices[0]
        )
        reduced, gated = reduce_model(max_input)
        assert isinstance(reduced, Buffer)
        assert isinstance(gated, Buffer)

        # `numpy.sum` over an empty axis is the additive identity, 0.
        expected = np.full(batch, float(dim), dtype=np.float32)
        # The fused consumer first: it is the half that used to go stale
        # without leaving any other trace.
        np.testing.assert_array_equal(
            gated.to_numpy(),
            (expected > _THRESHOLD).astype(np.float32),
            err_msg=f"fused comparison at dim={dim}",
        )
        np.testing.assert_array_equal(
            reduced.to_numpy(), expected, err_msg=f"sum at dim={dim}"
        )
