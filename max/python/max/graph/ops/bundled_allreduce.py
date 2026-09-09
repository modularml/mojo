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
"""Op implementation for bundled allreduce using mo.parallel."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo

from ..graph import Graph
from ..type import _ChainType
from ..value import (
    BufferValue,
    BufferValueLike,
    TensorValue,
    TensorValueLike,
    _ChainValue,
)
from .parallel import parallel
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike], signal_buffers: Iterable[BufferValueLike]
) -> list[TensorValue]:
    """Bundled allreduce sum using mo.parallel with per-device dispatch.

    Stages an ``mo.parallel`` region containing ``mo.bundled.expand`` and
    ``mo.bundled.allreduce.sum``, so the compiler can lower each device
    launch independently (async dispatch).

    Args:
        inputs: The input tensors to reduce, one per device.
        signal_buffers: Device buffer values used for synchronization,
            one per device (must match length of ``inputs``).

    Returns:
        A list of reduced tensors, one per device.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    devices = [inp.device for inp in inputs]
    graph = Graph.current

    in_chain = graph.device_chains.merge_for(devices)

    input_types = [inp.type for inp in inputs]

    def body_fn(
        tensor: TensorValue, _: BufferValue, body_chain: _ChainValue
    ) -> tuple[TensorValue, _ChainValue]:
        peers = graph._add_op_generated(mo.BundledExpandOp, input_types, tensor)
        out, out_chain = graph._add_op_generated(
            mo.BundledAllreduceSumOp,
            input_types[0],
            _ChainType(),
            peers,
            signal_buffers,
            body_chain,
        )
        return out.tensor, _ChainValue(out_chain._mlir_value)

    result = parallel(
        [inputs],
        body_fn,
        buffers=signal_buffers,
        chain=in_chain,
        result_types=[input_types],
    )
    assert isinstance(result, tuple)
    bundles, out_chain = result
    [results] = bundles

    graph._update_chain(out_chain)
    for d in devices:
        graph.device_chains[d] = out_chain

    return results
