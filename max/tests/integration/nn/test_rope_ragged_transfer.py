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
"""Graph-construction regression tests for ``rope_ragged`` device handling.

``rope_ragged`` runs the rope kernel on the input tensor's device (a GPU).
When ``freqs_cis`` is resident on a different device -- commonly a CPU-side
frequency table that the caller sliced (``freqs_cis[:seq_len]``) before
passing it in -- the wrapper must insert an explicit device transfer.
Otherwise the graph compiler fuses the CPU view (the ``mo.slice``) directly
into the GPU consumer, and that fused view races on the implicit transfer's
lifetime, reading out of bounds under host-side timing jitter.

These tests only build the graph (no execution), so they run deterministically
on any host and act as a stable regression gate for the transfer insertion.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import rope_ragged

_N_TOKENS = 8
_N_HEADS = 4
_ROPE_DIM = 64
_MAX_SEQ_LEN = 1024


def _build_rope_ragged_graph(freqs_cis_device: DeviceRef) -> Graph:
    """Builds a minimal rope_ragged graph with freqs_cis on ``freqs_cis_device``.

    The input/row-offsets/start-pos all live on GPU; only ``freqs_cis`` device
    varies. ``freqs_cis`` is a device-resident graph input (not a constant) so
    that the *only* device transfer the graph can contain is the one
    ``rope_ragged`` inserts -- constant placement would otherwise emit its own
    host-to-device transfer and mask the behavior under test. It is sliced
    first, mirroring the DeepseekV3.2 indexer path that surfaced the race.
    """
    # NOTE: the graph name must not contain the substring "transfer"; the
    # assertions below check for a transfer op via ``"transfer" in str(graph)``.
    with Graph(
        "rope_ragged_freqs_cis_device",
        input_types=[
            TensorType(
                DType.bfloat16,
                [_N_TOKENS, _N_HEADS, _ROPE_DIM],
                DeviceRef.GPU(),
            ),
            TensorType(DType.uint32, ["batch_plus_one"], DeviceRef.GPU()),
            TensorType(DType.uint32, ["batch"], DeviceRef.GPU()),
            TensorType(
                DType.float32, [_MAX_SEQ_LEN, _ROPE_DIM], freqs_cis_device
            ),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        input_row_offsets = graph.inputs[1].tensor
        start_pos = graph.inputs[2].tensor
        # Slice on whichever device freqs_cis lives on -- the CPU slice is the
        # pattern that triggered the fused-kernel OOB.
        freqs_cis = graph.inputs[3].tensor[:_N_TOKENS]

        out = rope_ragged(
            x,
            input_row_offsets,
            start_pos,
            freqs_cis,
            interleaved=False,
        )
        graph.output(out)
    return graph


def test_rope_ragged_transfers_cpu_freqs_cis_to_gpu() -> None:
    """A CPU-resident freqs_cis must be transferred onto the kernel's device.

    Without the transfer the compiler fuses the CPU ``mo.slice`` straight into
    the GPU rope kernel, which races and reads out of bounds.
    """
    graph = _build_rope_ragged_graph(DeviceRef.CPU())
    assert "transfer" in str(graph)


def test_rope_ragged_no_transfer_when_freqs_cis_on_gpu() -> None:
    """freqs_cis already on the kernel device must not add a transfer.

    This keeps the production path (freqs_cis pre-moved to GPU) transfer-free.
    """
    graph = _build_rope_ragged_graph(DeviceRef.GPU())
    assert "transfer" not in str(graph)
