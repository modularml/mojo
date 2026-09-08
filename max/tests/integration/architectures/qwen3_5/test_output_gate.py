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

"""Gated DeltaNet output-gate selection.

Qwen3.5 says ``swish``, which is ``silu``, so the hardcoded ``silu`` this
replaces was accidentally right. Qwen3.8-Flash-Next says ``sigmoid``, where it
is silently wrong: the model builds, runs, and answers plausibly. Nothing
downstream fails, so the selection needs pinning here -- both that a named
activation is the one that runs, and that a name the layer cannot honor is
refused rather than defaulted.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.pipelines.architectures.qwen3_5.layers.gated_deltanet import (
    _OUTPUT_GATE_ACTIVATIONS,
    GatedDeltaNet,
)

DEVICES = [DeviceRef.GPU(0), DeviceRef.GPU(1)]


def _gated_deltanet(**kwargs) -> GatedDeltaNet:
    return GatedDeltaNet(
        hidden_size=5120,
        num_key_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_size=4,
        dtype=DType.bfloat16,
        device=DEVICES[0],
        **kwargs,
    )


def _gate_mlir(output_gate_type: str) -> str:
    """The MLIR of the activation the forward pass reads out of the table.

    ``__call__`` applies ``_OUTPUT_GATE_ACTIVATIONS[self.output_gate_type]`` at
    one site, and reaching it needs the state pools and the recurrence kernel,
    so the table entry is built here on its own.
    """
    with Graph(
        "output_gate",
        input_types=[
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        graph.output(
            _OUTPUT_GATE_ACTIVATIONS[output_gate_type](graph.inputs[0].tensor)
        )
    return str(graph._mlir_op)


def test_sigmoid_gate_is_not_the_swish_gate() -> None:
    """The two are numerically different and both are plausible outputs."""
    sigmoid_mlir = _gate_mlir("sigmoid")
    assert "sigmoid" in sigmoid_mlir
    assert "silu" not in sigmoid_mlir

    assert "silu" in _gate_mlir("swish")


def test_swish_and_silu_name_the_same_activation() -> None:
    """Qwen3.5 writes ``swish`` where the stdlib writes ``silu``."""
    assert _gate_mlir("swish") == _gate_mlir("silu")


def test_unknown_output_gate_type_is_refused() -> None:
    """A default would run the wrong activation and report nothing."""
    with pytest.raises(ValueError, match="unsupported output_gate_type"):
        _gated_deltanet(output_gate_type="gelu")


def test_default_output_gate_is_the_qwen3_5_one() -> None:
    assert _gated_deltanet().output_gate_type == "swish"


def test_shards_keep_the_output_gate() -> None:
    """A shard rebuilds the layer from scratch, so the choice has to travel.

    Losing it leaves every tensor-parallel device on the ``swish`` default
    while the single-device graph gates correctly.
    """
    layer = _gated_deltanet(output_gate_type="sigmoid")
    layer.sharding_strategy = ShardingStrategy.tensor_parallel(len(DEVICES))

    shards = layer.shard(DEVICES)

    assert [shard.output_gate_type for shard in shards] == ["sigmoid"] * len(
        DEVICES
    )
