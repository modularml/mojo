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
"""Shared graph construction for the Hadamard transform CPU/GPU split.

The CPU producer (``precompile_hadamard``) and the GPU consumer
(``test_hadamard_transform``) both import this module so they build the
*identical* graph -- the producer compiles it to a MEF with no GPU, the consumer
initializes that MEF and executes it. Keeping the construction in one place is
what guarantees the compiled artifact and the runtime inputs can't drift apart.

The transform bakes its Hadamard matrix in as a graph constant, so the compiled
graphs are weightless: the consumer initializes them with no weights registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.deepseekV3_2.layers import HadamardTransform

DTYPE = DType.bfloat16


@dataclass(frozen=True)
class HadamardSpec:
    """One compiled-graph parametrization of the Hadamard transform."""

    name: str
    """Stable identifier used for the MEF filename and the test id."""

    shape: tuple[int, ...]
    """Static input shape. The last dim drives the (padded) transform size."""

    scale: float
    """Scale factor applied after the transform."""


HADAMARD_SPECS: tuple[HadamardSpec, ...] = (
    HadamardSpec("hadamard_1x2_s1", (1, 2), 1.0),
    # Last dim is not a power of 2, exercising the pad/slice path.
    HadamardSpec("hadamard_2x6_s0p5", (2, 6), 0.5),
    HadamardSpec("hadamard_2x1x3_s1", (2, 1, 3), 1.0),
)

SPECS_BY_NAME: dict[str, HadamardSpec] = {s.name: s for s in HADAMARD_SPECS}


def build_hadamard_graph(spec: HadamardSpec) -> Graph:
    """Builds the Hadamard transform graph for ``spec`` (device-independent).

    This is the single construction path the CPU producer compiles and the GPU
    consumer initializes.

    Args:
        spec: The parametrization to build.

    Returns:
        The constructed :class:`Graph`, ready to compile.
    """
    with Graph(
        spec.name,
        input_types=[
            TensorType(dtype=DTYPE, shape=spec.shape, device=DeviceRef.GPU())
        ],
    ) as graph:
        graph.output(HadamardTransform(spec.scale)(graph.inputs[0].tensor))
    return graph
