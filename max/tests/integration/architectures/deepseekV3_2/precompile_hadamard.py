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
"""CPU producer: compile one Hadamard transform graph to a MEF with no GPU.

Run as a Bazel build action by the ``precompiled_mefs`` macro (see the package
``BUILD.bazel``). All the compile/export/virtual-device machinery lives in the
shared :func:`~test_common.mef_precompile.precompile_entrypoint`; this file only
names the graph builder to use (:func:`build_hadamard_graph`, keyed by spec
name).
"""

from __future__ import annotations

from _hadamard_graphs import SPECS_BY_NAME, build_hadamard_graph
from max.graph import Graph
from test_common.mef_precompile import precompile_entrypoint


def _build(spec_name: str) -> Graph:
    spec = SPECS_BY_NAME.get(spec_name)
    if spec is None:
        raise SystemExit(
            f"unknown spec {spec_name!r}; known: {sorted(SPECS_BY_NAME)}"
        )
    return build_hadamard_graph(spec)


if __name__ == "__main__":
    precompile_entrypoint(_build)
