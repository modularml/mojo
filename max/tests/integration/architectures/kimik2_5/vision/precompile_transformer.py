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
"""CPU producer: compile the vision transformer graph to a MEF with no GPU.

Run as a Bazel build action by the ``precompiled_mefs`` macro (see the package
``BUILD.bazel``). All the compile/export/virtual-device machinery lives in the
shared :func:`~test_common.mef_precompile.precompile_entrypoint`; this file only
names the graph builder to use (:func:`build_transformer_graph`).
"""

from __future__ import annotations

from _transformer_graphs import TRANSFORMER_SPEC, build_transformer_graph
from max.graph import Graph
from test_common.mef_precompile import precompile_entrypoint


def _build(spec_name: str) -> Graph:
    if spec_name != TRANSFORMER_SPEC:
        raise SystemExit(
            f"unknown spec {spec_name!r}; known: [{TRANSFORMER_SPEC!r}]"
        )
    return build_transformer_graph()


if __name__ == "__main__":
    precompile_entrypoint(_build)
