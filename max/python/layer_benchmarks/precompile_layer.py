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
"""CPU producer: compile one layer spec's graph to a MEF with no GPU.

Run as a Bazel build action by the ``precompiled_mefs`` macro (see this
package's ``BUILD.bazel``); the harness tests then initialize the artifacts
instead of compiling. The compile/export/virtual-device machinery lives in the
shared :func:`~test_common.mef_precompile.precompile_entrypoint`, so this file
only maps a spec name to its graph.
"""

from __future__ import annotations

from max.graph import Graph
from test_common.mef_precompile import precompile_entrypoint
from testbed.runner import create_session
from testbed.specs import SPECS


def _build(spec_name: str) -> Graph:
    spec = SPECS.get(spec_name)
    if spec is None:
        raise SystemExit(f"unknown spec {spec_name!r}; known: {sorted(SPECS)}")
    # A session only because the harness signature takes one; graph building
    # never touches it, and the compile runs on the entrypoint's own session.
    session, device = create_session()
    harness = spec.harness_type(spec.static_params, session, device)
    graph, _ = harness.build_graph()
    return graph


if __name__ == "__main__":
    precompile_entrypoint(_build)
