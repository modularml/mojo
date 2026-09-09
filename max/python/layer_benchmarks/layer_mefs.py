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

"""GPU-side entry point for the harness tests: run a CPU-precompiled graph.

Each harness test builds its runner through :func:`create_runner`, which
initializes the MEF that ``precompile_layer`` compiled on a CPU build action
rather than compiling on the GPU worker (see
``docs/internal/CompileOnCpuRunOnGpu.md``). The test's Bazel target passes the
artifacts in ``LAYER_BENCHMARK_MEFS``; without that variable -- running under
kbench, or pytest outside Bazel -- the graph is compiled in-process as before.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from max.driver import DLPackArray
from max.engine import InferenceSession, Model
from test_common.mef_precompile import init_from_mef, mefs_from_env
from testbed.harness import ContextT, DynamicParamsT, StaticParamsT
from testbed.runner import LayerTestRunner, ModelInitializer, create_session
from testbed.specs import LayerSpec

MEF_ENV_VAR = "LAYER_BENCHMARK_MEFS"
"""Runfiles paths of the precompiled MEFs, injected by the test's Bazel target."""


def _mef_initializer(
    spec: LayerSpec[StaticParamsT, DynamicParamsT, ContextT],
    static_params: StaticParamsT,
) -> ModelInitializer | None:
    """Resolve ``spec``'s precompiled MEF, or ``None`` to compile in-process.

    Falls back to compiling when no artifacts were wired in, and when the
    params differ from the spec's -- an env-var-overridden config (see
    ``test_harness_text_encoder``) is a different graph than the one the build
    action compiled. A missing artifact for a spec that *does* match is an
    error rather than a silent recompile.
    """
    if MEF_ENV_VAR not in os.environ:
        return None
    if static_params != spec.static_params:
        return None

    mefs = mefs_from_env(MEF_ENV_VAR)
    mef_path = mefs.get(f"{spec.name}.mef")
    assert mef_path is not None, (
        f"no precompiled MEF for spec {spec.name!r} in {MEF_ENV_VAR}"
        f" (found {sorted(mefs)}); check the target's precompiled_mefs specs"
    )

    def initialize(
        session: InferenceSession, weights: Mapping[str, DLPackArray]
    ) -> Model:
        return init_from_mef(session, mef_path, weights)

    return initialize


def create_runner(
    spec: LayerSpec[StaticParamsT, DynamicParamsT, ContextT],
    static_params: StaticParamsT | None = None,
    num_devices: int = 1,
) -> LayerTestRunner[StaticParamsT, DynamicParamsT, ContextT]:
    """Create a runner for ``spec``, reusing its CPU-precompiled MEF.

    Args:
        spec: The layer spec to run.
        static_params: Params to build the harness with, overriding the spec's.
            Only for tests whose config comes from the environment; a value
            that differs from the spec's compiles the graph in-process.
        num_devices: Number of GPUs the harness needs.

    Returns:
        A runner over a freshly constructed harness.
    """
    params = spec.static_params if static_params is None else static_params
    session, device = create_session(num_devices)
    harness = spec.harness_type(params, session, device)
    return LayerTestRunner(harness, _mef_initializer(spec, params))
