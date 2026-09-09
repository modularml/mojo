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
"""Layering guard: importing an architecture must not import ``max.serve``.

``register_all_models()`` imports each architecture module, and an
architecture may declare a ``cascade_pipeline_factory`` that references a
cascade pipeline class (e.g. ``llama3`` -> ``CommonTextGenPipeline``, which
imports ``MAXModelWorker``). Those references are fine, but the import chain
must *dead-end* before ``max.serve``. When ``MAXModelWorker`` imported
``max.serve.pipelines.model_worker`` at module scope, importing an architecture
dragged in the serve stack, so resolving an architecture while ``model_worker``
was still initializing failed with a partially-initialized-module
``ImportError`` that broke serving.

This test imports the Llama architecture -- the one that carries a
``cascade_pipeline_factory`` -- and asserts no ``max.serve.*`` module was
pulled in. It is deliberately broad: it catches *any* architecture whose import
reaches the serve layer, not just the one symbol from the original bug.
"""

from __future__ import annotations

import importlib
import sys

# The module register_all_models() loads when the Llama architecture is
# resolved; it pulls in the architecture's cascade_pipeline_factory
# (CommonTextGenPipeline). Importing it is the behavior under test.
_ARCH_MODULE = "max.pipelines.architectures.llama3.arch"

# The serving layer sits above the architecture registry; importing an
# architecture must never drag it in.
_SERVE_LAYER = "max.serve"


def test_importing_architecture_does_not_import_serve() -> None:
    """Assert importing an architecture module pulls in no ``max.serve`` module."""
    # Guard so the check is meaningful: if the module were already imported,
    # import_module() would be a no-op and a leak would slip through.
    assert _ARCH_MODULE not in sys.modules

    # The import is the code under test: a reintroduced
    # arch -> cascade -> max.serve cycle raises ImportError here (a failure of
    # this test, not a collection error during setup), and a clean import must
    # not have pulled the serve layer in.
    importlib.import_module(_ARCH_MODULE)

    leaked = sorted(
        name
        for name in sys.modules
        if name == _SERVE_LAYER or name.startswith(_SERVE_LAYER + ".")
    )
    assert not leaked, (
        "Importing an architecture module pulled in the max.serve layer, "
        "reintroducing the arch -> cascade -> max.serve import cycle that broke "
        "serving. Keep MAXModelWorker's max.serve imports lazy (inside "
        "__init__/open(), not at module scope). Leaked modules: "
        + ", ".join(leaked)
    )
