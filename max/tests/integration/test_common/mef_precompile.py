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
"""Reusable helpers for splitting graph compilation from execution in tests.

Graph-compile-heavy GPU integration tests spend most of their GPU-worker time
compiling, not executing. This module lets a test compile its graphs to MEF
artifacts on a CPU-only Bazel build action (no accelerator attached) and then
initialize + execute those artifacts on the GPU, so the scarce accelerator is
only held for the run.

The three roles a test wires together:

1. A **graph builder** ``build_graph(spec_name) -> Graph`` it supplies.
2. A **producer** binary that calls :func:`precompile_entrypoint` with that builder;
   the ``mef_precompile.bzl`` ``precompiled_mefs`` macro runs it as a CPU build
   action, once per spec, writing each compiled ``<spec>.mef`` to the output
   path the macro passes.
3. A **consumer** GPU test that resolves the produced MEF paths with
   :func:`mefs_from_env` and initializes each MEF with :func:`init_from_mef`.

See ``docs/internal/CompileOnCpuRunOnGpu.md`` for the full adoption guide.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

import click
from max.driver import (
    Accelerator,
    DLPackArray,
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.engine import InferenceSession, Model, read
from max.graph import Graph


def set_virtual_gpu(target: str, cpu_target: str, count: int = 1) -> None:
    """Configures virtual devices so a GPU-less worker can compile GPU graphs.

    Sets the four virtual-device knobs MAX reads when a device is created, so a
    later ``Accelerator()`` yields a virtual device of the requested arch and
    the graph compiler emits code for it without a physical GPU attached. Must
    be called before the first device is created (and before importing
    ``max._interpreter_ops``, which freezes its device set at import).

    Args:
        target: The GPU target as ``"api:arch"`` (e.g. ``"cuda:sm_100a"``).
        cpu_target: The host-CPU codegen descriptor (e.g.
            ``"triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3"``); part of the
            compile key, so it must match the consuming host.
        count: The number of virtual devices to present. Defaults to ``1``.

    Raises:
        ValueError: If ``target`` is not of the form ``"api:arch"``.
    """
    api, _, arch = target.partition(":")
    if not api or not arch:
        raise ValueError(f"--target must be 'api:arch', got {target!r}")
    set_virtual_cpu_target(cpu_target)
    set_virtual_device_api(api)
    set_virtual_device_target_arch(arch)
    set_virtual_device_count(count)


def compile_and_export_mef(
    session: InferenceSession, graph: Graph, mef_path: str | os.PathLike[str]
) -> None:
    """Compiles ``graph`` and writes the artifact to ``mef_path``.

    This is the device-independent half of a load: :meth:`compile` binds no
    weights and allocates no device memory, and
    :meth:`~max.engine.CompiledModel.export_mef` serializes straight from the
    compiled artifact, so it works under virtual devices with no GPU attached.

    Args:
        session: The session to compile with.
        graph: The graph to compile.
        mef_path: Filesystem path to write the ``.mef`` artifact to.
    """
    session.compile(graph).export_mef(os.fspath(mef_path))


def init_from_mef(
    session: InferenceSession,
    mef_path: str | os.PathLike[str],
    weights: Mapping[str, DLPackArray] | None = None,
) -> Model:
    """Reads the ``.mef`` at ``mef_path`` and initializes it for execution.

    This is the device-bound half of a load: it reloads the compiled artifact
    (no recompilation) and binds ``weights`` onto the session's device. Pairs
    with :func:`compile_and_export_mef` so compilation and execution can run on
    different machines.

    Args:
        session: The session to initialize the model on.
        mef_path: Filesystem path to a ``.mef`` written by
            :func:`compile_and_export_mef`.
        weights: The weights registry to bind, or ``None`` for a
            weightless graph.

    Returns:
        The initialized model, ready to execute.
    """
    return session.init(read(os.fspath(mef_path)), weights_registry=weights)


def mefs_from_env(rlocation_env_var: str) -> dict[str, Path]:
    """Resolves a ``$(rlocationpaths ...)`` env var to a ``{filename: path}`` map.

    The consumer test's Bazel target injects
    ``env = {VAR: "$(rlocationpaths :your_mefs)"}`` -- a space-separated list of
    the precompiled MEF files. This resolves each runfiles path (working-
    directory-independent) and keys the result by MEF filename (``<spec>.mef``).

    Args:
        rlocation_env_var: Name of the environment variable holding the
            space-separated runfiles paths of the precompiled MEF files.

    Returns:
        A mapping from each MEF's filename to its absolute path.
    """
    # Deferred import: the runfiles helper is only present in a bazel-run test's
    # runfiles tree, not in every context that imports this module.
    from python.runfiles import runfiles

    r = runfiles.Create()
    assert r is not None, "runfiles unavailable"
    mefs: dict[str, Path] = {}
    for rloc in os.environ[rlocation_env_var].split():
        resolved = r.Rlocation(rloc)
        assert resolved is not None, f"could not resolve runfiles path {rloc!r}"
        mefs[Path(resolved).name] = Path(resolved)
    return mefs


def precompile_entrypoint(build_graph: Callable[[str], Graph]) -> None:
    """Runs the CPU producer for one spec: compile ``build_graph(spec)`` to a MEF.

    Wires the standard producer command (``--target``, ``--cpu-target``,
    ``--spec``, ``--out``) that the ``precompiled_mefs`` Bazel macro invokes as
    a build action. It sets the virtual-device knobs, builds the spec's graph,
    and exports the compiled MEF to ``--out``. Call it from a producer binary's
    ``__main__``::

        from test_common.mef_precompile import precompile_entrypoint
        from my_graphs import build_graph

        if __name__ == "__main__":
            precompile_entrypoint(build_graph)

    ``build_graph`` and its module must not create a device or read
    ``accelerator_count()`` at import time, since the knobs are set after import.

    Args:
        build_graph: Maps a spec name to the :class:`Graph` to compile.
    """

    @click.command()
    @click.option("--target", required=True, help="GPU target 'api:arch'.")
    @click.option(
        "--cpu-target", required=True, help="Host-CPU codegen descriptor."
    )
    @click.option("--spec", "spec_name", required=True, help="Spec to compile.")
    @click.option(
        "--out", "out_path", required=True, help="Path to write the MEF to."
    )
    def _main(
        target: str, cpu_target: str, spec_name: str, out_path: str
    ) -> None:
        # Set the virtual-device knobs before Accelerator() creates the device.
        set_virtual_gpu(target, cpu_target)
        compile_and_export_mef(
            InferenceSession(devices=[Accelerator()]),
            build_graph(spec_name),
            out_path,
        )

    _main()
