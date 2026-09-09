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
"""Consumer: adopt the eager-GC-sweep warm cache on a REAL GPU.

The `compose` build action (see `interp_cache.bzl`) merges the lane-shared
`warm_cpu` warm with this lane's `warm_accelerators` warm (a VIRTUAL x86 host +
the CI lane's GPU arch, e.g. `sm_100a`, warmed with no physical GPU) into one
dir with one MEF per device slot for `WARM_DEVICE_COUNT` accelerators + the CPU
slot. This test runs on a real x86 + GPU runner, points `MODULAR_DERIVED_PATH`
at that cache,
pins the same host-CPU target so the compile key matches (builder and runner can
differ within the x86 family), and checks the cache is populated, the warm stamp
is adoptable here, and CPU + GPU matmul and unary dispatches adopt their per-slot
MEFs via manifest force-load with no cold recompile, a box with fewer GPUs than
were warmed force-loads only slots 0..k-1.
"""

from __future__ import annotations

import json
import os

import torch
from max._interpreter_ops import gc_compile
from max.driver import set_virtual_cpu_target
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.executor import InterpreterExecutor, set_default_executor
from max.experimental.tensor import Tensor, realization_context
from python.runfiles import runfiles


def _derived_dir() -> str:
    # $(rlocationpath :compose) is injected via the test env in
    # BUILD.bazel; resolving it through the runfiles lib is CWD-independent.
    rloc = os.environ["XARCH_DERIVED_RLOCATION"]
    r = runfiles.Create()
    assert r is not None, "runfiles unavailable"
    resolved = r.Rlocation(rloc)
    assert resolved is not None, f"could not resolve runfiles path {rloc!r}"
    return resolved


def _setup() -> None:
    # Point the cache-dir resolver at the warmed artifact, then pin the same
    # host-CPU target the warmer recorded in the manifest. The producer sources
    # that target from a per-platform select(); reading it back here keeps the
    # two ends in lockstep without a shared constant. Real GPUs supply the
    # device arch/count, so no virtual-device knobs.
    os.environ["MODULAR_DERIVED_PATH"] = _derived_dir()
    manifest = gc_compile.read_manifest()
    assert manifest is not None, "warm manifest missing; cache not populated"
    set_virtual_cpu_target(manifest["envelope"]["cpu_target"])


def test_warm_cache_is_populated() -> None:
    _setup()
    cache = gc_compile._cache_dir()
    if cache is None or not cache.is_dir():
        raise RuntimeError(f"warm cache dir not populated (missing): {cache}")
    entries = sorted(p.name for p in cache.iterdir())
    if not entries:
        raise RuntimeError(f"warm cache dir is empty: {cache}")
    if "eager_gc_warm_stamp.json" not in entries:
        raise RuntimeError(f"warm cache dir missing warm stamp: {entries}")
    print("XARCH_WARM_CACHE_POPULATED", entries)


def test_warm_is_adoptable_on_real_gpu() -> None:
    _setup()
    cache = gc_compile._cache_dir()
    assert cache is not None, "MODULAR_DERIVED_PATH unset"
    stamp = json.loads((cache / "eager_gc_warm_stamp.json").read_text())
    # The warm has WARM_DEVICE_COUNT slots and this box has no more than that, so
    # the ceiling check (SKU exact, warmed >= runner) passes with slack.
    runner_ctx = gc_compile._context_signature()
    print(
        f"XARCH_WARM_STAMP warmed={stamp.get('context')!r} runner={runner_ctx!r}"
    )
    assert gc_compile.warm_stamp_matches(), (
        f"virtual warm context {stamp.get('context')!r} not adoptable on this "
        f"runner {runner_ctx!r}, SKU mismatch or runner needs more devices "
        "than were warmed"
    )


def test_manifest_force_load_cpu_and_gpu(capfd) -> None:  # noqa: ANN001
    _setup()
    # The per-slot force-load loads the CPU slot + gpu:0..k-1, so a CPU matmul
    # and a GPU matmul both hit an adopted model with no recompile.
    a_cpu = torch.randn(8, 16, dtype=torch.float32)
    w_cpu = torch.randn(16, 32, dtype=torch.float32)
    a_gpu = torch.randn(8, 16, dtype=torch.float32, device="cuda")
    w_gpu = torch.randn(16, 32, dtype=torch.float32, device="cuda")
    with set_default_executor(InterpreterExecutor()):
        with rc.EagerRealizationContext() as ctx, realization_context(ctx):
            y_cpu = Tensor.from_dlpack(a_cpu) @ Tensor.from_dlpack(w_cpu)
            y_gpu = Tensor.from_dlpack(a_gpu) @ Tensor.from_dlpack(w_gpu)
        out_cpu = torch.from_dlpack(y_cpu)
        out_gpu = torch.from_dlpack(y_gpu)
    torch.testing.assert_close(out_cpu, a_cpu @ w_cpu, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_gpu, a_gpu @ w_gpu, rtol=1e-2, atol=1e-2)

    err = capfd.readouterr().err
    # adopted_from_manifest: persistent proof the force-load ran (the marker
    # only prints on first adoption). compileMojoToBinary guards cold recompile.
    assert gc_compile.adopted_from_manifest("matmul"), (
        "matmul force-load path not taken (adopted_from_manifest is False)"
    )
    assert "compileMojoToBinary" not in err, (
        f"cold Mojo kernel recompile on the consumer:\n{err[-3000:]}"
    )


def test_manifest_force_load_unary_cpu_and_gpu(capfd) -> None:  # noqa: ANN001
    _setup()
    # unary_gc has its own per-slot force-load path; loading the CPU slot +
    # gpu:0..k-1 lets a CPU and a GPU sqrt both hit an adopted model, no recompile.
    a_cpu = torch.randn(64, dtype=torch.float32).abs()
    a_gpu = torch.randn(64, dtype=torch.float32, device="cuda").abs()
    with set_default_executor(InterpreterExecutor()):
        with rc.EagerRealizationContext() as ctx, realization_context(ctx):
            y_cpu = F.sqrt(Tensor.from_dlpack(a_cpu))
            y_gpu = F.sqrt(Tensor.from_dlpack(a_gpu))
        out_cpu = torch.from_dlpack(y_cpu)
        out_gpu = torch.from_dlpack(y_gpu)
    torch.testing.assert_close(out_cpu, a_cpu.sqrt(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_gpu, a_gpu.sqrt(), rtol=1e-2, atol=1e-2)

    err = capfd.readouterr().err
    # Same proof as matmul: adopted_from_manifest + no compileMojoToBinary.
    assert gc_compile.adopted_from_manifest("unary"), (
        "unary force-load path not taken (adopted_from_manifest is False)"
    )
    assert "compileMojoToBinary" not in err, (
        f"cold Mojo kernel recompile on the consumer:\n{err[-3000:]}"
    )
