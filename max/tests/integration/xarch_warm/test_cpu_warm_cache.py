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
"""Consumer: adopt a CPU-only eager-GC-sweep warm cache on a GPU-less box.

The `warm_cpu` build action (see `interp_cache.bzl`) warms the sweep with no
`--target`, so it sets no virtual device and warms just the CPU slot into a
manifest with no `gpu` key. This test runs on a GPU-less box, the m7i/m7g CPU
CI lanes, or any linux box with `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`
cleared so `accelerator_count() == 0`, points `MODULAR_DERIVED_PATH` at that
cache, and dispatches a CPU matmul + unary. It runs under
`MAX_EAGER_OP_PRECOMPILE=1`, so the sweep runs at the first `_interpreter_ops`
import and force-loads the CPU MEFs instead of a cold compile: the receipt
asserts the force-load markers appear and no `compileMojoToBinary` does.

`_interpreter_ops` is deliberately NOT imported at module top, the executor
defers `import max._interpreter` to first dispatch, so the precompile sweep runs
after `_setup()` has pointed `MODULAR_DERIVED_PATH` at the warmed cache. The
manifest is read with `json` (not `gc_compile`) for the same reason.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from max.driver import accelerator_count, set_virtual_cpu_target
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.executor import InterpreterExecutor, set_default_executor
from max.experimental.tensor import Tensor, realization_context
from python.runfiles import runfiles


def _derived_dir() -> str:
    # $(rlocationpath :warm_cpu) is injected via the test env in BUILD.bazel;
    # resolving it through the runfiles lib is CWD-independent.
    rloc = os.environ["XARCH_DERIVED_RLOCATION"]
    r = runfiles.Create()
    assert r is not None, "runfiles unavailable"
    resolved = r.Rlocation(rloc)
    assert resolved is not None, f"could not resolve runfiles path {rloc!r}"
    return resolved


def _setup() -> None:
    # Point the cache-dir resolver at the warmed artifact, then pin the host-CPU
    # target the warmer recorded. Read the manifest with json (not gc_compile) so
    # this does NOT import _interpreter_ops before MODULAR_DERIVED_PATH is set --
    # under MAX_EAGER_OP_PRECOMPILE=1 that import triggers the force-load sweep,
    # which must see the warmed cache dir. json.loads returns Any, so no bare
    # generic annotation is needed to read the loose manifest shape.
    derived = _derived_dir()
    os.environ["MODULAR_DERIVED_PATH"] = derived
    manifest = json.loads((Path(derived) / "manifest.json").read_text())
    set_virtual_cpu_target(manifest["envelope"]["cpu_target"])


def test_cpu_warm_cache_is_populated() -> None:
    _setup()
    derived = Path(os.environ["MODULAR_DERIVED_PATH"])
    manifest = json.loads((derived / "manifest.json").read_text())
    # A CPU-only manifest has no "gpu" key and warms only the CPU slot.
    assert "gpu" not in manifest["envelope"], (
        f"cpu_warm manifest unexpectedly has a gpu key: {manifest['envelope']}"
    )
    for mef in ("matmul_cpu.mef", "unary_cpu.mef"):
        if not (derived / mef).is_file():
            raise RuntimeError(f"CPU warm MEF missing: {derived / mef}")
    stamp = derived / "cache" / ".max_cache" / "eager_gc_warm_stamp.json"
    if not stamp.is_file():
        raise RuntimeError(f"warm stamp missing: {stamp}")
    print("XARCH_CPU_WARM_POPULATED", sorted(p.name for p in derived.iterdir()))


def test_cpu_only_box_has_no_accelerator() -> None:
    # The whole CPU-only path hinges on accelerator_count()==0 (manifest_adoptable
    # admits a CPU-only manifest only there). Verify CUDA/HIP_VISIBLE_DEVICES=""
    # actually zeroed the count in this harness before relying on it below.
    _setup()
    assert accelerator_count() == 0, (
        f"expected a CPU-only box but accelerator_count()={accelerator_count()};"
        " CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES did not hide the GPUs"
    )


def test_manifest_force_load_cpu_precompile(capfd) -> None:  # noqa: ANN001
    _setup()
    # Precondition for the CPU-only force-load path (see the accel test above).
    assert accelerator_count() == 0, "GPUs not hidden; see accel test"

    # The first dispatch triggers the executor's deferred `import max._interpreter`
    # -> _interpreter_ops import -> MAX_EAGER_OP_PRECOMPILE sweep, which now sees
    # MODULAR_DERIVED_PATH + the manifest and force-loads the CPU matmul + unary
    # MEFs (both markers emitted here). The unary dispatch then hits the already
    # force-loaded model.
    a = torch.randn(8, 16, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    s = torch.randn(64, dtype=torch.float32).abs()
    with set_default_executor(InterpreterExecutor()):
        with rc.EagerRealizationContext() as ctx, realization_context(ctx):
            y = Tensor.from_dlpack(a) @ Tensor.from_dlpack(w)
            z = F.sqrt(Tensor.from_dlpack(s))
        out = torch.from_dlpack(y)
        out_z = torch.from_dlpack(z)
    torch.testing.assert_close(out, a @ w, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_z, s.sqrt(), rtol=1e-2, atol=1e-2)

    err = capfd.readouterr().err
    # Deferred import (see docstring): a top-level import would trigger the
    # PRECOMPILE sweep before _setup. adopted_from_manifest is the persistent
    # proof; compileMojoToBinary guards against cold recompile.
    from max._interpreter_ops import adopted_from_manifest

    assert adopted_from_manifest("matmul"), (
        "matmul force-load path not taken (adopted_from_manifest is False)"
    )
    assert adopted_from_manifest("unary"), (
        "unary force-load path not taken (adopted_from_manifest is False)"
    )
    assert "compileMojoToBinary" not in err, (
        f"cold Mojo kernel recompile on the consumer:\n{err[-3000:]}"
    )
