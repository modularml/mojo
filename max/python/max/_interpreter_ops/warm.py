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

"""Batch-compiles every GC op family for ``max warm-interpreter-cache``.

Lives in this package rather than the CLI so the implementation can
statically import :class:`~max._interpreter_ops.gc_compile.GCOpFamily` and
friends under mypy; the CLI reaches it via its usual dynamic import of this
package. Only the warm command imports this module, so its heavier imports
stay off the eager dispatch path.
"""

from collections.abc import Iterator, Sequence
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool

from max import mlir
from max._core import Operation
from max.driver import DeviceSpec
from max.experimental.compile_pool import ProcessCompilePool
from max.graph import Module

from . import gc_compile
from .gc_compile import GCOpFamily


def _roundtrips_stably(module_bytecode: bytes) -> bool:
    """Whether *module_bytecode* survives a parse/re-serialize unchanged.

    A pool worker rebuilds the module from bytecode before compiling, and the
    MEF cache key hashes the rebuilt IR. A module that parses back different
    (the ``mo.graph`` ``counter`` property is recomputed on parse) would be
    cached under a key no in-process consumer computes, making its warm
    unadoptable — such a family must compile in-process instead.
    """
    roundtripped = Operation.from_bytecode(module_bytecode, mlir.Context())
    return roundtripped.bytecode == module_bytecode


def _compile_in_process(family: GCOpFamily) -> int:
    before = len(family.cache)
    family.compile_sweep()
    return len(family.cache) - before


def _warm_serial(
    families: Sequence[GCOpFamily],
) -> Iterator[tuple[str, int]]:
    for family in families:
        yield family.name, _compile_in_process(family)


def _warm_parallel(
    families: Sequence[GCOpFamily], jobs: int
) -> Iterator[tuple[str, int]]:
    """Compiles the families concurrently in compile-pool worker processes.

    Workers insert into the same on-disk MEF cache a later in-process
    ``compile_sweep`` reads, so each worker's session must span exactly the
    family's sweep devices, in sweep order — the session's default device
    enters the cache key. The compiled artifacts themselves are discarded;
    the cache entries are the product.

    Families whose module doesn't survive the bytecode trip to a worker
    (see :func:`_roundtrips_stably`) compile in this process instead,
    overlapped with the pool batch.
    """
    pool_specs = [
        gc_compile.device_spec_of(d) for d in gc_compile.DISCOVERED_DEVICES
    ]
    pooled: list[tuple[GCOpFamily, Module, list[DeviceSpec]]] = []
    local: list[GCOpFamily] = []
    for family in families:
        devices = family.sweep_devices()
        if not devices:
            # The pool rejects an empty device_specs (MXF-599).
            yield family.name, 0
            continue
        module = family.build_sweep_module()
        if _roundtrips_stably(module.mlir_module.bytecode):
            specs = [gc_compile.device_spec_of(d) for d in devices]
            pooled.append((family, module, specs))
        else:
            local.append(family)

    with ProcessCompilePool(device_specs=pool_specs, max_workers=jobs) as pool:
        futures = {
            pool.compile_module(module, device_specs=specs): (
                family.name,
                len(module.top_level_graph_names()),
            )
            for family, module, specs in pooled
        }
        for family in local:
            yield family.name, _compile_in_process(family)
        for warmed, future in enumerate(as_completed(futures)):
            name, op_count = futures[future]
            try:
                future.result()
            except BrokenProcessPool as e:
                # A dead worker fails every pending future, so this family
                # is merely the first reported casualty, not the culprit.
                pool.close()
                raise RuntimeError(
                    "a compile worker process died with"
                    f" {len(futures) - warmed} op families unfinished; this"
                    " machine is not stamped as warmed. Re-run, or pass"
                    " --jobs 1 to isolate the failure."
                ) from e
            except Exception as e:
                pool.close()
                raise RuntimeError(
                    f"compiling the {name} op family failed: {e}"
                ) from e
            yield name, op_count


def warm_families(jobs: int) -> Iterator[tuple[str, int]]:
    """Compiles every registered GC family's sweep into the on-disk cache.

    Yields ``(family_name, ops_compiled)`` as each family finishes, in
    completion order, so the caller can render progress. Does not write the
    warm stamp — that stays with the caller, which must only stamp after
    every family succeeds.

    Args:
        jobs: Concurrent compile worker processes; ``1`` compiles serially
            in-process.

    Raises:
        RuntimeError: If a family fails to compile or a worker process dies;
            nothing further compiles.
    """
    families = gc_compile.registered_families()
    if jobs == 1:
        yield from _warm_serial(families)
    else:
        yield from _warm_parallel(families, jobs)
