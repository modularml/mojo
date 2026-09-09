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

"""Shared compile-mode state for the eager interpreter's GC model caches.

The per-op-type GC caches (``matmul_gc``, ``unary_elementwise_gc``, and any
future one) share process-wide singletons defined here (or re-exported from
``_eager_policy``):

- :func:`should_precompile` — whether to compile the full GC matrix at import
  or compile each target lazily on first dispatch (the default).
- :data:`COMPILE_LOCK` — serializes the lazy check-compile-cache so concurrent
  first-dispatches don't race.
- :func:`session_for` — a per-device :class:`~max.engine.InferenceSession`
  cache, shared across op types so lazy compiles reuse one session per device.
- :func:`write_warm_stamp` / :func:`warm_stamp_matches` — marker letting a
  separate lazy process adopt a batched warm instead of compiling target-by-target.

This module must not import from ``handlers.py``.
"""

import json
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Protocol, TypeVar

from max import _core, _eager_policy, engine
from max._core.engine import get_config_value
from max._mlir_context import in_default_mlir_context
from max.driver import (
    Device,
    DeviceSpec,
    accelerator_architecture_name,
    accelerator_count,
    load_devices,
)
from max.dtype import DType
from max.graph import Module

logger = logging.getLogger(__name__)


def canonical_op_name(
    op_type: type[_core.Operation], known_names: Container[str]
) -> str:
    """Canonical ``mo`` op-name for an mo- or rmo-dialect *op_type*.

    The ops library emits ``rmo`` ops; most share the ``mo`` name (``AddOp``)
    but some prefix ``Mo`` (``rmo.MoExpOp`` vs ``mo.ExpOp``). Strip a leading
    ``Mo`` when that yields a *known_names* entry, reusing the ``mo`` sweep's
    cache entry; unknown names pass through.
    """
    name = op_type.__name__
    if name in known_names:
        return name
    if name.startswith("Mo") and name[2:] in known_names:
        return name[2:]
    return name


# Stateless building blocks the eager GC op-family caches share (binary, unary,
# reduce-axis, shape-rearrange); hoisted here so a new family imports them.
CPU_FLOAT_DTYPES = [DType.float32, DType.float64]
GPU_FLOAT_DTYPES = [DType.float16, DType.float32, DType.bfloat16]
SIGNED_INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
UNSIGNED_INT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]

_UINT_FOR_SIZE = {
    1: DType.uint8,
    2: DType.uint16,
    4: DType.uint32,
    8: DType.uint64,
}

WIDTH_DTYPES = list(_UINT_FOR_SIZE.values())
"""The four uint widths a pure-copy family sweeps instead of real dtypes."""

MAX_RANK = 5
"""Historical interpreter rank cap, shared by every rank-capped family."""


def uint_view_dtype(dtype: DType) -> DType:
    """Returns the same-bit-width unsigned int a dtype is bit-cast to for copying.

    Pure-copy ops (shape rearrange, gather, band-part masking) never interpret
    an element's value, so one graph per width serves every dtype of that
    width -- including float16, which has no typed CPU kernel, and bool on GPU.

    Args:
        dtype: The dtype to reinterpret.

    Returns:
        The unsigned integer dtype of the same bit width.

    Raises:
        NotImplementedError: For sub-byte dtypes (e.g. ``float4_e2m1fn``), which
            pack multiple elements per byte and so cannot be reinterpreted
            element-for-element as a whole-byte unsigned int.
    """
    bits = dtype.size_in_bits
    if bits % 8 != 0 or bits // 8 not in _UINT_FOR_SIZE:
        raise NotImplementedError(
            f"eager GC copy path does not support sub-byte dtype {dtype}"
        )
    return _UINT_FOR_SIZE[bits // 8]


_SpecT = TypeVar("_SpecT")


def float_dtypes(device: Device) -> list[DType]:
    """Returns the float dtypes swept on *device*.

    CPU is f32/f64 (16-bit float kernels don't compile on CPU); GPU is
    f16/f32/bf16 (no f64 — NVIDIA rejects it for some ops, Metal lacks it;
    MSTDL-2711).
    """
    return CPU_FLOAT_DTYPES if device.label == "cpu" else GPU_FLOAT_DTYPES


def canonical_shape_rank1(shape: Sequence[int]) -> tuple[int]:
    """Flattens to rank 1; bare ``prod`` keeps scalars at 1 and empty at 0."""
    return (prod(shape),)


def canonical_rank2(shape: Sequence[int]) -> tuple[int, int]:
    """Collapses *shape* to rank 2 ``[rows, features]``.

    The last dim is ``features``; every leading dim is flattened into
    ``rows``. ``prod(())`` is 1, so a rank-1 input (no leading dims) yields
    ``rows == 1``, keeping the rank-1 case branchless.
    """
    return (prod(shape[:-1]), shape[-1])


def canonical_rank3(shape: Sequence[int], axis: int) -> tuple[int, int, int]:
    """Collapses *shape* to rank 3 ``[outer, axis, inner]``.

    *axis* is normalized for negatives; ``prod(())`` is 1, so a leading or
    trailing axis yields an outer/inner dim of 1.
    """
    ndim = len(shape)
    if axis < 0:
        axis += ndim
    return (prod(shape[:axis]), shape[axis], prod(shape[axis + 1 :]))


def flat_index_strides(indexed_shape: Sequence[int]) -> list[int]:
    """Row-major strides of *indexed_shape* (product of trailing sizes).

    Used by gather_nd/scatter_nd to flatten a multi-dimensional index vector
    into a single scalar: ``flat_idx = sum(idx[j] * stride[j])``. E.g.
    ``(4, 3, 2) -> [6, 2, 1]``. Plain Python arithmetic on a list of at most
    a handful of small ints (bounded by the interpreter's rank cap) --
    never touches a Buffer.
    """
    n = len(indexed_shape)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * indexed_shape[i + 1]
    return strides


def discover_devices() -> list[Device]:
    """Returns CPU + every accelerator slot in sweep/warm key order."""
    return load_devices([DeviceSpec.cpu()]) + load_devices(
        [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
    )


# Discovered once at import (not per family) so a missing driver fails here,
# not at first dispatch; every family sweeps this same device set.
DISCOVERED_DEVICES = discover_devices()


def device_class_of(device: Device) -> str:
    """Manifest slot key for *device*: ``"cpu"`` or ``"gpu:{id}"``.

    One source for the device->slot naming, shared by the warm producer
    (writes slots) and adoption (reads them)."""
    return "cpu" if device.label == "cpu" else f"gpu:{device.id}"


def device_spec_of(device: Device) -> DeviceSpec:
    """The :class:`~max.driver.DeviceSpec` that reloads *device* elsewhere.

    Lets an out-of-process warm worker rebuild a session over the same
    devices this process discovered (specs pickle; live devices don't).
    """
    return (
        DeviceSpec.cpu()
        if device.label == "cpu"
        else DeviceSpec.accelerator(device.id)
    )


def spec_for(
    op_type: type[_core.Operation], ops_by_name: Mapping[str, _SpecT]
) -> _SpecT | None:
    """Looks up *op_type*'s spec by canonical (mo) name in *ops_by_name*."""
    name = canonical_op_name(op_type, ops_by_name)
    return ops_by_name.get(name)


# Lazy-per-dispatch by default; ``=1`` precompiles the whole matrix at import
# (MXF-508).
# Re-exported so the CLI can consult it before importing this module.
EAGER_OP_PRECOMPILE_ENV_VAR = _eager_policy.OP_PRECOMPILE_ENV_VAR
should_precompile = _eager_policy.should_precompile

# Stored in the MEF cache dir (see _cache_dir) so it can't outlive the artifacts
# it vouches for.
_WARM_STAMP_NAME = "eager_gc_warm_stamp.json"


def _derived_root() -> Path | None:
    """The ``MODULAR_DERIVED_PATH`` root, or None if unset or empty."""
    try:
        derived = get_config_value("derived_path")
    except KeyError:
        return None
    return Path(derived) if derived else None


def _cache_dir() -> Path | None:
    """MEF cache dir the warm stamp lives in, or None if unresolvable.

    Prefers ``MODULAR_DERIVED_PATH``, else asks the engine, which may create
    the directory as a side effect.
    """
    root = _derived_root()
    if root is not None:
        return root / "cache" / ".max_cache"
    return _core.engine.max_cache_dir()


def _context_signature() -> str:
    """Signature a warm must match before a lazy process adopts it.

    Pins accelerator count + host arch + SKU + MAX version — versioned
    because the engine prunes sibling version directories beneath the stamp.
    """
    n = accelerator_count()
    accel = accelerator_architecture_name() if n else ""
    return (
        f"accelerators={n};cpu={platform.machine()};accel={accel}"
        f";version={_core.__version__}"
    )


def _split_device_count(signature: str) -> tuple[int, str]:
    """Split a context signature into ``(device count, everything else)``.

    Everything after the leading ``accelerators=N`` field must match exactly
    between warm and consumer; only the count is compared as a ceiling.

    Example:
        ``"accelerators=8;cpu=x86_64;accel=sm_100a;version=26.5.0.dev0"`` ->
        ``(8, "cpu=x86_64;accel=sm_100a;version=26.5.0.dev0")``.
    """
    head, _, sku = signature.partition(";")
    return int(head.removeprefix("accelerators=")), sku


def write_warm_stamp() -> bool:
    """Records a batched warm for this context. Returns False if the cache dir
    can't be located (the warm is then unadoptable; caller should surface it).

    Written via a temp file and rename, because concurrent
    ``MAX_EAGER_OP_PRECOMPILE=1`` processes race here.
    """
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"context": _context_signature()})
    tmp = cache_dir / f"{_WARM_STAMP_NAME}.{os.getpid()}.tmp"
    tmp.write_text(payload)
    tmp.replace(cache_dir / _WARM_STAMP_NAME)
    return True


def warm_stamp_matches() -> bool:
    """Returns whether a warm stamp this process can adopt is present.

    Device count is a ceiling; every other field must match exactly.
    """
    cache_dir = _cache_dir()
    if cache_dir is None:
        return False
    try:
        stamp = json.loads((cache_dir / _WARM_STAMP_NAME).read_text())
        warmed_count, warmed_sku = _split_device_count(stamp["context"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    current_count, current_sku = _split_device_count(_context_signature())
    return current_sku == warmed_sku and current_count <= warmed_count


_MANIFEST_NAME = "manifest.json"
_ADOPT_ASSERTED_ENV = "MODULAR_EAGER_WARM_ADOPT_ASSERTED"


def _manifest_path() -> Path | None:
    """Path to the force-load manifest in MODULAR_DERIVED_PATH, or None."""
    root = _derived_root()
    return root / _MANIFEST_NAME if root else None


def write_manifest(
    envelope: dict[str, object], entries: list[dict[str, str]]
) -> bool:
    """Write the force-load manifest. Returns False if MODULAR_DERIVED_PATH is
    unset (nowhere to write it)."""
    path = _manifest_path()
    if path is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"envelope": envelope, "entries": entries}, indent=2)
    )
    return True


def read_manifest() -> dict[str, Any] | None:
    """Read the force-load manifest, or None if absent/unreadable."""
    path = _manifest_path()
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def manifest_adoptable(manifest: dict[str, Any]) -> bool:
    """Whether this process may force-load from the manifest.

    Two manifest kinds, told apart by the envelope's ``gpu`` key:

    - **GPU manifest** (``gpu`` present): adoptable on a box whose accelerator
      arch matches ``gpu.arch`` and whose device count is *at most* the warmed
      count. The manifest indexes one MEF per device slot (a CPU MEF, a gpu:0
      MEF, a gpu:1 MEF; not one combined MEF), so a consumer needing ``k``
      accelerators force-loads slots ``0..k-1``: a warm made for ``N >= k``
      slots adopts (extras unused), but one for fewer than ``k`` cannot (those
      slots were never compiled), so device count is a ceiling, not equality.
    - **CPU-only manifest** (``gpu`` absent/None): adoptable only on a box with
      no accelerator (``accelerator_count() == 0``). There is no GPU slot to
      match, and a box that *does* have an accelerator would need GPU MEFs this
      warm never built.

    Both kinds also require the asserted-toolchain opt-in (closed-loop CI only),
    an asserted-mode envelope, and an *exact* host-arch match: a force-loaded
    MEF's embedded host-ELF kernels are ABI-specific to the host arch (the
    compiled CPU *target* is a floor, so a coarse host-arch match suffices).
    """
    if os.environ.get(_ADOPT_ASSERTED_ENV) != "1":
        return False
    envelope = manifest.get("envelope", {})
    if envelope.get("toolchain", {}).get("mode") != "asserted":
        return False
    # Coarse: platform.machine() misses cpu_target's microarch level; a sub-v3
    # x86 host could adopt v3 kernels and SIGILL. A self-enforcing vN check is
    # deferred, level detection is per-arch (x86 flags; aarch64 core map).
    if envelope.get("host_arch") != platform.machine():
        return False
    gpu = envelope.get("gpu")
    n = accelerator_count()
    if gpu is None:
        # CPU-only manifest: no GPU slot to match, so adoptable only where there
        # is no accelerator that would need GPU MEFs this warm never built.
        return n == 0
    # Device-count ceiling for the manifest path (the stamp path enforces the
    # same via warm_stamp_matches). Default 0 rejects a manifest missing
    # device_count (0 < n) instead of a None comparison.
    if n == 0 or envelope.get("device_count", 0) < n:
        return False
    return gpu.get("arch") == accelerator_architecture_name()


def manifest_entry_path(
    manifest: dict[str, Any], family: str, device_class: str
) -> Path | None:
    """Absolute path to the MEF for (family, device_class), or None if the
    manifest has no such entry."""
    root = _derived_root()
    if root is None:
        return None
    for entry in manifest.get("entries", []):
        if (
            entry.get("family") == family
            and entry.get("device_class") == device_class
        ):
            mef = entry.get("mef")
            return root / mef if mef else None
    return None


def provisioned() -> bool:
    """Returns whether this process can get its eager op models without
    compiling.

    True when a warm manifest can be force-loaded, or a compatible warm stamp
    is present — the single definition of "warmed", shared by the
    import-time gate and ``max warm-interpreter-cache --check``.
    """
    manifest = read_manifest()
    if manifest is not None and manifest_adoptable(manifest):
        return True
    return warm_stamp_matches()


def adopted_from_manifest(family: str) -> bool:
    """Whether ``family`` sourced its models from the warm manifest this process,
    rather than a cold or lazy compile."""
    cache = _REGISTRY.get(family)
    return cache.manifest_adopted if cache is not None else False


# Serializes lazy first-dispatches so concurrent threads don't race on cache
# mutation or a shared session's ``load_all``.
COMPILE_LOCK = threading.Lock()

# Per-device InferenceSession cache. Keyed by (label, id) since a CPU and an
# accelerator can share id 0.
_SESSION_CACHE: dict[tuple[str, int], engine.InferenceSession] = {}


def session_for(device: Device) -> engine.InferenceSession:
    """Returns a cached single-device :class:`~max.engine.InferenceSession`.

    Caching keeps lazy single-target compiles from recreating a session on
    every cache miss; the session is reused for the process lifetime.
    """
    cache_key = (device.label, device.id)
    session = _SESSION_CACHE.get(cache_key)
    if session is None:
        session = engine.InferenceSession(devices=[device])
        _SESSION_CACHE[cache_key] = session
    return session


class GCFamilySpec(Protocol):
    """Immutable per-family strategy: how one ``*_gc.py`` builds/sweeps its
    models.

    Each ``*_gc.py`` provides a concrete class implementing this Protocol
    (e.g. ``matmul_gc._MatmulFamily``); registered once via
    :func:`register_family` by wrapping it in a :class:`GCOpFamily`.
    Implementations are typically stateless, with methods closing over
    their module's own globals rather than instance state.
    """

    name: str
    """Family id; namespaces cache keys and per-slot MEF names."""

    def build_module(self) -> Module:
        """Builds the full sweep matrix (every device it sweeps)."""
        ...

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Builds one device's slot module.

        Args:
            device: The device to build for.
            module: When given, appends this device's graphs into it instead
                of a fresh :class:`Module`, so :meth:`build_module` can share
                one module across every device in the sweep.
        """
        ...

    def sweep_devices(self) -> list[Device]:
        """Returns the devices this family sweeps.

        Every family sweeps the same discovered set by default, so this is a
        concrete, inherited implementation rather than a per-family override
        point -- except for a family whose kernel is restricted to a device
        subset, which must override this to exclude the devices it never
        builds for: the ``MO_HostOnly`` families (``resize_gc``,
        ``nonzero_gc``, ``nms_gc``, ``roi_align_gc``) which are CPU-only,
        since sweeping an accelerator for a host-only op would only produce
        an empty per-slot module. ``_adopt_from_manifest`` requires a
        manifest entry for every device this returns, so a family that lists
        a device it never populates fails adoption entirely. An empty list
        (none of the family's devices are present) warms as a no-op.
        """
        return list(DISCOVERED_DEVICES)


@dataclass
class GCOpFamily:
    """One eager GC op family's runtime state plus warm/adopt/dispatch logic.

    Wraps a :class:`GCFamilySpec` (registered once per family via
    :func:`register_family`); replaces the per-module ``_swept`` /
    ``_X_MODEL_CACHE`` globals each ``*_gc.py`` copied.
    """

    spec: GCFamilySpec
    """This family's build/sweep strategy."""

    cache: dict[str, engine.Model] = field(default_factory=dict)
    """Compiled models, keyed by graph_name."""
    swept: bool = False
    """Whether a whole-matrix sweep or adoption has been attempted."""
    manifest_adopted: bool = False
    """Whether the cache was force-loaded from a manifest."""

    @property
    def name(self) -> str:
        """Family id; namespaces cache keys and per-slot MEF names."""
        return self.spec.name

    def sweep_devices(self) -> list[Device]:
        return self.spec.sweep_devices()

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        return self.spec.build_module_for_device(device, module)

    @in_default_mlir_context
    def build_sweep_module(self) -> Module:
        """Builds the full sweep matrix module without compiling it.

        Exists so the out-of-process warm can compile the same module
        :meth:`compile_sweep` builds — its MEF cache key must match a
        later in-process sweep's.
        """
        return self.spec.build_module()

    @in_default_mlir_context
    def compile_sweep(self) -> bool:
        """Force-load an adoptable manifest, else batch-compile every target.

        Returns:
            True if it batch-compiled; False if it force-loaded the
            manifest instead, in which case the caller must not stamp.
        """
        if self._adopt_manifest_if_adoptable():
            return False
        devices = self.sweep_devices()
        if not devices:
            # InferenceSession(devices=[]) silently defaults to a CPU session.
            self.swept = True
            return True
        session = engine.InferenceSession(devices=devices)
        self.cache.update(
            session.load_all(self.spec.build_module(), weights_registry={})
        )
        self.swept = True
        return True

    def ensure_swept(self) -> None:
        """Attempt whole-cache adoption once before per-target compilation.

        Force-loads an adoptable manifest, else batch-sweeps a matching warm
        stamp — the one path that may batch-compile under
        ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
        """
        if self.swept:
            return
        if self._adopt_manifest_if_adoptable():
            return
        if warm_stamp_matches():
            self.swept = True
            start = time.perf_counter()
            try:
                self.compile_sweep()
            except RuntimeError:
                # If the sweep fails to compile, fall back to lazy per-target
                # compilation rather than breaking import.
                logger.warning(
                    "Eager interpreter warm-cache adoption failed;"
                    " compiling %s targets on demand.",
                    self.name,
                    exc_info=True,
                )
            else:
                _eager_policy.note_batched_adopt_end(
                    self.name, time.perf_counter() - start
                )

    def model_for(
        self,
        key: str,
        device: Device,
        build: Callable[[Module], None],
        *,
        unsupported_reason: Callable[[], str | None] | None = None,
    ) -> engine.Model:
        """Shared lazy-dispatch skeleton every family's ``*_model`` function
        wraps: cache hit, optional support-guard, then a locked adopt, the
        cold-start policy check, and a per-target compile.

        The policy check runs after adoption, not before: refusing ahead of
        adoption would leave a warmed machine unable to serve any eager op.

        Args:
            key: The cache key for this target (family-specific format).
            device: The target device, passed through to *build* on a lazy
                miss.
            build: Builds the single-target graph into a fresh ``Module``.
            unsupported_reason: When given, called only on a cache miss (a
                cache hit never pays for the support check) to test whether
                the target is outside the family's supported set; a
                non-``None`` return is raised as ``KeyError`` verbatim before
                anything is adopted or compiled (each family formats its own
                wording/details).

        Returns:
            The compiled :class:`~max.engine.Model`.

        Raises:
            KeyError: *unsupported_reason*'s message if it returns one.
            EagerLazyCompileDisallowed: if the target is still missing after
                adoption and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
        """
        model = self.cache.get(key)
        if model is not None:
            return model
        if unsupported_reason is not None:
            reason = unsupported_reason()
            if reason is not None:
                raise KeyError(reason)
        with COMPILE_LOCK:
            model = self.cache.get(key)
            if model is not None:
                return model
            self.ensure_swept()
            model = self.cache.get(key)
            if model is not None:
                return model
            if not _eager_policy.allow_lazy_compile():
                raise _eager_policy.EagerLazyCompileDisallowed(
                    _eager_policy.refusal_message(
                        key, provisioned=provisioned()
                    )
                )
            start = time.perf_counter()
            model = compile_single_target(self, key, device, build)
            _eager_policy.record_cold_compile(
                key, time.perf_counter() - start, provisioned=provisioned()
            )
            return model

    def _adopt_manifest_if_adoptable(self) -> bool:
        manifest = read_manifest()
        if manifest is None or not manifest_adoptable(manifest):
            return False
        self.swept = True
        return self._adopt_from_manifest(manifest)

    def _adopt_from_manifest(self, manifest: dict[str, Any]) -> bool:
        """Force-load this family's per-slot MEFs named in *manifest*."""
        # Load the slots for exactly the devices the session spans: both come
        # from the one sweep_devices() call, so they can't diverge.
        devices = self.sweep_devices()
        if not devices:
            # Nothing to adopt; avoid the silent default-CPU session.
            return True
        try:
            session = engine.InferenceSession(devices=devices)
            loaded: dict[str, engine.Model] = {}
            names: list[str] = []
            for device in devices:
                device_class = device_class_of(device)
                mef = manifest_entry_path(manifest, self.name, device_class)
                if mef is None or not mef.exists():
                    return False
                loaded.update(session.load_all(str(mef), weights_registry={}))
                names.append(mef.name)
        except Exception:
            # Loading a .mef re-raises whatever the backend threw, so catch
            # broadly and fall back to compiling on demand.
            logger.warning(
                "Eager warm manifest force-load failed; compiling on demand.",
                exc_info=True,
            )
            return False
        self.cache.update(loaded)
        self.manifest_adopted = True
        # Force-load bypasses the keyed cache, so [modular-cache] logging is
        # silent; this stderr marker signals it ran.
        print(
            f"[eager-warm] manifest force-load: {self.name} {names}"
            f" ({len(loaded)} models)",
            file=sys.stderr,
            flush=True,
        )
        return True


_REGISTRY: dict[str, GCOpFamily] = {}


def register_family(family: GCOpFamily) -> None:
    """Register *family*; each ``*_gc.py`` calls this once at import."""
    _REGISTRY[family.name] = family


def registered_families() -> tuple[GCOpFamily, ...]:
    """Every registered family, in registration (import) order."""
    return tuple(_REGISTRY.values())


@in_default_mlir_context
def compile_single_target(
    family: GCOpFamily,
    key: str,
    device: Device,
    build_graph: Callable[[Module], None],
) -> engine.Model:
    """Compile one graph into *family*'s cache and return it (lazy path)."""
    module = Module()
    build_graph(module)
    session = session_for(device)
    family.cache.update(session.load_all(module, weights_registry={}))
    return family.cache[key]
