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

"""Provides utility functions for context variable management and driver tensor type conversion."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Generator, Iterable
from contextvars import ContextVar
from pathlib import Path
from types import TracebackType
from typing import Generic, TypeVar

from max import driver, engine
from max.graph import DeviceRef, TensorType

T = TypeVar("T")

_SESSION_LOCK = threading.Lock()
_SESSION: engine.api.InferenceSession | None = None

# Directory of compiled-graph artifacts for the module-global session to reuse,
# or to record into. A session takes its store at construction, since the store
# tracks its position through the graphs the session compiles, so setting either
# of these discards the cached session rather than reconfiguring it.
# One directory or several: a large set of graphs is often split across
# producers, since a build action has a time limit.
_Dirs = tuple[Path, ...] | None
_PRECOMPILED_MEFS: _Dirs = None
_EXPORT_MEFS: Path | None = None


def _session() -> engine.api.InferenceSession:
    """Returns the module-global inference session, creating it on first call."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            device_specs = driver.scan_available_devices()
            if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
                device_specs.append(cpu)
            devices = driver.load_devices(device_specs)
            _SESSION = engine.api.InferenceSession(
                devices=devices,
                precompiled_mefs=_PRECOMPILED_MEFS,
                export_mefs=_EXPORT_MEFS,
            )
        return _SESSION


def _set_mef_dirs(precompiled: _Dirs, export: Path | None) -> None:
    global _PRECOMPILED_MEFS, _EXPORT_MEFS, _SESSION
    _PRECOMPILED_MEFS, _EXPORT_MEFS = precompiled, export
    with _SESSION_LOCK:
        _SESSION = None


class SetterContext(Generic[T], contextlib.AbstractContextManager[T]):
    """An optional undo handle returned by eager setters.

    The set has already happened by the time this object exists. Use it as
    a context manager for scoped semantics -- the previous value is
    restored on exit -- or discard it to keep the new value:

    .. code-block:: python

        from max.experimental.support import SetterContext

        _thing = "initial"

        def set_thing(value: str) -> SetterContext[str]:
            global _thing
            previous, _thing = _thing, value

            def restore(v: str) -> None:
                global _thing
                _thing = v

            return SetterContext(value, previous, restore)

        set_thing("permanent")          # permanent: return value ignored

        with set_thing("scoped"):       # scoped: previous restored on exit
            inside = _thing
        after = _thing

    .. invisible-code-block: python

        assert inside == "scoped"
        assert after == "permanent"  # previous value restored on block exit

    Restoration is value-based (the previous value is captured when the
    setter runs), so out-of-LIFO-order exits restore stale values; nest
    scopes in stack order.
    """

    def __init__(
        self, value: T, previous: T, restore: Callable[[T], None]
    ) -> None:
        self._value = value
        self._previous = previous
        self._restore = restore

    def __enter__(self) -> T:
        return self._value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._restore(self._previous)


def set_precompiled_mefs(
    directory: str | Path | Iterable[str | Path] | None,
) -> SetterContext[tuple[_Dirs, Path | None]]:
    """Initializes graphs from artifacts in ``directory`` instead of compiling.

    Graph compilation does not need the accelerator it targets, only that
    target's arch, so a caller that compiles where it executes spends
    accelerator time on work a CPU could have done. Point an earlier run's
    :func:`set_export_mefs` directory at this to split the two, and the
    accelerator is only held for execution. See
    ``docs/internal/CompileOnCpuRunOnGpu.md``.

    Every graph the module-global session compiles after this -- whether through
    :meth:`~max.experimental.nn.Module.compile` or eager execution -- is matched
    to an artifact by its name and signature, so a divergence raises rather than
    quietly recompiling. The consuming run has to build the same graphs, but not
    in the same order, and may build only some of them.

    Discards the cached session, so call this before compiling anything whose
    artifact should come from ``directory``. Passing :obj:`None` restores
    compiling in process.

    Args:
        directory: A directory written by :func:`set_export_mefs`, or
            :obj:`None` to stop reusing artifacts.

    Returns:
        An undo handle restoring the previous directories.
    """
    previous = (_PRECOMPILED_MEFS, _EXPORT_MEFS)
    if directory is None:
        resolved: _Dirs = None
    elif isinstance(directory, (str, Path)):
        resolved = (Path(directory),)
    else:
        resolved = tuple(Path(one) for one in directory)
    _set_mef_dirs(resolved, None)

    def restore(dirs: tuple[_Dirs, Path | None]) -> None:
        _set_mef_dirs(*dirs)

    return SetterContext((resolved, None), previous, restore)


def set_export_mefs(
    directory: str | Path | None,
) -> SetterContext[tuple[_Dirs, Path | None]]:
    """Records every graph the module-global session compiles into ``directory``.

    The producing half of the split :func:`set_precompiled_mefs` describes: each
    compiled graph lands there as a MEF alongside a manifest naming it, for a
    later run to initialize.

    Discards the cached session, as :func:`set_precompiled_mefs` does. Passing
    :obj:`None` stops recording.

    Args:
        directory: Where to write the artifacts and their manifest, created if
            it does not exist, or :obj:`None` to stop recording.

    Returns:
        An undo handle restoring the previous directories.
    """
    previous = (_PRECOMPILED_MEFS, _EXPORT_MEFS)
    resolved = Path(directory) if directory is not None else None
    _set_mef_dirs(None, resolved)

    def restore(dirs: tuple[_Dirs, Path | None]) -> None:
        _set_mef_dirs(*dirs)

    return SetterContext((None, resolved), previous, restore)


@contextlib.contextmanager
def contextvar_context(var: ContextVar[T], value: T) -> Generator[T]:
    """Context manager that temporarily sets a context variable's value.

    Sets the context variable to the specified value for the duration of the
    context, then resets it to the previous value when the context exits.
    This is useful for scoped configuration changes.

    Args:
        var: The context variable to temporarily modify.
        value: The value to set for the duration of the context.

    Yields:
        The value that was set.

    Example::

        _MY_VAR: ContextVar[int] = ContextVar("_MY_VAR")

        with contextvar_context(_MY_VAR, 42):
            assert _MY_VAR.get() == 42
        # _MY_VAR is now reset to its previous value
    """
    token = var.set(value)
    try:
        yield value
    finally:
        var.reset(token)


def driver_tensor_type(t: driver.Buffer) -> TensorType:
    """Converts a driver tensor to a :obj:TensorType.

    Creates a TensorType instance from a driver-level tensor by extracting
    its dtype, shape, and device information.

    Args:
        t: The driver tensor to convert.

    Returns:
        TensorType: A tensor type representing the driver tensor's properties.
    """
    return TensorType(t.dtype, t.shape, DeviceRef.from_device(t.device))


def driver_tensor_of_type(t: TensorType) -> driver.Buffer:
    """Creates a driver buffer matching the given tensor type."""
    return driver.Buffer(
        t.dtype, [int(d) for d in t.shape], t.device.to_device()
    )
