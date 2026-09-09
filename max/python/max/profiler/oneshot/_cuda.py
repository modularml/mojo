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

"""In-process ``cudaProfilerStart``/``Stop`` bracketing via ``ctypes``.

Used to bound the nsys capture window to the timed region only, so warmup
and graph-compile time are excluded from the report.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("max.profiler.oneshot")

_CUDART_LIB_CANDIDATES: tuple[str, ...] = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart.dylib",
)


def _load_cudart() -> ctypes.CDLL | None:
    """Locate and dlopen the CUDA runtime library, or return ``None``."""
    found = ctypes.util.find_library("cudart")
    candidates: tuple[str, ...] = (
        (found,) + _CUDART_LIB_CANDIDATES if found else _CUDART_LIB_CANDIDATES
    )
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def can_load_cudart() -> bool:
    """Return True if libcudart can be dlopened on this host.

    Used by the backend gate to refuse the nsys re-exec when ``cudaProfilerStart``
    cannot be called, since the nsys child is launched with
    ``--capture-range=cudaProfilerApi`` and would otherwise produce an empty
    ``.nsys-rep``.
    """
    return _load_cudart() is not None


@contextmanager
def _cuda_profiler_region() -> Iterator[None]:
    """Bracket a region with ``cudaProfilerStart`` / ``cudaProfilerStop``.

    Assumes libcudart is loadable; the backend gate
    (``ProfileBackend.can_capture_gpu``) already refuses to re-exec under nsys
    when it is not. If we somehow get here without it, raise — the nsys child
    runs with ``--capture-range=cudaProfilerApi`` and would otherwise produce
    an empty ``.nsys-rep``.
    """
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError(
            "libcudart not loadable inside nsys child; cudaProfilerStart "
            "cannot be called and the trace would be empty"
        )
    for fn in (cudart.cudaProfilerStart, cudart.cudaProfilerStop):
        fn.argtypes = []
        fn.restype = ctypes.c_int
    rc = cudart.cudaProfilerStart()
    if rc != 0:
        logger.debug("cudaProfilerStart returned %d", rc)
    try:
        yield
    finally:
        rc = cudart.cudaProfilerStop()
        if rc != 0:
            logger.debug("cudaProfilerStop returned %d", rc)
