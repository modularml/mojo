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

"""Cold-start policy for the eager interpreter's graph-compiler models.

Standard-library only, so :mod:`max.experimental.executor` can read the policy
without importing the Mojo-backed :mod:`max._interpreter_ops`.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

ALLOW_LAZY_COMPILE_ENV_VAR = "MAX_EAGER_ALLOW_LAZY_COMPILE"

# Named here so the CLI can read it before importing the Mojo-backed package.
OP_PRECOMPILE_ENV_VAR = "MAX_EAGER_OP_PRECOMPILE"

_WARN_LOCK = threading.Lock()
_warned_cold_compile = False


class EagerLazyCompileDisallowed(RuntimeError):
    """Raised when a process forbidden from compiling needs an eager op model."""


def allow_lazy_compile() -> bool:
    """Returns whether this process may compile one eager op model on demand.

    Forbids the per-target compile at dispatch, not a batched adoption or
    sweep. Read at call time, since ``max serve`` sets the variable after this
    module is imported; only the literal ``"0"`` disables.
    """
    return os.environ.get(ALLOW_LAZY_COMPILE_ENV_VAR, "1") != "0"


def should_precompile() -> bool:
    """Returns whether to precompile the full GC matrix at import.

    Read at call time, since the sweep runs from ``__init__``, which may be
    imported before a launcher sets the variable; only ``"1"`` enables.
    """
    return os.environ.get(OP_PRECOMPILE_ENV_VAR, "0") == "1"


def refusal_message(key: str, *, provisioned: bool) -> str:
    """Returns the error text for a refused compile of *key*.

    Args:
        key: The family cache key that missed.
        provisioned: Whether this machine has an adoptable warm. Selects the
            remedy; passed in so this module stays standard-library-only.

    Returns:
        A message naming the target and the remedies that apply to it.
    """
    if provisioned:
        return (
            f"{key} has not been compiled, and on-the-fly compilation of eager"
            " ops has been disabled. This machine has a precompiled eager"
            " cache, but it does not include this operation.\n\n"
            "Please use `Module` and `Module.compile`, or allow on-the-fly"
            " compilation with `max serve --allow-cold-interpreter-cache` (or"
            f" {ALLOW_LAZY_COMPILE_ENV_VAR}=1)."
        )
    return (
        f"{key} has not been compiled, and on-the-fly compilation of eager ops"
        " has been disabled.\n\n"
        "Run `max warm-interpreter-cache` on this machine, or allow on-the-fly"
        " compilation with `max serve --allow-cold-interpreter-cache` (or"
        f" {ALLOW_LAZY_COMPILE_ENV_VAR}=1)."
    )


def record_cold_compile(key: str, seconds: float, *, provisioned: bool) -> None:
    """Warns about the first on-demand compile of this process.

    Only the first is reported, so a run compiling several targets understates
    its cold-start cost.

    Args:
        key: The family cache key that was compiled.
        seconds: Wall-clock time the compile took.
        provisioned: Whether this machine has an adoptable warm; on a warmed
            machine the advice to warm is wrong.
    """
    global _warned_cold_compile
    with _WARN_LOCK:
        if _warned_cold_compile:
            return
        _warned_cold_compile = True
    if provisioned:
        logger.warning(
            "compiled %s on the fly in %.1fs. This machine has a precompiled"
            " eager cache, but it does not include this operation, so warming"
            " again will not skip it.",
            key,
            seconds,
        )
        return
    logger.warning(
        "compiled %s on the fly in %.1fs. Run `max warm-interpreter-cache`"
        " once to skip this next time.",
        key,
        seconds,
    )


def note_sweep_start() -> None:
    """Warns that the full eager op matrix is about to compile at import."""
    logger.warning(
        "compiling the full eager op matrix at import; this may take a while."
        " `max warm-interpreter-cache` moves this one-time cost out of startup."
    )


def note_sweep_end(seconds: float) -> None:
    """Reports how long the import-time sweep took.

    Args:
        seconds: Wall-clock time the sweep took.
    """
    logger.info("eager op matrix ready (%.1fs).", seconds)


def note_batched_adopt_end(family: str, seconds: float) -> None:
    """Reports how long one family's batched adoption took.

    Args:
        family: The family id that was adopted.
        seconds: Wall-clock time the batched load took.
    """
    logger.info("eager op family %s ready (%.1fs).", family, seconds)


def _reset_for_test() -> None:
    """Clears the warned-once flag. For tests only."""
    global _warned_cold_compile
    with _WARN_LOCK:
        _warned_cold_compile = False
