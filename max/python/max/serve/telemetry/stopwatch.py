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

import contextlib
import time
from collections.abc import Callable, Generator
from types import TracebackType


class StopWatch:
    """A simple stopwatch that supports the context manager protocol.

    When used as a context manager, ``elapsed`` reports the time since the
    scope was entered and holds the total time after the scope is exited.
    The stopwatch can be re-entered. It can also be used without scopes by
    calling ``reset``. To override the start time externally, use the
    ``time_ns`` class method to ensure the same baseline is used.

    .. code-block:: python

        from max.serve.telemetry.stopwatch import StopWatch

        def do_work():
            return sum(range(1000))

        with StopWatch() as sw:
            do_work()
        print(sw.elapsed_ms)

    .. invisible-code-block: python

        assert isinstance(sw.elapsed_ms, float)
        assert sw.elapsed_ms >= 0.0
    """

    @classmethod
    def start(cls):  # noqa: ANN206
        sw = cls()
        sw.reset()
        return sw

    @staticmethod
    def time_ns() -> int:
        return time.perf_counter_ns()

    def __init__(self, start_ns: int | None = None) -> None:
        self.start_ns: int = (
            start_ns if start_ns is not None else self.time_ns()
        )
        self.exit_ns: int = 0

    def reset(self, start_ns: int | None = None) -> None:
        if start_ns is None:
            start_ns = self.time_ns()
        self.start_ns = start_ns

    def __enter__(self):
        self.reset()
        self.exit_ns = 0
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException],
        _exc_value: BaseException,
        _exc_tb: TracebackType,
    ) -> None:
        self.exit_ns = self.time_ns()

    @property
    def elapsed_ns(self) -> int:
        if not self.start_ns:
            raise RuntimeError("Stopwatch not started")
        end = self.exit_ns or self.time_ns()
        elapsed = end - self.start_ns
        return elapsed

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1e6

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ns / 1e9


@contextlib.contextmanager
def record_ms(
    fn: Callable[[float], None], on_error: bool = False
) -> Generator[StopWatch, None, None]:
    """Starts a :class:`StopWatch` and calls ``fn(elapsed_ms)`` when complete.

    Args:
        fn: Called with the elapsed duration in milliseconds when the block
            completes.
        on_error: If ``True``, records results even when an exception is raised.
            Defaults to ``False``.

    Yields:
        A :class:`StopWatch` instance for intermediate timing measurements.
    """
    sw = StopWatch()
    try:
        yield sw
        fn(sw.elapsed_ms)
    except Exception:
        if on_error:
            fn(sw.elapsed_ms)
        raise
