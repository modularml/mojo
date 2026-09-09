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
"""Implements benchmark progress bar."""

from std.os import getenv

from std.builtin.range import _StridedRange


def _get_terminal_size(fallback: Tuple[Int, Int] = (80, 24)) -> Tuple[Int, Int]:
    """Gets the size of the terminal.

    Args:
      fallback: The size of the terminal if it cannot be queried.

    Returns:
      The width and height of the terminal.
    """
    try:
        var columns = Int(getenv("COLUMNS", String(fallback[0])))
        var rows = Int(getenv("LINES", String(fallback[1])))

        return (columns, rows)
    except:
        return (80, 24)


def _hide_cursor():
    print("\x1b[?25l", end="")


def _show_cursor():
    print("\x1b[?25h", end="")


def _clear_line():
    print("\033[A\33[2K")


def _del():
    print("\r\33[2K", end="")


def _end():
    print("\033[0m\r", end="")


struct Progress(ImplicitlyCopyable):
    """
    Implements a basic progress bar with the following usage.

    ```mojo
    from std.time import sleep
    from std.benchmark._progress import Progress

    def main() raises:
        with Progress(10) as p:
            for i in range(10):
                p.advance()
                sleep(0.1)
    ```
    """

    var _range: _StridedRange[.int]
    var _percentage: Float64
    var _term_dims: Tuple[Int, Int]

    @always_inline("nodebug")
    def __init__(out self, end: Int):
        self = Self(0, end)

    @always_inline("nodebug")
    def __init__(out self, start: Int, end: Int, step: Int = 1):
        self._range = _StridedRange(start, end, step)
        self._percentage = Float64(1) / Float64(len(self._range))
        self._term_dims = _get_terminal_size()
        print("")

    def advance(mut self, steps: Int = 1) raises StopIteration:
        comptime BLOCK = "▇"
        comptime PLACE_HOLDER = " "

        if len(self._range) <= 0 or steps <= 0:
            return

        var i = self._range.start
        for _ in range(steps):
            i = self._range.__next__()

        var width = self._term_dims[0]
        var blocks_to_print = Int(Float64(i * width) * self._percentage) + 1
        var placeholders_to_print = max(width - blocks_to_print, 0)

        _del()
        print(BLOCK * blocks_to_print, end="")
        print(PLACE_HOLDER * placeholders_to_print, end="")
        _end()

    def __enter__(self) -> Self:
        return self

    def __exit__(self):
        print("")
        _hide_cursor()
        _show_cursor()

    def __exit__(self, err: Error) -> Bool:
        self.__exit__()
        return False
