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
"""Contains bounds checks which are on by default for CPU and off by default for GPU.
"""

from . import OptionalReg
from std.builtin.builtin_slice import ContiguousSlice
from std.builtin.debug_assert import _assert_enabled
from std.reflection import SourceLocation
from std.sys.info import is_gpu
from std.reflection import call_location

comptime _AssertMode[
    cpu_default: Bool
] = "safe" if cpu_default and not is_gpu() else "none"


@always_inline
def check_bounds[
    cpu_default: Bool = True,
](idx: Some[Indexer], size: Int, location: Optional[SourceLocation] = None):
    """Bounds check which is on by default for CPU, and off by default for GPU.

    You can turn off CPU bounds checks for a specific collection by setting
    `check_bounds[cpu_default=False](idx, size, loc)`, but turn them on for
    tests with:

    ```bash
    mojo build -D ASSERT=all main.mojo
    ```

    The defaults are optimal for most use cases, where CPU bounds checks are
    cheap and valuable, but GPU bounds checks are too expensive due to branching
    costs. For maximum performance you can turn off all asserts regardless of
    defaults with:

    ```bash
    mojo build -D ASSERT=none main.mojo
    ```

    Parameters:
        cpu_default: If the bounds check is on by default on CPU.

    Args:
        idx: The index for the bounds check.
        size: The size of the container, and first index that would be out of range.
        location: `SourceLocation` shown on assert error. Defaults to showing the callsite
            two levels of function calls above this one. So if `check_bounds` is called
            inside a `__getitem__` method, it will show the source location where
            the incorrect index was provided.
    """
    debug_assert[assert_mode=_AssertMode[cpu_default]](
        UInt(index(idx)) < UInt(size),
        "index ",
        index(idx),
        " is out of bounds, valid range is 0 to ",
        size - 1,
        location=location.or_else(call_location[inline_count=2]()),
    )


@always_inline
def check_slice_bounds[
    cpu_default: Bool = True,
](
    slice: ContiguousSlice,
    len: Int,
    location: Optional[SourceLocation] = None,
) -> Tuple[Int, Int]:
    """Bounds check for slice indexing, which is on by default for CPU, and
    off by default for GPU.

    Resolves `slice`'s optional `start`/`end` against `len` and aborts if the
    resolved start or end index is out of bounds, or if start is greater than
    end. `ContiguousSlice` does not support negative (from-the-end) indices,
    so a negative start or end is always out of bounds.

    Parameters:
        cpu_default: If the bounds check is on by default on CPU.

    Args:
        slice: The slice to resolve and bounds check.
        len: The size of the container.
        location: `SourceLocation` shown on assert error. Defaults to showing
            the callsite two levels of function calls above this one. So if
            `check_slice_bounds` is called inside a `__getitem__` method, it
            will show the source location where the incorrect slice was
            provided.

    Returns:
        The slice's resolved `(start, end)`.
    """
    comptime mode = _AssertMode[cpu_default]
    var start = slice.start.or_else(0)
    var end = slice.end.or_else(len)

    @no_inline
    def do_asserts(location: SourceLocation) {imm}:
        debug_assert[assert_mode=mode](
            UInt(start) <= UInt(len),
            "slice start index ",
            start,
            " is out of bounds, valid range is 0 to ",
            len,
            location=location,
        )
        debug_assert[assert_mode=mode](
            UInt(end) <= UInt(len),
            "slice end index ",
            end,
            " is out of bounds, valid range is 0 to ",
            len,
            location=location,
        )
        debug_assert[assert_mode=mode](
            start <= end,
            "slice start index ",
            start,
            " is greater than slice end index ",
            end,
            location=location,
        )

    # TODO(MSTDL-3004): Figure out if we can enable bounds checking
    # on GPU.
    comptime if _assert_enabled[mode, True]():
        # Combine the failure conditions here so valid slices avoid the
        # no-inline assertion helper, while invalid slices still bounds check.
        if UInt(start) > UInt(len) or UInt(end) > UInt(len) or start > end:
            do_asserts(location.or_else(call_location[inline_count=2]()))

    return start, end
