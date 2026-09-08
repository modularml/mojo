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
"""Implements the Int class.

These are Mojo built-ins, so you don't need to import them.
"""

from std.collections.interval import IntervalElement
from std.collections.string.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
)
from std.hashlib.hasher import Hasher
from std.math import Ceilable, CeilDivable, Floorable, Truncable
from std.sys.info import is_32bit
from std.sys.info import bit_width_of

from std.math import Absable, DivModable, Powable
from std.python import (
    ConvertibleFromPython,
    Python,
    PythonObject,
)

from std.utils._select import _select_register_value as select
from std.utils._visualizers import lldb_formatter_wrapping_type
from std.utils.coord import Coord, CoordLike

# ===----------------------------------------------------------------------=== #
#  Indexer
# ===----------------------------------------------------------------------=== #


trait Indexer:
    """
    The `Indexer` trait is used for types that can index into a collection or
    pointer. The type returned is the underlying __mlir_type.index, enabling
    types like `SIMD` to not have to be converted to an `Int` first.
    """

    def __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  index
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
def index[T: Indexer](idx: T, /) -> Int:
    """Returns the value of `__mlir_index__` for the given value.

    Parameters:
        T: A type conforming to the `Indexer` trait.

    Args:
        idx: The value.

    Returns:
        An `__mlir_type` representing the index value.
    """
    return Int(SIMDLength(mlir_value=idx.__mlir_index__()))


# ===----------------------------------------------------------------------=== #
#  Intable
# ===----------------------------------------------------------------------=== #


trait Intable:
    """The `Intable` trait describes a type that can be converted to an Int.

    Any type that conforms to `Intable` or
    [`IntableRaising`](/docs/std/builtin/int/IntableRaising/) can construct an
    `Int`.

    This trait requires the type to implement the `__int__()` method. For
    example:

    ```mojo
    @fieldwise_init
    struct Foo(Intable):
        var i: Int

        def __int__(self) -> Int:
            return self.i
    ```

    Now you can construct an `Int`:

    ```mojo
    from std.testing import assert_equal

    foo = Foo(42)
    assert_equal(Int(foo), 42)
    ```

    **Note:** If the `__int__()` method can raise an error, use the
    [`IntableRaising`](/docs/std/builtin/int/IntableRaising/) trait
    instead.
    """

    def __int__(self) -> Int:
        """Get the integral representation of the value.

        Returns:
            The integral representation of the value.
        """
        ...


trait IntableRaising:
    """
    The `IntableRaising` trait describes a type can be converted to an Int, but
    the conversion might raise an error.

    Any type that conforms to [`Intable`](/docs/std/builtin/int/Intable/)
    or `IntableRaising` can construct an `Int`.

    This trait requires the type to implement the `__int__()` method, which can
    raise an error. For example:

    ```mojo
    @fieldwise_init
    struct Foo(IntableRaising):
        var i: Int

        def __int__(self) raises -> Int:
            return self.i
    ```

    Now you can construct an `Int`:

    ```mojo
    from std.testing import assert_equal

    foo = Foo(42)
    assert_equal(Int(foo), 42)
    ```
    """

    def __int__(self) raises -> Int:
        """Get the integral representation of the value.

        Returns:
            The integral representation of the type.

        Raises:
            If the type does not have an integral representation.
        """
        ...


trait _FromInt:
    def __init__(out self, *, from_int: Int):
        ...
