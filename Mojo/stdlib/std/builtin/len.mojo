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
"""Provides the `len()` function and its associated traits.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------=== #
#  Sized
# ===----------------------------------------------------------------------=== #


trait Sized:
    """The `Sized` trait describes a type that has an integer length (such as a
    string or array).

    Any type that conforms to `Sized` or
    [`SizedRaising`](/docs/std/builtin/len/SizedRaising/) works with the
    built-in [`len()`](/docs/std/builtin/len/len/) function.

    The `Sized` trait requires a type to implement the `__len__()`
    method. For example:

    ```mojo
    @fieldwise_init
    struct Foo(Sized):
        var length: Int

        def __len__(self) -> Int:
            return self.length
    ```

    You can pass an instance of `Foo` to the `len()` function to get its
    length:

    ```mojo
    @fieldwise_init
    struct Foo(Sized):
        var length: Int

        def __len__(self) -> Int:
            return self.length

    var foo = Foo(42)
    print(len(foo) == 42)
    ```

    ```plaintext
    True
    ```

    **Note:** If the `__len__()` method can raise an error, use the
    [`SizedRaising`](/docs/std/builtin/len/SizedRaising/) trait instead.

    """

    def __len__(self) -> Int:
        """Get the length of the type.

        Returns:
            The length of the type.
        """
        ...


trait SizedRaising:
    """The `SizedRaising` trait describes a type that has an integer length,
    which might raise an error if the length can't be determined.

    Any type that conforms to [`Sized`](/docs/std/builtin/len/Sized/) or
    `SizedRaising` works with the built-in
    [`len()`](/docs/std/builtin/len/len/) function.

    The `SizedRaising` trait requires a type to implement the `__len__()`
    method, which can raise an error. For example:

    ```mojo
    @fieldwise_init
    struct Foo(SizedRaising):
        var length: Int

        def __len__(self) raises -> Int:
            if self.length < 0:
                raise Error("Length is negative")
            return self.length
    ```

    You can pass an instance of `Foo` to the `len()` function to get its
    length:

    ```mojo
    @fieldwise_init
    struct Foo(SizedRaising):
        var length: Int

        def __len__(self) raises -> Int:
            if self.length < 0:
                raise Error("Length is negative")
            return self.length

    def main() raises:
        var foo = Foo(42)
        print(len(foo) == 42)
    ```

    ```plaintext
    True
    ```
    """

    def __len__(self) raises -> Int:
        """Get the length of the type.

        Returns:
            The length of the type.

        Raises:
            If the length cannot be computed.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  len
# ===----------------------------------------------------------------------=== #


@doc_hidden
@unavailable(
    "`len(String/StringSlice)` is not supported because Mojo strings are UTF-8"
    " encoded, so a single length is ambiguous: it could mean the number of"
    " UTF-8 bytes, the number of Unicode code points, or the number of"
    " user-visible characters (grapheme clusters). Use `s.byte_length()` or"
    " `len(s.bytes())` for the number of UTF-8 bytes, `len(s.codepoints())` for"
    " Unicode code points, or `len(s.graphemes())` for grapheme clusters."
)
def len(value: StringSlice) -> Int:
    ...


@always_inline
def len[T: Sized](value: T) -> Int:
    """Get the length of a value.

    Parameters:
        T: The Sized type.

    Args:
        value: The object to get the length of.

    Returns:
        The length of the object.
    """
    return value.__len__()


@always_inline
def len[T: SizedRaising](value: T) raises -> Int:
    """Get the length of a value.

    Parameters:
        T: The Sized type.

    Args:
        value: The object to get the length of.

    Returns:
        The length of the object.

    Raises:
        If the length cannot be computed.
    """
    return value.__len__()
