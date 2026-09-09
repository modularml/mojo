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
"""Defines `AnyType`, the most basic trait that all Mojo types extend by default.

This is a Mojo built-in, so you don't need to import it.
"""


@stable(since="1.0")
trait AnyType:
    """The most basic trait that all Mojo types extend by default.

    All Mojo struct types always conform to `AnyType`. This trait imposes no
    requirements on the types that conform to it, not even that they provide
    a `__deinit__()` implicit destructor.

    A type that conforms to `AnyType` but not to `Deinitable` is
    called a linear type, also known as a non-`Deinitable` type.

    Generic code will commonly want to use `T: Deinitable` instead
    of `T: AnyType`.

    **`AnyType`, Object Destructors, and Linear Types**

    Mojo's `AnyType` is a lower-level, more powerful building block than is
    found in many mainstream programming languages today.

    In most programming languages that enforce strong object initialization and
    destruction lifecycle semantics ("RAII"), the programmer is required to
    define a destructor function that will "tear down" an object instance and
    release any resources it has logical ownership over. In such languages, the
    compiler is permitted to destroy an object instance *implicitly* whenever it
    determines that the instance is no longer used, or has "gone out of scope".

    Another way to state the above is that, in many programming languages, the
    minimum requirement of all types is that they provide *at least* a
    trivial (possibly empty) destructor function. Mojo's `AnyType` is more
    basic than that seemingly minimum requirement.

    *Unlike* in programming languages the reader is likely to be familiar with,
    Mojo enforces strong object lifecycles, but does *not* require that a type
    provide an implicitly-callable destructor function. Instead, a type may
    choose to provide only named, explicitly-callable destructor methods.

    Said another way, Mojo gives type authors a type to provide either:

    * A `__deinit__()` destructor method that the compiler may call implicitly
      whenever an owned object instances has no further uses. Such types
      conform to `Deinitable`.

    * Named destructor methods that type user must choose to call explicitly.
      Failing to explicitly destroy such a type will lead to a compile-time
      error, requiring the programmer to chose how to destroy the object or
      keep it alive for longer.

    (Technically, a type can choose to provide *neither* implicit nor named
    destructors, but an instance of such a type would effectively be a
    "hot potato", getting tossed along forever without any way to "consume" the
    instance.)

    Named destructors give library type authors a powerful tool to enforce
    correctness and safety invariants. A type that provides only named
    destructor methods (a linear type) makes object destruction the explicit
    choice of the downstream user, instead of something done implicitly when the
    compiler thinks it is appropriate.

    Linear types can act as a guard that some explicit action must be performed
    sometime "in the future" after initial object construction.

    The following is a simple example of a non-`Deinitable` type with
    a named destructor method:

    ```mojo
    from std.pathlib import Path

    struct FileBuffer(Deinitable where False):
        def __init__(out self, path: Path):
            pass  # ... open the file at the specified `path` ...

        def write(self, data: Some[Writable]):
            pass  # ... buffered write of the specified data to this file ...

        def save_and_close(deinit self):
            pass  # ... save out the buffered data ...

    # ERROR: 'file' abandoned without being explicitly destroyed
    # def write_greeting_to_file(var file: FileBuffer):
    #     file.write("Hello there!")
    #
    # FIX: add `file^.save_and_close()`
    ```

    In the above example, the user is saved from forgetting to flush any
    buffered data because `FileBuffer` cannot simply be "dropped on the floor" —
    the programmer must choose to call `FileBuffer.save_and_close()` when they
    are finished with `file`.

    The `FileBuffer.save_and_close()` method is special because it takes
    `deinit self`. The `deinit` argument convention is special, and signals that
    the object is consumed by calling that method, with no further tear down
    logic required.
    """

    pass
