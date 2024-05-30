# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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


# TODO: Make this public when we are certain of the design.
# TODO: Move _size into an alias when the bug https://github.com/modularml/mojo/issues/2889
# is fixed.
struct _MaybeUninitialized[ElementType: CollectionElement, _size: Int = 1](
    CollectionElement
):
    alias type = __mlir_type[
        `!pop.array<`, _size.value, `, `, Self.ElementType, `>`
    ]
    var _array: Self.type

    @always_inline
    fn __init__(inout self):
        """The mpemory is now considered uninitialized."""
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

    @always_inline
    fn __init__(inout self, owned value: Self.ElementType):
        """The memory is now considered initialized."""
        self = Self()
        self.write(value^)

    @always_inline
    fn __copyinit__(inout self, other: Self):
        """Calling this method assumes that the memory is initialized."""
        self = Self()
        initialize_pointee_copy(self.unsafe_ptr(), other.assume_initialized())

    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        """Calling this method assumes that the memory is initialized."""
        self = Self()
        move_pointee(src=other.unsafe_ptr(), dst=self.unsafe_ptr())

    @always_inline
    fn write(inout self, owned value: Self.ElementType):
        """Calling this method assumes that the memory is uninitialized."""
        self.unsafe_ptr()[] = value^

    @always_inline
    fn assume_initialized(
        self: Reference[Self, _, _]
    ) -> ref [self.lifetime] Self.ElementType:
        """Calling this method assumes that the memory is initialized."""
        return self[].unsafe_ptr()[]

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Self.ElementType]:
        """Get a pointer to the underlying element."""
        return UnsafePointer(self._array).bitcast[Self.ElementType]()

    @always_inline
    fn assume_initialized_destroy(inout self):
        """Calling this method assumes that the memory is initialized."""
        destroy_pointee(self.unsafe_ptr())

    @always_inline
    fn __del__(owned self):
        """Calling this method assumes that the memory is uninitialized. This is a no-op.

        If the memory was initialized, the caller should use `assume_initialized_destroy` before.
        """
        pass
