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

"""Implement a generic `HeapValue` type.

You can import it from the `memory` package. For example:

```mojo
from memory import HeapValue
```
"""

@register_passable
struct HeapValue[T:AnyType]:
    """Created by moving a value in, destroyed by moving the value out.

    If the value has not been moved to the stack (`instance^.move_value_out()`),
    
    `T.__del__` will dispose of the value and the heap memory will be freed.

    `move_value_out(owned self)` destroys the instance of `HeapValue` inplace.

    That way an instance of HeapValue can only exist with an initialized value.

    For thoses reasons, `.create_copy()` exists in favour of `__copyinit__`.
    
    Example usage:

    ```mojo
    var A = HeapValue[String]("A")
    var B = A.create_copy()
    print(A[], B[]) # A A
    B[] = "B"
    print(A[], B[]) # A B
    
    var a: String = A^.move_value_out()
    var b: String = B^.move_value_out()
    print(a, b) #A B
    ```

    """
    var _data: UnsafePointer[T]
    
    fn __init__[T:Movable](inout self: HeapValue[T], owned arg: T):
        "Initialize an `HeapValue` by moving a value in."
        self._data = UnsafePointer[T].alloc(1)
        self._data.init_pointee_move(arg^)
    
    fn __del__(owned self):
        if self._data != UnsafePointer[T]():
            self._data.destroy_pointee()
            self._data.free()

    fn move_value_out[T:Movable](owned self: HeapValue[T]) -> T:
        """Destroys the instance, move its value out of the heap, returns it.

        `self` must be passed as `owned` using the transfer suffix (`^`).
        
        The memory that was used to hold the value on the heap is freed.

        Example usage:

        ```mojo
        var A = HeapValue[String]("Mojo")
        var B: String = A^.move_value_out()
        print(B) # Mojo
        ```

        """
        var tmp = self._data.take_pointee()
        self._data.free()
        self._data = UnsafePointer[T]()
        return tmp^
    
    fn move_copy_out[T:Copyable](self: HeapValue[T]) -> T:
        """Create a copy of the value stored on the heap and returns it.

        Example usage:

        ```mojo
        var A = HeapValue[String]("Mojo")
        var B: String = A.move_copy_out()
        print(B) # Mojo
        print(A[]) #Mojo
        ```

        """
        var tmp: T
        T.__copyinit__(tmp,self._data[])
        return tmp^

    fn create_copy[T:Copyable](self: HeapValue[T]) -> HeapValue[T]:
        """Creates a copy of the value and store it into a new `HeapValue`.
        
        Example usage:

        ```mojo
        var x = HeapValue[Int](1)
        var y = x.create_copy()
        print(x[], y[]) # 1 1
        y[] = 2
        print(x[], y[]) # 1 2
        ```

        """
        var ptr = UnsafePointer[T].alloc(1)
        T.__copyinit__(
            __get_address_as_uninit_lvalue(ptr.address),
            self._data[]
        )
        return HeapValue[T]{
            _data : ptr
        }

    fn move_into_another_heapvalue[T:Movable](
        owned self: HeapValue[T], 
        inout other: HeapValue[T]
    ):
        """Destroys the instance and move its value into another `HeapValue`.

        `self` have to be passed as `owned` using the `^` transfer suffix.

        It ensures that self free its own memory and destroys its own instance.

        The value will be stored on the current heap memory of other.

        Example usage:
        
        ```mojo
        var x = HeapValue[Int](1)
        var y = x.create_copy()
        y[] = 2
        y^.move_into_another_heapvalue(x)
        print(x[]) # 2
        ```

        """
        debug_assert(
            other._data != UnsafePointer[T](),
            "Unexpected behaviour"
        )
        other._data[] = self._data.take_pointee()
        self._data.free()
        self._data = UnsafePointer[T]()
        HeapValue[T].__del__(self^)

    fn copy_into_another_heapvalue[T:Copyable](
        self: HeapValue[T],
        inout other: HeapValue[T]
    ):
        """Copy the stored value into another existing `HeapValue`.

        The value will be stored on the current heap memory of other.

        Example usage:

        ```mojo
        var x = HeapValue[Int](1)
        var y = HeapValue[Int](2)
        y.copy_into_another_heapvalue(x)
        print(x[], y[]) # 2 2
        ```

        """
        debug_assert(
            other._data != UnsafePointer[T](),
            "Unexpected behaviour"
        )
        other._data[] = self._data[]

    fn __getitem__[T:AnyType](
        ref[_]self: HeapValue[T]
    ) -> ref [__lifetime_of(self)] T:
        """Returns a Reference to the value stored on the heap.
        
        Example usage:

        ```mojo
        var x = HeapValue[Int](1)
        print(x[]) # 1
        x[] = 2
        print(x[]) # 2
        ```

        Returns:
            A Reference to the value stored on the heap.
        """
        return self._data[]

    fn mut_ref[L:MutableLifetime](ref[L]self)->ref[L]T:
        """Returns a mutable reference to the value.

        Example usage:
        ```mojo
        fn mutate(inout arg: Int): arg += 1
        var x = HeapValue[Int](1)
        mutate(x.mut_ref())
        print(x[]) # 2
        ```
        
        """
        return self._data[]

    fn immut_ref[L:ImmutableLifetime](ref[L]self)->ref[L]T:
        """Returns an immutable reference to the value.
        
        Example usage:

        ```mojo
        fn print_arg(arg: Int): print(arg)
        var x = HeapValue[Int](1)
        print_arg(x.immut_ref())
        ```

        """
        return self._data[]
