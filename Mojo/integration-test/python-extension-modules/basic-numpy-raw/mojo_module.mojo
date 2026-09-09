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

from std.os import abort

from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    """Create a Python module with a function binding for `mojo_incr_np_array`.
    """

    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_function[mojo_incr_np_array](
            "mojo_incr_np_array",
            docstring="Increment the contents of a numpy array by one",
        )
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


@fieldwise_init
struct PyArrayObject[dtype: DType](ImplicitlyCopyable):
    """
    Container for a numpy array.

    See: https://numpy.org/doc/2.1/reference/c-api/types-and-structures.html#c.PyArrayObject
    """

    var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var nd: Int

    var dimensions: Pointer[Int, MutUntrackedOrigin]

    var strides: Pointer[Int, MutUntrackedOrigin]
    var base: PyObjectPtr
    var descr: PyObjectPtr
    var flags: Int
    var weakreflist: PyObjectPtr

    # version dependent private members are omitted
    # ...


def mojo_incr_np_array(
    py_array_object: PythonObject,
) raises -> PythonObject:
    comptime dtype = DType.int32

    print("Hello from mojo_incr_np_array")

    var py_array_object_ptr = Pointer[PyArrayObject[dtype], ...](
        unchecked_downcast_value=py_array_object
    )

    var nd = py_array_object_ptr[].nd
    var data_ptr = py_array_object_ptr[].data

    # Print each field of the struct
    print()
    print("Numpy Array Struct:")
    print("  data:", String(data_ptr))
    print("  nd:", nd)
    print("  dimensions:", end=" ")
    for i in range(nd):
        print(py_array_object_ptr[].dimensions[unsafe_offset=i], end=" ")
    print()
    print("  strides:", end=" ")
    for i in range(nd):
        print(py_array_object_ptr[].strides[unsafe_offset=i], end=" ")
    print()
    print("  descr:", py_array_object_ptr[].descr)
    print("  flags:", hex(py_array_object_ptr[].flags))
    print(
        "  weakreflist:",
        py_array_object_ptr[].weakreflist,
    )
    print()

    var num_elts = 1
    for i in range(nd):
        var dim = py_array_object_ptr[].dimensions[unsafe_offset=i]
        num_elts *= dim

    for i in range(num_elts):
        data_ptr[unsafe_offset=i] += 1

    print("Goodbye from mojo_incr_np_array")
    return PythonObject(None)
