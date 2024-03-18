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
"""Implements Python Dictionary.

You can import these APIs from the `python` package. For example:

```mojo
from python.dictionary import Dictionary
```
"""

from memory.unsafe import Pointer

from utils.loop import unroll

from ._cpython import CPython, PyKeyValuePair, PyObjectPtr
from .object import PythonObject
from .python import Python, _get_global_python_itf


struct _PyIter(Sized):
    """A Python iterator."""

    var dict: Dictionary
    """The source dictionary."""
    var nextItem: PyKeyValuePair
    """The next item in the dictionary."""

    fn __copyinit__(inout self, existing: Self):
        """Copy another iterator.

        Args:
            existing: The source _PyIter.
        """
        self.dict = existing.dict
        self.nextItem = existing.nextItem

    fn __init__(inout self, dict: Dictionary):
        """Initialize an iterator.

        Args:
            dict: The Dictionary to traverse.
        """
        var cpython = _get_global_python_itf().cpython()
        self.dict = dict
        self.nextItem = cpython.PyDict_Next(self.dict.py_object, 0)

    fn __next__(inout self: _PyIter) -> PythonObject:
        """Return the next item and update to point to subsequent item.

        Returns:
            The next item in the sequence.
        """
        var result = self.nextItem.key
        var cpython = _get_global_python_itf().cpython()
        # PyDict_Next borrows the key and value but the PythonObject
        # instance we wrap it in steals it. To maintain accurate ref
        # counting, we force a steal by incrementing the ref count.
        cpython.Py_IncRef(result)
        if self.nextItem.success:
            self.nextItem = cpython.PyDict_Next(
                self.dict.py_object, self.nextItem.position
            )
        return result

    fn __len__(self) -> Int:
        """Halting condition.

        Returns:
            0 to halt traversal, 1 otherwise.
        """
        if self.nextItem.success:
            return 1
        else:
            return 0


struct Dictionary(Boolable):
    """A Python dictionary."""

    var py_object: PyObjectPtr
    """The underlying pyobject pointer of this dictionary."""

    fn __init__(inout self, dict: PyObjectPtr):
        """Initialize the dictionary with the given pyobject pointer.

        Args:
            dict: The pyobject pointer used to initialize the new dictionary.
        """
        self.py_object = dict

    fn __copyinit__(inout self, existing: Self):
        """Copy the dictionary.

        This increments the underlying refcount of the existing object.

        Args:
            existing: The value to copy.
        """
        self.py_object = existing.py_object
        var cpython = _get_global_python_itf().cpython()
        cpython.Py_IncRef(self.py_object)

    fn __del__(owned self):
        """Destroy the object.

        This decrements the underlying refcount of the pointed-to object.
        """
        var cpython = _get_global_python_itf().cpython()
        if not self.py_object.is_null():
            cpython.Py_DecRef(self.py_object)

    fn __setitem__(self, key: PythonObject, value: PythonObject) raises:
        """Sets the value with the specified key.

        Args:
          key: The key of the value to set.
          value: The value to store.
        """
        self._setitem(key.py_object, value.py_object)

    fn __setitem__(self, key: PythonObject, value: Dictionary) raises:
        """Sets the value with the specified key.

        Args:
          key: The key of the value to set.
          value: The value to store.
        """
        self._setitem(key.py_object, value.py_object)

    fn _setitem(self, key: PyObjectPtr, value: PyObjectPtr) raises:
        var cpython = _get_global_python_itf().cpython()
        var result = cpython.PyDict_SetItem(self.py_object, key, value)
        Python.throw_python_exception_if_error_state(cpython)

    fn __getitem__(self, key: PythonObject) raises -> PythonObject:
        """Gets the object at the specified key.

        Args:
          key: The key of the value to retrieve.

        Returns:
          The value at the specified key.
        """
        var cpython = _get_global_python_itf().cpython()
        var result = cpython.PyDict_GetItemWithError(
            self.py_object, key.py_object
        )
        Python.throw_python_exception_if_error_state(cpython)
        if result.is_null():
            raise Error("Attribute is not found.")
        cpython.Py_IncRef(result)
        return PythonObject(result)

    fn __bool__(self) -> Bool:
        """Whether or not the Dictionary is nonnull.

        Returns:
            True if the Dictionary is nonnull, False otherwise.
        """
        return not self.py_object.is_null()

    fn __iter__(inout self) -> _PyIter:
        """Iterate over the dictionary.

        Returns:
            An iterator object.
        """
        return _PyIter(self)

    fn __getattr__(self, name: StringLiteral) raises -> PythonObject:
        """Return the value of the dictionary attribute with the given name.

        Args:
            name: The name of the dictionary attribute to return.

        Returns:
            The value of the dictionary attribute with the given name.
        """
        var cpython = _get_global_python_itf().cpython()
        var result = cpython.PyObject_GetAttrString(self.py_object, name)
        Python.throw_python_exception_if_error_state(cpython)
        if result.is_null():
            raise Error("Attribute is not found.")
        return PythonObject(result)
