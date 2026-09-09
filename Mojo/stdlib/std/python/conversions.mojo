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
"""Implements conversion traits to and from PythonObject.

You can import these APIs from the `python` package. For example:

```mojo
from std.python import ConvertibleToPython
```
"""

from . import PythonObject


trait ConvertibleToPython(Deinitable):
    """A trait that indicates a type can be converted to a PythonObject, and
    that specifies the behavior with a `to_python_object` method."""

    def to_python_object(var self) raises -> PythonObject:
        """Convert a value to a PythonObject.

        Returns:
            A PythonObject representing the value.

        Raises:
            If the conversion to a PythonObject failed.
        """
        ...


trait ConvertibleFromPython(Copyable, Deinitable):
    """Denotes a type that can attempt construction from a read-only Python
    object.
    """

    def __init__(out self, *, py: PythonObject) raises:
        """Attempt to construct an instance of this object from a read-only
        Python value.

        Args:
            py: The Python object to convert from.

        Raises:
            If conversion was not successful.
        """
        ...
