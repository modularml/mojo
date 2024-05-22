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
"""Higher level abstraction for file stream.

These are Mojo built-ins, so you don't need to import them.

For example, here's how to print to a file

```mojo
var f = open("my_file.txt", "r")
print("hello", file=f)
f.close()
```

"""


struct FileDescriptor:
    """File descriptor of a file."""

    var value: Int
    """The underlying value of the file descriptor."""

    fn __init__(inout self):
        """Default constructor to stdout."""
        self.value = 1

    fn __init__(inout self, x: Int):
        """Constructs the file descriptor from an integer.

        Args:
            x: The integer.
        """
        self.value = x

    fn __init__(inout self, f: FileHandle):
        """Constructs the file descriptor from a file handle.

        Args:
            f: The file handle.
        """
        self.value = f._get_raw_fd()
