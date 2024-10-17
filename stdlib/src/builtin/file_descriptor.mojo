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
from utils import Span
from sys.ffi import external_call, OpaquePointer
from sys.info import triple_is_nvidia_cuda
from memory import UnsafePointer


@value
@register_passable("trivial")
struct FileDescriptor(Writer):
    """File descriptor of a file."""

    var value: Int
    """The underlying value of the file descriptor."""

    fn __init__(inout self, value: Int = 1):
        """Constructs the file descriptor from an integer.

        Args:
            value: The file identifier (Default 1 = stdout).
        """
        self.value = value

    fn __init__(inout self, f: FileHandle):
        """Constructs the file descriptor from a file handle.

        Args:
            f: The file handle.
        """
        self.value = f._get_raw_fd()

    @always_inline
    fn write_bytes(inout self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """

        @parameter
        if triple_is_nvidia_cuda():
            var tmp = 0
            var arg_ptr = UnsafePointer.address_of(tmp)
            # TODO: Debug assert to check bytes written when GPU print calls are buffered
            _ = external_call["vprintf", Int32](
                bytes.unsafe_ptr(), arg_ptr.bitcast[OpaquePointer]()
            )
        else:
            written = external_call["write", Int32](
                self.value, bytes.unsafe_ptr(), len(bytes)
            )
            debug_assert(
                written == len(bytes),
                "expected amount of bytes not written. expected: ",
                len(bytes),
                "but got: ",
                written,
            )

    fn write[*Ts: Writable](inout self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """

        @parameter
        fn write_arg[i: Int, T: Writable](arg: T):
            arg.write_to(self)

        args.each_idx[write_arg]()
