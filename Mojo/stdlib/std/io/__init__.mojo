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
"""Core I/O operations: console input/output, file handling, writing traits.

The `io` package provides fundamental input/output functionality for reading
from and writing to various sources including the console, files, and custom
streams. It defines the core traits for the I/O system (`Writer` and
`Writable`) that enable formatted output across different backends, along with
concrete implementations for file operations and standard streams.

Use this package for console interaction, file operations, implementing custom
output formatting for your types, or building I/O abstractions. Most programs
use `print()` and file operations from this package, while library authors
implement `Writable` to enable their types to work with any `Writer`.
"""

from std.format import Writable, Writer
from .file import FileHandle
from .file_descriptor import FileDescriptor
from .io import input, print
