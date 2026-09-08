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
"""File type constants and detection from stat system calls.

The `stat` package provides constants and utility functions for working with
file metadata from POSIX stat system calls. It defines standard file type bit
masks and predicates for determining file types from mode values. This package
enables portable file type checking across Unix-like systems.

Use this package when working with file system metadata, implementing portable
file type detection, or interfacing with POSIX file operations.
"""

from .stat import (
    S_IFBLK,
    S_IFCHR,
    S_IFDIR,
    S_IFIFO,
    S_IFLNK,
    S_IFMT,
    S_IFREG,
    S_IFSOCK,
    S_ISBLK,
    S_ISCHR,
    S_ISDIR,
    S_ISFIFO,
    S_ISLNK,
    S_ISREG,
    S_ISSOCK,
)
