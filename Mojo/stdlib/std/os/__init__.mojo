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
"""OS interface layer: environment, filesystem, process control.

The `os` package provides platform-independent access to operating system
functionality including filesystem operations, environment variables, and
process control. It offers portable interfaces to OS-dependent features,
abstracting platform differences while exposing system-level capabilities. This
package serves as the foundation for system programming in Mojo.

Use this package for system-level operations, filesystem management,
environment configuration, or platform abstraction. For file I/O operations,
use the built-in `open()` function. For path manipulation, see the `os.path`
package for path functions or the `pathlib` package for the object-oriented
`Path` type.
"""

from .env import getenv, setenv, unsetenv
from .fstat import lstat, stat, stat_result
from .os import (
    SEEK_CUR,
    SEEK_END,
    SEEK_SET,
    abort,
    getuid,
    isatty,
    listdir,
    makedirs,
    mkdir,
    remove,
    removedirs,
    rmdir,
    sep,
    unlink,
    symlink,
    link,
    chdir,
)
from .pathlike import PathLike
from .process import Process, Pipe

from . import path
