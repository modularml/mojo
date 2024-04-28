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
"""Implements a number of high-level operations on files and
collections of files.

You can import these APIs from the `shutil` package. For example:

```mojo
from shutil import rmtree
```
"""

from os import listdir
from os.path import islink, isdir, isfile


fn rmtree(path: String, ignore_errors: Bool = False) raises:
    """Removes the specified directory and all its contents.
    If the path is a symbolic link, an error is raised.
    If ignore_errors is True, errors resulting from failed removals will be ignored.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Args:
      path: The path to the directory.
      ignore_errors: Whether to ignore errors.
    """
    if os.path.islink(path):
        raise Error("`path`can not be a symbolic link")

    for file_or_dir in listdir(path):
        if isfile(path + "/" + file_or_dir[]):
            try:
                os.remove(path + "/" + file_or_dir[])
            except e:
                if not ignore_errors:
                    raise Error(e)
            continue
        if isdir(path + "/" + file_or_dir[]):
            try:
                rmtree(path + "/" + file_or_dir[], ignore_errors)
            except e:
                if ignore_errors:
                    continue
                raise Error(e)

    os.rmdir(path)


fn rmtree[
    pathlike: os.PathLike
](path: pathlike, ignore_errors: Bool = False) raises:
    """Removes the specified directory and all its contents.
    If the path is a symbolic link, an error is raised.
    If ignore_errors is True, errors resulting from failed removals will be ignored.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Args:
      path: The path to the directory.
      ignore_errors: Whether to ignore errors.
    """
    rmtree(path.__fspath__(), ignore_errors)
