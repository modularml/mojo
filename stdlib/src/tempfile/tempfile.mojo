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
"""Implements tempfile methods.

You can import a method from the `tempfile` package. For example:

```mojo
from tempfile import gettempdir
```
"""

from collections import Optional
import os
import sys
from pathlib import Path


alias TMP_MAX = 10_000


fn _get_random_name(size: Int = 8) -> String:
    alias characters = String("abcdefghijklmnopqrstuvwxyz0123456789_")
    var name_list = List[UInt8](capacity=size + 1)
    for _ in range(size):
        var rand_index = int(random.random_ui64(0, len(characters) - 1))
        name_list.append(ord(characters[rand_index]))
    name_list.append(0)
    return String(name_list^)


fn _candidate_tempdir_list() -> List[String]:
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""

    constrained[not sys.os_is_windows(), "windows not supported yet"]()

    var dirlist = List[String]()
    var possible_env_vars = List("TMPDIR", "TEMP", "TMP")
    var dirname: String

    # First, try the environment.
    for env_var in possible_env_vars:
        if dirname := os.getenv(env_var[]):
            dirlist.append(dirname^)

    # Failing that, try OS-specific locations.
    dirlist.extend(List(String("/tmp"), String("/var/tmp"), String("/usr/tmp")))

    # As a last resort, the current directory if possible,
    # os.path.getcwd() could raise
    try:
        dirlist.append(Path())
    except:
        pass

    return dirlist


fn _get_default_tempdir() raises -> String:
    """Calculate the default directory to use for temporary files.

    We determine whether or not a candidate temp dir is usable by
    trying to create and write to a file in that directory.  If this
    is successful, the test file is deleted. To prevent denial of
    service, the name of the test file must be randomized."""

    var dirlist = _candidate_tempdir_list()

    for dir_name in dirlist:
        if not os.path.isdir(dir_name[]):
            continue
        if _try_to_create_file(dir_name[]):
            return dir_name[]

    raise Error("No usable temporary directory found")


fn _try_to_create_file(dir: String) -> Bool:
    for _ in range(TMP_MAX):
        var name = _get_random_name()
        # TODO use os.join when it exists
        var filename = Path(dir) / name

        # prevent overwriting existing file
        if os.path.exists(filename):
            continue

        # verify that we have writing access in the target directory
        try:
            with FileHandle(filename, "w"):
                pass
            os.remove(filename)
            return True
        except:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
            return False

    return False


fn gettempdir() -> Optional[String]:
    """Return the default directory to use for temporary files.

    Returns:
        The name of the default temporary directory.
    """
    # TODO In python _get_default_tempdir is called exactly once so that the default
    # tmp dir is the same throughout the program execution/
    # Since there is no global scope in mojo yet, this is not possible for now.
    try:
        return _get_default_tempdir()
    except:
        return None


fn mkdtemp(
    suffix: String = "", prefix: String = "tmp", dir: Optional[String] = None
) raises -> String:
    """Create a temporary directory.
    Caller is responsible for deleting the directory when done with it.

    Args:
        suffix: Suffix to use for the directory name.
        prefix: Prefix to use for the directory name.
        dir: Directory in which the directory will be created.

    Returns:
        The name of the created directory.

    Raises:
        If the directory can not be created.
    """
    var final_dir: Path
    if not dir:
        final_dir = Path(_get_default_tempdir())
    else:
        final_dir = Path(dir.value()[])

    for _ in range(TMP_MAX):
        var dir_name = final_dir / (prefix + _get_random_name() + suffix)
        if os.path.exists(dir_name):
            continue
        try:
            os.mkdir(dir_name, mode=0o700)
            # TODO for now this name could be relative,
            # python implementation expands the path,
            # but several functions are not yet implemented in mojo
            # i.e. abspath, normpath
            return dir_name
        except:
            continue
    raise Error("Failed to create temporary file")
