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

import std.sys
from std.collections import Set
from std.ffi import external_call
from std.pathlib import Path
from std.python import Python
from std import os

# We can't check much more than this at the moment, because the license year
# changes and the language is not mature enough to do regex yet.
comptime LICENSE = """# ===----------------------------------------------------------------------=== #
# Copyright (c)"""

# NOTE: This copyright year needs to be updated (m)annually
comptime LICENSE_TO_ADD = """# ===----------------------------------------------------------------------=== #
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

"""


def is_ignored_file(filename: StringSlice) -> Bool:
    if not (filename.endswith(".py") or filename.endswith(".mojo")):
        return True

    # Generated files
    if (
        filename == "max/python/max/serve/schemas/kserve.py"
        or filename == "max/python/max/serve/schemas/openai.py"
    ):
        return True

    return False


def get_git_files() raises -> Set[String]:
    # Defer file discovery to the shared lint_helpers Python library so this
    # linter stays consistent with the other wrappers (FAST picks changed vs.
    # all files; jj/git is handled there).
    var lint_helpers = Python.import_module("lint_helpers")

    var py_files = lint_helpers.get_changed_files() if Bool(
        py=lint_helpers.is_fast()
    ) else lint_helpers.get_all_files()

    var result = Set[String]()
    for file in py_files:
        var name = String(py=file)
        if not is_ignored_file(name):
            result.add(name)
    return result^


def check_path(path: Path, mut files_without_license: List[Path]) raises:
    var file_text = path.read_text()

    var has_license: Bool
    # Ignore #! in scripts
    if file_text.startswith("#!"):
        has_license = "\n".join(List(file_text.splitlines()[1:])).startswith(
            LICENSE
        )
    else:
        has_license = file_text.startswith(LICENSE)

    if not has_license:
        files_without_license.append(path)


def main() raises:
    # Import lint_helpers (and thereby initialize the embedded CPython) before
    # the chdir below: libpython is dlopen'd via a runfiles-relative path, so it
    # must be loaded while the current directory is still the runfiles root.
    var lint_helpers = Python.import_module("lint_helpers")

    var workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        # TODO: this should be in stdlib
        _ = external_call["chdir", Int32](workspace.as_c_string_slice())

    var target_paths = std.sys.argv()

    var fix = False
    for arg in target_paths:
        if arg == "--fix":
            fix = True
            break
    if os.getenv("CHECK"):
        fix = not Bool(py=lint_helpers.is_check())

    var files_without_license = List[Path]()
    if (
        len(target_paths) < 2
        or len(target_paths) == 2
        and target_paths[1] == "--fix"
    ):
        for file in get_git_files():
            check_path(file, files_without_license)
    else:
        for i in range(len(target_paths)):
            if i == 0:
                # this is the current file
                continue
            if target_paths[i] == "--fix":
                continue
            check_path(Path(target_paths[i]), files_without_license)

    if len(files_without_license) > 0:
        if fix:
            print("Appending copyright notices to the following files:")
            for file in files_without_license:
                print(file)
                var content = file.read_text()
                file.write_text(LICENSE_TO_ADD + content)
        else:
            print("The following files have missing licences 💥 💔 💥")
            for file in files_without_license:
                print(file)
            print("Please add the license to each file before committing.")
            print("You can run `./bazelw run format` to do this automatically.")
            std.sys.exit(1)
