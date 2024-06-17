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
# RUN: %mojo %s

from os.path import abspath, join, isfile
from pathlib import Path, cwd

from builtin._location import __source_location
from testing import assert_false, assert_true, assert_equal, assert_not_equal


def main():
    var this_dir = cwd().__fspath__()
    var this_file = __source_location().file_name.__str__()

    assert_equal(join(this_dir, this_file), abspath(this_file))
    assert_equal(join(this_dir, "file.txt"), abspath("file.txt"))
    assert_equal("/file.txt", abspath("/file.txt"))
    assert_equal(join(this_dir, this_file), abspath("/dir/../" + this_file))
    assert_true(isfile(abspath(Path(this_file))))
