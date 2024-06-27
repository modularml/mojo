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


from os.path import exists, lexists
from pathlib import Path, cwd

from builtin._location import __source_location
from testing import assert_false, assert_true


def main():
    assert_true(exists(__source_location().file_name))
    assert_true(lexists(__source_location().file_name))

    assert_false(exists("this/file/does/not/exist"))
    assert_false(lexists("this/file/does/not/exist"))

    assert_true(exists(cwd()))
    assert_true(lexists(cwd()))

    assert_true(exists(Path()))
    assert_true(lexists(Path()))
