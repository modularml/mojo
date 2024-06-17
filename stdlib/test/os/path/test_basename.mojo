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

from os.path import basename
from testing import assert_equal


def main():
    assert_equal(basename("a/path/to/file.txt"), "file.txt")
    assert_equal(basename("a/path/to/"), "")
    assert_equal(basename("a/path/to"), "to")
    assert_equal(basename("a/path/to"), "to")
    assert_equal(basename(""), "")
    assert_equal(basename("/"), "")
