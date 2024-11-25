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
# RUN: echo "Hello, World" | %mojo %s

from builtin.io import _fdopen
from testing import testing


fn test_stdin() raises:
    # "Hello, World" piped from RUN command above
    var stdin = _fdopen["r"](0)
    testing.assert_equal(stdin.read_until_delimiter(","), "Hello")
    testing.assert_equal(stdin.readline(), " World")


fn main() raises:
    test_stdin()
