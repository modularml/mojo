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

from builtin.io import stdin, input
from testing import testing


fn test_stdin() raises:
    # stdin input: Hello, World
    testing.assert_equal(stdin().read_until_delimiter(","), "Hello")

    # stdin input: Hello, World
    with stdin() as s:
        testing.assert_equal(s.readline(), "Hello, World")


fn test_input() raises:
    # stdin input: Mojo
    testing.assert_equal(input("What's your name?"), "Mojo")


fn main() raises:
    test_stdin()
    test_input()
