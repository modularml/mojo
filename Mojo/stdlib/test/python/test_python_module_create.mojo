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

from std.python import Python
from std.testing import TestSuite


def test_create_module() raises:
    var module = Python.create_module("test_module")

    # TODO: inspect properties about the module
    # First though, let's see if we can even import it
    # var imported_module = Python.import_module(module_name)
    #
    # _ = module_name


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
