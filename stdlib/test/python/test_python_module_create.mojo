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
# TODO(MSTDL-875): Fix and un-XFAIL this
# XFAIL: asan && !system-darwin
# RUN: %mojo %s

from python import Python, PythonObject
from testing import assert_equal


def test_create_module():
    var module_name = "test_module"
    var module = Python.create_module(module_name)

    # TODO: inspect properties about the module
    # First though, let's see if we can even import it
    # var imported_module = Python.import_module(module_name)
    #
    # _ = module_name


def main():
    test_create_module()
