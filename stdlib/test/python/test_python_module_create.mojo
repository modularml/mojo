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
<<<<<<<< HEAD:stdlib/test/builtin/test_range_uint_reverse_range_bad.mojo
# REQUIRES: has_not
# RUN: not --crash mojo -D MOJO_ENABLE_ASSERTIONS %s 2>&1
========
# TODO(MSTDL-875): Fix and un-XFAIL this
# XFAIL: asan && !system-darwin
# RUN: %mojo %s
>>>>>>>> origin/nightly:stdlib/test/python/test_python_module_create.mojo

from python import Python, PythonObject
from testing import assert_equal


<<<<<<<< HEAD:stdlib/test/builtin/test_range_uint_reverse_range_bad.mojo
def test_range_uint_bad_step_size():
    # Ensure constructing a range with a "-1" step size (i.e. reverse range)
    # with UInt is rejected and aborts now via `debug_assert` handler.
    var r = range(UInt(0), UInt(10), UInt(Int(-1)))


def main():
    test_range_uint_bad_step_size()
========
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
>>>>>>>> origin/nightly:stdlib/test/python/test_python_module_create.mojo
