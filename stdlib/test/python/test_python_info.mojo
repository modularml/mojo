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
# XFAIL: asan && !system-darwin
# RUN: %mojo %s


from python import Python
from python._cpython import PythonVersion
from testing import assert_equal


fn test_python_version(inout python: Python) raises:
    var version = "3.10.8 (main, Nov 24 2022, 08:08:27) [Clang 14.0.6 ]"
    var pythonVersion = PythonVersion(version)
    assert_equal(pythonVersion.major, 3)
    assert_equal(pythonVersion.minor, 10)
    assert_equal(pythonVersion.patch, 8)


def main():
    var python = Python()
    test_python_version(python)
