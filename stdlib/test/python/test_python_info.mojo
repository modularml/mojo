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
# RUN: %mojo-no-debug %s | FileCheck %s


from python._cpython import PythonVersion
from python import Python


fn test_python_version(inout python: Python):
    var version = "3.10.8 (main, Nov 24 2022, 08:08:27) [Clang 14.0.6 ]"
    var pythonVersion = PythonVersion(version)
    # CHECK: 3
    print(pythonVersion.major)
    # CHECK: 10
    print(pythonVersion.minor)
    # CHECK: 8
    print(pythonVersion.patch)


fn main():
    var python = Python()
    test_python_version(python)
