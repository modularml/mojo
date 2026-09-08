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

# UNSUPPORTED: system-darwin
# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s

from std.python import Python
from std.sys import argv


def main() raises:
    var python = Python()

    # CHECK: This was built inside of python
    var py_string = Python.evaluate("'This was built' + ' inside of python'")
    print(python.as_string_slice(py_string.__str__()))
