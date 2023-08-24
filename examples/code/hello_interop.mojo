# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

# This sample demonstrates some basic Mojo
# Range and print functions available in the standard library
# It also demonstrates importing a simple Python program into Mojo

from python.python import (
    Python,
    _destroy_python,
    _init_python,
)


def main():
    print("Hello Mojo 🔥!")
    for x in range(9, 0, -3):
        print(x)
    alias test_dir = "/root/mojo-examples"
    try:
        Python.add_to_path(test_dir)
        let test_module: PythonObject = Python.import_module("simple_interop")
        if test_module:
            _ = test_module.test_interop_func()
        else:
            print("Could not locate module: ", test_module)
    except e:
        print(e.value)
        pass
