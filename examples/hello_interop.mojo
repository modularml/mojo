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
    try:
        Python.add_to_path(".")
        Python.add_to_path("./examples")
        let test_module = Python.import_module("simple_interop")
        test_module.test_interop_func()
    except e:
        print(e.value)
        print("could not find module simple_interop")
