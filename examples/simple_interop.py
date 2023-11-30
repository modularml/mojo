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


# Simple python program to test interop with Mojo.
# This file is imported from hello_interop.mojo.

import check_mod

check_mod.install_if_missing("numpy")
import numpy as np


def test_interop_func():
    print("Hello from Python!")
    a = np.array([1, 2, 3])
    print("I can even print a numpy array: ", a)


if __name__ == "__main__":
    from timeit import timeit

    print(timeit(lambda: test_interop_func(), number=1))
