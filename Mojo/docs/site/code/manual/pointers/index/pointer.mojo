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

from std.memory import Pointer


def main():
    var some_value: Int = 42
    # start-pointer-create
    var ptr = Pointer(to=some_value)
    # end-pointer-create

    print(ptr[])

    # Pointers can be copied
    # start-pointer-copy
    var copied_ptr = ptr
    # end-pointer-copy

    print(copied_ptr[])
