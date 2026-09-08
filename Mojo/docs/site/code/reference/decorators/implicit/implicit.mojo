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


struct MyInt:
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, value: Float64):
        self.value = Int(value)


def func(n: MyInt):
    print("MyInt value: ", n.value)


def main():
    func(Int(42))  # Implicit conversion from Int: OK
    func(MyInt(Float64(4.2)))  # Explicit conversion from Float64: OK
    # func(Float64(4.2))  # Error: can't convert Float64 to MyInt
