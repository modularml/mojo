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


from std.math import Floorable, floor


# This decorator limits the instance variable count to one
# and creates an implicit initializer for you.
@fieldwise_init("implicit")
struct FlooringInt:
    var floored: Int

    # This decorator allows implicit conversion of types
    # that can be floored (the decimal portion omitted)
    # and converted to an Integer
    @implicit
    def __init__[T: Floorable & Intable & Deinitable](out self, value: T):
        self.floored = Int(floor(value))


# This function returns the stored `floored` value within
# a `FlooringInt` instance.
def floored(value: FlooringInt) -> Int:
    return value.floored


def main():
    print(floored(FlooringInt(42)))  # pass `FlooringInt` instance, output: 42
    print(floored(2))  # pass Int, without `FlooringInt` constructor, output: 2
    print(floored(52.6))  # pass Float64, output: 52
    var x = BFloat16(192.3)
    print(floored(x))  # pass BFloat, output: 192
    var y: FlooringInt = 180
    print(y.floored)  # output 180
    var z: FlooringInt = 3.14159
    print(z.floored)  # output: 3
