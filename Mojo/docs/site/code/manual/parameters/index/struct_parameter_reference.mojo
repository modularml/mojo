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


struct Circle[radius: Float64]:
    comptime pi = 3.14159265359
    comptime circumference = 2.0 * Self.pi * Self.radius


# start-reference-parameter-on-type
def on_type():
    print(Int(SIMD[DType.float32, 2].length))  # prints 2
    # end-reference-parameter-on-type


# start-reference-parameter-on-instance
def on_instance():
    var x = SIMD[.int32, 2](4, 8)
    print(x.dtype)  # prints int32
    # end-reference-parameter-on-instance
    _ = x


def slice_example():
    # start-simd-slice-example
    var m = SIMD[.int32, 4](1, 3, 5, 7)
    var n = m.slice[2]()
    print(n)  # prints [1, 3]
    # end-simd-slice-example


def main():
    on_type()
    on_instance()
    slice_example()
