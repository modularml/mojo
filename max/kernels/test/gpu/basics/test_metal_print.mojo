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

from max.gpu.host import DeviceContext


# CHECK-LABEL: test_metal_print_basic
def test_metal_print_basic() raises:
    print("test_metal_print_basic")

    def do_print():
        print("Hello from Metal GPU")

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK: Hello from Metal GPU


# CHECK-LABEL: test_metal_print_int
def test_metal_print_int() raises:
    print("test_metal_print_int")

    def do_print(x_dev: Int32):
        var x = Int(x_dev)
        print("x =", x)

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](Int32(42), grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK: x = 42


# CHECK-LABEL: test_metal_print_float32
def test_metal_print_float32() raises:
    print("test_metal_print_float32")

    # Note: Apple GPU does not support Float64. Use Float32.
    def do_print(y: Float32):
        print("y =", y)

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](Float32(3.14), grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK: y = 3.14{{[0-9]*}}


# CHECK-LABEL: test_metal_print_empty
def test_metal_print_empty() raises:
    print("test_metal_print_empty")

    def do_print():
        print("")

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](grid_dim=1, block_dim=1)
        ctx.synchronize()


# CHECK-LABEL: test_metal_print_string_slice
def test_metal_print_string_slice() raises:
    print("test_metal_print_string_slice")

    # Exercises static_string globals passed as {ptr, i64} slices — the form
    # that was left unrewritten before this fix.
    def do_print(x: Int32):
        var label = "value"
        print(label, "=", Int(x))

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](Int32(7), grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK: value = 7


# CHECK-LABEL: test_metal_print_multiple_strings
def test_metal_print_multiple_strings() raises:
    print("test_metal_print_multiple_strings")

    def do_print():
        print("a =", 1)
        print("b =", 2)
        print("c =", 3)

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK: a = 1
    # CHECK: b = 2
    # CHECK: c = 3


def main() raises:
    test_metal_print_basic()
    test_metal_print_int()
    test_metal_print_float32()
    test_metal_print_empty()
    test_metal_print_string_slice()
    test_metal_print_multiple_strings()
