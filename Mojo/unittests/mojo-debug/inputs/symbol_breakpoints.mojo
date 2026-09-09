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


def simple_fn(x: Int):
    print(x)  # simple_fn stop


# Passing a generic value into a variadic pack (`String(x)`, `print(x)`)
# attributes the pack-setup code to the `def` line, so the breakpoint would
# stop on the signature instead of the first statement. Build the string with
# non-variadic `write` calls instead.
def parametrized_fn[T: Writable](x: T):
    var s = String()  # parametrized_fn stop
    s.write(x)
    print(s)


@fieldwise_init
struct Struct[T1: Writable]:
    def parametrized_method[T2: Writable](self, x: Self.T1, y: T2):
        var s = String()  # parametrized_method stop
        s.write(x)
        s.write(y)
        print(s)


def main():
    print("start")  # breakpoint
    simple_fn(12)
    parametrized_fn[Int](13)
    Struct[Float32]().parametrized_method[Int](12.25, 13)
