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


@fieldwise_init
struct Foo(ImplicitlyCopyable):
    var x: Int
    var y: String
    var z: Int

    def __init__(out self):
        self.x = 123
        self.y = "This is a string"
        self.z = 234


@always_inline
def func(a: Int, b: Foo) raises -> Foo:
    if a == 420:
        raise "some exception"  # breakpoint
    return b


def main() raises:
    print(func(420, Foo()).x)
