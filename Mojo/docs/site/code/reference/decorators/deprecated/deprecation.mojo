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


@deprecated(use=OtherStruct)
@fieldwise_init
struct MyStruct:
    var x: Int


@deprecated(use=Honkable)
trait Quackable:
    def quack(self):
        ...


trait Honkable:
    def honk(self):
        ...


struct OtherStruct:
    var x: Int
    var y: Int

    @deprecated("Ignore. Deprecation test.")
    def function(self):
        pass

    def __init__(out self, x: Int):
        self.x = x
        self.y = 0


@deprecated("Ignore. Deprecation test.")
comptime pi = 3.141592


def main():
    _ = MyStruct(x=5)

    OtherStruct(2).function()

    print(pi)

    # Demonstrate that only warnings are issued
    print("This is a functioning app")
