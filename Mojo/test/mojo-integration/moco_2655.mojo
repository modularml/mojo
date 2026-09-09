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

# RUN: %mojo %s | FileCheck %s


trait ConfigTrait:
    @staticmethod
    def trait_method_0() -> Int:
        ...

    @staticmethod
    def trait_method_1(i: Int) -> Int:
        ...

    @staticmethod
    def trait_method_2[T: TrivialRegisterPassable](i: T) -> T:
        ...


struct Config(ConfigTrait):
    # All trait methods are trivial enough to be inlined.

    # Case 1: return a constant.
    @staticmethod
    def trait_method_0() -> Int:
        comptime src = 1
        return src

    # Case 2: return a argument.
    @staticmethod
    def trait_method_1(i: Int) -> Int:
        return i

    # Case 3: return a argument, but the type is parametric.
    @staticmethod
    def trait_method_2[T: TrivialRegisterPassable](i: T) -> T:
        return i

    def __init__(out self):
        pass


struct Attention[config_t: ConfigTrait](Writable):
    # Fields of dependent type, whose type can only be determined by
    # a indirect static call to a trait method.
    var dep_val_0: Array[Int, Self.config_t.trait_method_0()]
    var dep_val_1: Array[Int, Self.config_t.trait_method_1(2)]
    var dep_val_2: Array[Int, Self.config_t.trait_method_2[Int](3)]

    def __init__(out self):
        self.dep_val_0 = type_of(self.dep_val_0)(uninitialized=True)
        self.dep_val_1 = type_of(self.dep_val_1)(uninitialized=True)
        self.dep_val_2 = type_of(self.dep_val_2)(uninitialized=True)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("len_0: ", len(self.dep_val_0), "\n")
        writer.write("len_1: ", len(self.dep_val_1), "\n")
        writer.write("len_2: ", len(self.dep_val_2), "\n")


def main():
    var attention = Attention[Config]()
    # CHECK:      len_0: 1
    # CHECK-NEXT: len_1: 2
    # CHECK-NEXT: len_2: 3
    print(attention)
