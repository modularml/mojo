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

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s


struct ConditionalArraySize[T: AnyType]:
    comptime Size: Int = 0 if conforms_to(Self.T, ImplicitlyCopyable) else 1
    var array: Array[Int, Self.Size]

    def __init__(out self):
        self.array = {fill = 0}


def generic(conditional: ConditionalArraySize) raises:
    print(conditional.array.length)


struct ConditionalValueType[T: AnyType]:
    comptime Type = List[Int] if conforms_to(
        Self.T, ImplicitlyCopyable
    ) else String

    var value: Self.Type

    def __init__(out self):
        self.value = {}


def main() raises:
    comptime IsImplicitlyCopyable: AnyType = Int
    comptime IsNotImplicitlyCopyable: AnyType = List[Int]

    var size_0 = ConditionalArraySize[IsImplicitlyCopyable]()
    # CHECK: 0
    generic(size_0)
    # CHECK: 0
    print(size_0.array.length)

    var size_1 = ConditionalArraySize[IsNotImplicitlyCopyable]()
    # CHECK: 1
    generic(size_1)
    # CHECK: 1
    print(size_1.array.length)

    var _list_int = ConditionalValueType[IsImplicitlyCopyable]()
    # CHECK: True
    print(List[Int] == type_of(_list_int.value))

    var _string = ConditionalValueType[IsNotImplicitlyCopyable]()
    # CHECK: True
    print(String == type_of(_string.value))
