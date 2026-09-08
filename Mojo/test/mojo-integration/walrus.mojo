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


struct Counter(Movable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __getitem__(self, i: Int) -> Int:
        return self.value

    def __setitem__(mut self, i: Int, new_value: Int):
        self.value = new_value

    def overwrite(mut self) -> Int:
        self.value = 99
        return 0


@fieldwise_init
struct Cell(ImplicitlyCopyable, Movable):
    var value: Int


struct NestedField(Movable):
    var cell: Cell

    def __init__(out self):
        self.cell = Cell(0)

    def __getitem__(self, i: Int) -> Cell:
        return self.cell

    def __setitem__(mut self, i: Int, new_value: Cell):
        self.cell = new_value

    def overwrite(mut self) -> Int:
        self.cell = Cell(99)
        return 0


struct RefSubscript(Movable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __getitem__(ref self, i: Int) -> ref[self.value] Int:
        return self.value

    def overwrite(mut self) -> Int:
        self.value = 99
        return 0


struct DynamicAttr(Movable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __getattr__(self, name: StringLiteral) -> Int:
        return self.value

    def __setattr__(mut self, name: StringLiteral, value: Int):
        self.value = value

    def overwrite(mut self) -> Int:
        self.value = 99
        return 0


def clobber(mut n: Int) -> Int:
    n = 99
    return 0


def main():
    # Overwriting the target after the store must not change what the walrus
    # returned. A target is an MLValue, an RLValue or a DLValue.

    # An MLValue target is a storable address.
    var field = Counter()
    # CHECK: 1 0
    print(field.value := 1, field.overwrite())

    var name: Int
    # CHECK-NEXT: 1 0
    print(name := 1, clobber(name))

    # A subscript with no `__setitem__` is a memory location, not a
    # getter/setter pair.
    var reference = RefSubscript()
    # CHECK-NEXT: 1 0
    print(reference[0] := 1, reference.overwrite())

    # An RLValue and `_` have no case: both need an initializer type, which a
    # walrus target lacks.

    # A DLValue has a __setitem__ or __setattr__ method, or is a tuple.
    var subscript = Counter()
    # CHECK-NEXT: 1 0
    print(subscript[0] := 1, subscript.overwrite())

    var nested = NestedField()
    # CHECK-NEXT: 1 0
    print(nested[0].value := 1, nested.overwrite())

    var attribute = DynamicAttr()
    # CHECK-NEXT: 1 0
    print(attribute.computed := 1, attribute.overwrite())

    var a: Int
    var b: Int
    # CHECK-NEXT: (1, 2) 0 0
    print((a, b) := (1, 2), clobber(a), clobber(b))
