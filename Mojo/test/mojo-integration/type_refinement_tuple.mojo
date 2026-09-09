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

# Type refinement for tuple unpacking patterns in for loops.
#
# RUN: %mojo -debug-level full %s | FileCheck %s


trait Greetable:
    def greet(self) -> String:
        ...


struct Dog(
    Copyable,
    Deinitable,
    Greetable,
    ImplicitlyCopyable,
    Movable,
):
    var name: String

    def __init__(out self, name: String):
        self.name = name

    def greet(self) -> String:
        return "Woof! I'm " + self.name


# Value unpack from zip: each iteration binds two copied element values.
def test_for_bind[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for left, right in zip(items1, items2):
        print("bind:", left.greet(), right.greet())


# Ref unpack from zip: tuple pair exposed as two refs into list elements.
def test_for_ref[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for ref left, right in zip(items1, items2):
        print("ref:", left.greet(), right.greet())


# Var unpack from zip: per-iteration `var` slots (not refs into the lists).
def test_for_var_each[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for var left, right in zip(items1, items2):
        print("var:", left.greet(), right.greet())


# Whole-tuple `var (left, right)` unpack from each zip-produced pair.
def test_for_var_whole[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for var (left, right) in zip(items1, items2):
        print("var_tuple:", left.greet(), right.greet())


# `var` unpack then reassign one side after first use; store-time refinement
# must track the loop-carried `var` binding across the assignment.
def test_for_var_reassign[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for (var left), (var right) in zip(items1, items2):
        var before = left.greet()
        left = right.copy()
        print("var_reassign:", before, left.greet(), right.greet())


# Single binding for the zip pair; read tuple fields via subscript (no unpack).
def test_implicit_subscript[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for pair in zip(items1, items2):
        print("subscript:", pair[0].greet(), pair[1].greet())


# NOTE: The `enumerate` refinement cases moved to
# `type_refinement_tuple_enumerate_xfail.mojo`. They no longer compile now that
# `Tuple` is conditionally `Deinitable`: `enumerate` yields
# `Tuple[Int, Iterator.Element]`, and `Iterator.Element: Movable` erases the
# element's `Deinitable` refinement in a generic context. See
# MOCO-4361.


# `ref` to each zip pair; subscript reads the two tuple elements.
def test_ref_subscript[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for ref pair in zip(items1, items2):
        print("ref_sub:", pair[0].greet(), pair[1].greet())


# `var` copy of each enumerate pair; subscript reads on the copy.
def test_var_subscript[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for var pair in zip(items1, items2):
        print("var_sub:", pair[0].greet(), pair[1].greet())


# Single-column `for`: default element binding is an immutable ref.
def test_for_single_ref[
    T: Copyable
](items: List[T]) where conforms_to(T, Greetable):
    for item in items:
        print("single_ref:", item.greet())


# Single-column `for` with explicit `var` element binding.
def test_for_single_var[
    T: ImplicitlyCopyable & Deinitable
](items: List[T]) where conforms_to(T, Greetable):
    for var item in items:
        print("single_var:", item.greet())


# Sequential loops reuse `left`/`right` names — refinement must attach to the
# right VarDecl per loop, not by name alone. (Value unpack variant.)
def test_shadow_bind[
    T: ImplicitlyCopyable & Deinitable
](a1: List[T], a2: List[T], b1: List[T], b2: List[T]) where conforms_to(
    T, Greetable
):
    for left, right in zip(a1, a2):
        print("bind_first:", left.greet(), right.greet())
    for left, right in zip(b1, b2):
        print("bind_second:", left.greet(), right.greet())


# Same shadowing pattern with `ref left, ref right` unpacks.
def test_shadow_ref[
    T: ImplicitlyCopyable & Deinitable
](a1: List[T], a2: List[T], b1: List[T], b2: List[T]) where conforms_to(
    T, Greetable
):
    for ref left, right in zip(a1, a2):
        print("ref_first:", left.greet(), right.greet())
    for ref left, right in zip(b1, b2):
        print("ref_second:", left.greet(), right.greet())


# Callee takes `ref x: T`; forces refinement to flow through an extra ref edge.
def greet_through_ref_param[T: Greetable](ref x: T) -> String:
    return x.greet()


# Unpacked `ref` tuple elements passed into `ref` parameters (not only direct
# `.greet()` on the binding).
def test_ref_pass_through_unpack[
    T: ImplicitlyCopyable & Deinitable
](items1: List[T], items2: List[T]) where conforms_to(T, Greetable):
    for ref left, right in zip(items1, items2):
        print(
            "ref_pass:",
            greet_through_ref_param(left),
            greet_through_ref_param(right),
        )


# Store into a tuple held in a list: `for ref row` + `row[i]=` (tuple
# subscript preserves mutability; unpacked `ref (a,b)` does not allow this
# whole-value assign today). Checks post-loop list contents too.
def test_mutable_tuple_ref_unpack(imm left_dog: Dog, imm right_dog: Dog):
    var pairs = List[Tuple[Dog, Dog]]()
    pairs.append((left_dog.copy(), right_dog.copy()))
    for ref row in pairs:
        var before = greet_through_ref_param(row[0])
        row[0] = row[1].copy()
        print(
            "mut_ref_unpack:",
            before,
            greet_through_ref_param(row[0]),
            greet_through_ref_param(row[1]),
        )
    print("mut_ref_after:", pairs[0][0].greet(), pairs[0][1].greet())


def main():
    var dogs1 = List[Dog]()
    dogs1.append(Dog("Buddy"))
    dogs1.append(Dog("Max"))

    var dogs2 = List[Dog]()
    dogs2.append(Dog("Charlie"))
    dogs2.append(Dog("Luna"))

    # CHECK: bind: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: bind: Woof! I'm Max Woof! I'm Luna
    test_for_bind(dogs1, dogs2)

    # CHECK: ref: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: ref: Woof! I'm Max Woof! I'm Luna
    test_for_ref(dogs1, dogs2)

    # CHECK: var: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: var: Woof! I'm Max Woof! I'm Luna
    test_for_var_each(dogs1, dogs2)

    # CHECK: var_tuple: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: var_tuple: Woof! I'm Max Woof! I'm Luna
    test_for_var_whole(dogs1, dogs2)

    # CHECK: var_reassign: Woof! I'm Buddy Woof! I'm Charlie Woof! I'm Charlie
    # CHECK: var_reassign: Woof! I'm Max Woof! I'm Luna Woof! I'm Luna
    test_for_var_reassign(dogs1, dogs2)

    # CHECK: subscript: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: subscript: Woof! I'm Max Woof! I'm Luna
    test_implicit_subscript(dogs1, dogs2)

    # CHECK: ref_sub: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: ref_sub: Woof! I'm Max Woof! I'm Luna
    test_ref_subscript(dogs1, dogs2)

    # CHECK: var_sub: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: var_sub: Woof! I'm Max Woof! I'm Luna
    test_var_subscript(dogs1, dogs2)

    var dogs3 = List[Dog]()
    dogs3.append(Dog("Rex"))
    dogs3.append(Dog("Bella"))

    # CHECK: single_ref: Woof! I'm Rex
    # CHECK: single_ref: Woof! I'm Bella
    test_for_single_ref(dogs3)

    # CHECK: single_var: Woof! I'm Rex
    # CHECK: single_var: Woof! I'm Bella
    test_for_single_var(dogs3)

    var a1 = List[Dog]()
    a1.append(Dog("Alpha"))
    var a2 = List[Dog]()
    a2.append(Dog("Bravo"))
    var b1 = List[Dog]()
    b1.append(Dog("Delta"))
    var b2 = List[Dog]()
    b2.append(Dog("Echo"))

    # CHECK: bind_first: Woof! I'm Alpha Woof! I'm Bravo
    # CHECK: bind_second: Woof! I'm Delta Woof! I'm Echo
    test_shadow_bind(a1, a2, b1, b2)

    # CHECK: ref_first: Woof! I'm Alpha Woof! I'm Bravo
    # CHECK: ref_second: Woof! I'm Delta Woof! I'm Echo
    test_shadow_ref(a1, a2, b1, b2)

    # CHECK: ref_pass: Woof! I'm Buddy Woof! I'm Charlie
    # CHECK: ref_pass: Woof! I'm Max Woof! I'm Luna
    test_ref_pass_through_unpack(dogs1, dogs2)

    # CHECK: mut_ref_unpack: Woof! I'm Milo Woof! I'm Otis Woof! I'm Otis
    # CHECK: mut_ref_after: Woof! I'm Otis Woof! I'm Otis
    test_mutable_tuple_ref_unpack(Dog("Milo"), Dog("Otis"))
