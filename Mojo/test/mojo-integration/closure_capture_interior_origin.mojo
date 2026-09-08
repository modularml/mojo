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

from std.memory import Pointer


def test_immutable_interior_origin():
    var list = List[Int]()
    list.append(42)
    var p = Pointer(to=list[0])

    def inner() {var}:
        _ = p == p
        print(p[])

    # CHECK: 42
    inner()


def test_mutable_interior_origin():
    var list = List[Int]()
    list.append(10)
    var p = Pointer(to=list[0])

    def inner() {var}:
        p[] = 99

    inner()
    # CHECK: 99
    print(list[0])


def test_parametric_base_mutability[
    is_mut: Bool,
    //,
    origin: Origin[mut=is_mut],
](ref[origin] list: List[Int]):
    var p = Pointer(to=list[0])

    def inner() {imm}:
        _ = p == p
        print(p[])

    inner()


def test_promoted_interior_origin():
    var list = List[Int]()
    list.append(21)
    ref r = list[0]

    def inner() {imm r}:
        print(r)

    # CHECK: 21
    inner()


def test_keep_base_origin():
    var list = List[Int]()
    list.append(5)
    var p = Pointer(to=list[0])

    def inner() {imm}:
        print(len(list))
        print(p[])

    # CHECK: 1
    # CHECK-NEXT: 5
    inner()


def test_two_interiors():
    var a = List[Int]()
    var b = List[Int]()
    a.append(1)
    b.append(2)
    var p = Pointer(to=a[0])
    var q = Pointer(to=b[0])

    def inner() {var}:
        print(p[])
        print(q[])

    # CHECK: 1
    # CHECK-NEXT: 2
    inner()


def test_derive_interior_without_interior_capture():
    var list = List[Int]()
    list.append(55)

    def inner() {imm}:
        print(list[0])

    # CHECK: 55
    inner()


def test_parametric_base_mutability_mutable():
    var list = List[Int]()
    list.append(7)
    # CHECK: 7
    test_parametric_base_mutability(list)


def test_parametric_base_mutability_immutable():
    var list = List[Int]()
    list.append(3)

    def take_imm(imm list: List[Int]):
        test_parametric_base_mutability(list)

    # CHECK: 3
    take_imm(list)


def main():
    test_immutable_interior_origin()
    test_mutable_interior_origin()
    test_promoted_interior_origin()
    test_keep_base_origin()
    test_two_interiors()
    test_derive_interior_without_interior_capture()
    test_parametric_base_mutability_mutable()
    test_parametric_base_mutability_immutable()
