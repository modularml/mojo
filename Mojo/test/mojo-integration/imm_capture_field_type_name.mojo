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


def main():
    var a: Int = 0
    var b: Int = 1

    def foo() {imm}:
        _ = a
        _ = b

    foo()

    # CHECK: ref[SIMD[DType.int, 1]]
    print(reflect[reflect[type_of(foo)].field_types()[0]].name())

    var xs = List[Int]()

    def by_imm() {imm xs}:
        _ = len(xs)

    def by_mut() {mut xs}:
        xs.append(3)

    var ys = List[Int]()

    def by_var() {var ys^}:
        _ = len(ys)

    by_imm()
    by_mut()
    by_var()

    # CHECK: list-imm: ref[List[SIMD[DType.int, 1]]]
    print(
        "list-imm:",
        reflect[reflect[type_of(by_imm)].field_types()[0]].name(),
    )
    # CHECK: list-mut: ref[List[SIMD[DType.int, 1]]]
    print(
        "list-mut:",
        reflect[reflect[type_of(by_mut)].field_types()[0]].name(),
    )
    # CHECK: list-var: List[SIMD[DType.int, 1]]
    print(
        "list-var:",
        reflect[reflect[type_of(by_var)].field_types()[0]].name(),
    )
