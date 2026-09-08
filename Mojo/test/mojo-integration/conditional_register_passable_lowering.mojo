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

# Verify that conditional RegisterPassable conformance correctly affects
# argument convention lowering at the KGEN IR level. We use `kgen -elaborate`
# to run the full pipeline (including lower-arg-conventions) and check:
#
# - RP instantiations: struct arg has NO memoryOnly marker
# - Memory-only instantiations: struct arg HAS memoryOnly marker

# RUN: kgen -elaborate -S -o - %s | FileCheck %s


struct ConditionalRP[T: Movable & Deinitable](
    Deinitable,
    Movable,
    RegisterPassable where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


struct TwoFieldRP[A: Movable & Deinitable, B: Movable & Deinitable](
    Deinitable,
    Movable,
    RegisterPassable where conforms_to(A, RegisterPassable) and conforms_to(
        B, RegisterPassable
    ),
):
    var a: Self.A
    var b: Self.B

    def __init__(out self, var a: Self.A, var b: Self.B):
        self.a = a^
        self.b = b^


# --- Register-passable instantiations ---
# The lowered arg type should NOT have memoryOnly.


# CHECK:     kgen.func @"{{.*}}take_rp_int{{.*}}"(
# CHECK-NOT: memoryOnly
# CHECK-SAME: ) no_inline
@no_inline
def take_rp_int(x: ConditionalRP[Int]):
    print(x.value)


# Nested: ConditionalRP[ConditionalRP[Int]] should also be RP.
# CHECK:     kgen.func @"{{.*}}take_rp_nested{{.*}}"(
# CHECK-NOT: memoryOnly
# CHECK-SAME: ) no_inline
@no_inline
def take_rp_nested(x: ConditionalRP[ConditionalRP[Int]]):
    print(x.value.value)


# Two-field struct where both fields are RP.
# CHECK:     kgen.func @"{{.*}}take_two_field_rp{{.*}}"(
# CHECK-NOT: memoryOnly
# CHECK-SAME: ) no_inline
@no_inline
def take_two_field_rp(x: TwoFieldRP[Int, Bool]):
    print(x.a)


# --- Memory-only instantiations ---
# The lowered arg type SHOULD have memoryOnly.


# CHECK: kgen.func @"{{.*}}take_mem_string{{.*}}"({{.*}}memoryOnly
@no_inline
def take_mem_string(x: ConditionalRP[String]):
    print(x.value)


# Two-field struct where one field is not RP.
# CHECK: kgen.func @"{{.*}}take_two_field_mem{{.*}}"({{.*}}memoryOnly
@no_inline
def take_two_field_mem(x: TwoFieldRP[Int, String]):
    print(x.a)


# Nested negative: ConditionalRP[ConditionalRP[String]] — inner type is
# memory-only, so the outer must also be memory-only. Requires constraint
# evaluation on the inner ConformanceOp, not just existence checking.
# CHECK: kgen.func @"{{.*}}take_mem_nested{{.*}}"({{.*}}memoryOnly
@no_inline
def take_mem_nested(x: ConditionalRP[ConditionalRP[String]]):
    print(x.value.value)


def main():
    take_rp_int(ConditionalRP[Int](42))
    take_rp_nested(ConditionalRP[ConditionalRP[Int]](ConditionalRP[Int](99)))
    take_two_field_rp(TwoFieldRP[Int, Bool](1, True))
    take_mem_string(ConditionalRP[String](String("hello")))
    take_two_field_mem(TwoFieldRP[Int, String](0, String("world")))
    take_mem_nested(
        ConditionalRP[ConditionalRP[String]](
            ConditionalRP[String](String("nested"))
        )
    )
