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

# Verify that conditional RegisterPassable correctly affects move behavior:
# - RP instantiations: register copies, no memoryOnly, no move init call
# - Memory-only instantiations: memoryOnly arg, custom move init called

# RUN: kgen -elaborate -S -o - %s | FileCheck %s --check-prefix=CHECK-IR
# RUN: %mojo -debug-level full %s | FileCheck %s --check-prefix=CHECK-EXEC


@fieldwise_init
struct CondRPMove[T: Movable & Deinitable](
    Deinitable,
    Movable,
    RegisterPassable where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, *, deinit move: Self):
        print("custom move called")
        self.value = move.value^


# RP case: arg/return have no memoryOnly, body is a register copy.
# CHECK-IR:      kgen.func @"{{.*}}do_move{{.*}}::SIMD[DType.int, 1]{{.*}}"(
# CHECK-IR-NOT:  memoryOnly
# CHECK-IR-SAME: ) ->
# CHECK-IR-NOT:  memoryOnly
# CHECK-IR-SAME: no_inline
# CHECK-IR-NEXT: kgen.return %arg0
@no_inline
def do_move(var x: CondRPMove[Int]) -> CondRPMove[Int]:
    return x^


# Memory-only case: arg has memoryOnly, body calls the custom move init.
# CHECK-IR:      kgen.func @"{{.*}}do_move{{.*}}String{{.*}}"(
# CHECK-IR-SAME: memoryOnly
# CHECK-IR:      kgen.call {{.*}}@"{{.*}}CondRPMove::__init__{{.*}}move:{{.*}}"
@no_inline
def do_move(var x: CondRPMove[String]) -> CondRPMove[String]:
    return x^


def main():
    # RP case: Int is RegisterPassable, move is a register copy.
    # CHECK-EXEC-LABEL: === rp move ===
    # CHECK-EXEC-NOT:   custom move called
    # CHECK-EXEC:       42
    print("=== rp move ===")
    var rp = do_move(CondRPMove[Int](42))
    print(rp.value)

    # Memory-only case: String is not RegisterPassable, custom move fires.
    # CHECK-EXEC-LABEL: === mem move ===
    # CHECK-EXEC:       custom move called
    # CHECK-EXEC:       hello
    print("=== mem move ===")
    var mem = do_move(CondRPMove[String](String("hello")))
    print(mem.value)
