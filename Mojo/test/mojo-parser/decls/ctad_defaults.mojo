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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Verify kw-only defaults are searched during binding verification.


struct MyUnsafePointer[
    type: AnyType,
    x: Int = 3,
    *,
    origin: Origin[mut=True],
    address_space: AddressSpace = .GENERIC,
    exclusive: Bool = False,
    alignment: Int = 1,
](Movable where False):
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    var address: Self._mlir_type

    @always_inline
    @implicit
    def __init__(out self, value: Self._mlir_type):
        self.address = value


# CHECK-LABEL: lit.fn @"unsafe_ptr
def unsafe_ptr(s: __mlir_type.`!kgen.string`):
    # CHECK:      lit.call {{.*}}::@MyUnsafePointer::@"__init__{{.*}}"[mut *"{{.*}}"]
    # CHECK-SAME: :!AnyType #type_value,
    # CHECK-SAME: :!Int {:scalar<index> 3},
    # CHECK-SAME: :!AddressSpace {_value: !SIMDLength = {0}},
    # CHECK-SAME: :!Bool {:scalar<bool> false},
    # CHECK-SAME: :!Int {:scalar<index> 1}>>
    var ptr = MyUnsafePointer[origin=AnyOrigin[mut=True]](
        __mlir_op.`pop.string.address`(s)
    )
