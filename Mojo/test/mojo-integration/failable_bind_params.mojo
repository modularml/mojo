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
# COM: Verify that if a generator that contains a bind parameters expression
#      needs to await the concretization of a another function it is properly
#      skipped and rescheduled.

# RUN: %mojo -debug-level full %s | FileCheck %s

from std.builtin.variadics import TypeList
from std.sys.info import _current_target


@fieldwise_init
struct Box[w: Int](Movable):
    var p: Pointer[Int, MutUntrackedOrigin]


trait HasMember(Movable):
    comptime m[p: Int]: Optional[Box[p]] = None


struct EntryA(HasMember):
    pass


struct EntryB(HasMember):
    pass


def _deferred_index[
    target: __mlir_type.`!kgen.target` = _current_target()
]() -> __mlir_type.index:
    comptime sbw = __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        target,
        `, "simd_bit_width" : !kgen.string`,
        `> : index`,
    ]
    return __mlir_op.`index.remu`(
        sbw, __mlir_op.`index.constant`[value=__mlir_attr.`2:index`]()
    )


def main():
    # COM: Create a parametric expression that cannot fold because of dependency on
    #      unconcretized function in a parameter bind operation.
    comptime b = TypeList.of[Trait=HasMember, EntryA, EntryB].__getitem_param__[
        SIMDLength(mlir_value=_deferred_index())
    ].m[0].__bool__()
    # CHECK: False
    print(b)
