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
# RUN: kgen -O0 -elaborate -S -o - %s | FileCheck %s


# CHECK-NOT: no_impl
# CHECK: kgen.func export @conditional_alias
def no_impl() -> __mlir_type.index:
    __mlir_op.`kgen.param.assert`[
        cond=__mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`,
        message=__mlir_attr.`"bad" : !kgen.string`,
    ]()
    return __mlir_attr.`0 : index`


def make_true() -> __mlir_type.`!kgen.scalar<bool>`:
    return __mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`


@export
def conditional_alias() abi("Mojo"):
    comptime value = __mlir_attr.`1 : index` if make_true() else no_impl()
