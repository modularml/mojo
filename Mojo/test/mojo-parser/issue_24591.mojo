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
# RUN: %parse-mojo-isolated %s | kgen-opt -mlir-print-op-generic | FileCheck %s

comptime NoneType = __mlir_type.`!kgen.none`


struct Optional[T: __mlir_type.`!kgen.type`](RegisterPassable):
    @implicit
    def __init__(out self, none: NoneType):
        pass


struct Param[x: Int](RegisterPassable):
    pass


# Check the TypeSignatureType attribute. This is the only memory-only
# struct so we can match with 0.
# CHECK: "lit.struct.decl"() {{.*}} convention = 0 :
# CHECK-SAME: signature = !lit.type_signature<"x": !Int, "y": !lit.struct<#Optional{{.*}}!lit.generator<<"y": !Int>() -> !lit.struct<#Param <:!Int *(1,0)>>
struct Thing[x: Int, y: Optional[def[y: Int]() thin -> Param[x]] = None](Movable where False):
    comptime z = 1
