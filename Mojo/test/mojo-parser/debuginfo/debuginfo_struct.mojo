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

# RUN: %parse-mojo-isolated -debug-level full -O0 -mlir-print-debuginfo %s | FileCheck %s

# CHECK: #[[SOURCENAME_INT:.*]] = #debuginfo.source_name<(struct)"SIMD"[#DType_name, #SIMDLength_name]<":{{.*}} {:dtype index}", ":{{.*}} {1}"> from {{.*}}>
# CHECK-DAG: #[[SOURCENAME_RP:.*]] = #debuginfo.source_name<(struct)"MyRP"[#[[SOURCENAME_INT]]] from <(module)"debuginfo_struct">>

# CHECK-DAG: #[[SOURCENAME_RP3:.*]] = #debuginfo.source_name<(struct)"MyRP"[#[[SOURCENAME_INT]]]<":{{.*}} 3}"> from <(module)"debuginfo_struct">>
# CHECK-DAG: #[[SOURCENAME_DATA:.*]] = #debuginfo.source_name<(struct)"MyData"[#[[SOURCENAME_INT]], #[[SOURCENAME_RP3]], <"{{.*}}@TrivialRegisterPassable>">] from <(module)"debuginfo_struct">>


# CHECK: lit.struct.decl @MyRP
# CHECK-SAME: sourceName = #[[SOURCENAME_RP]]
@fieldwise_init
struct MyRP[A: Int](TrivialRegisterPassable):
    var a: Int
    var b: Int

    @implicit
    def __init__(out self, b: Int):
        self.a = Self.A
        self.b = b


# CHECK: lit.struct.decl @MyData
# CHECK-SAME: sourceName = #[[SOURCENAME_DATA]]
struct MyData[A: Int, B: MyRP[3], C: TrivialRegisterPassable](Movable where False):
    var a: Int
    var b: MyRP[3]
    var c: Self.C

    @implicit
    def __init__(out self, c: Self.C):
        self.a = Self.A
        self.b = Self.B
        self.c = c


def entry():
    comptime rp = MyRP[3](4)
    var data = MyData[7, rp, MyRP[3]](rp)
