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

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s


# COM: Synthesized constructors should not emit debuginfo.

# CHECK: lit.struct.decl @MyValueStruct(!AnyType_Copyable_Deinitable_Movable)
# CHECK-SAME: attributes {sourceName = #MyValueStruct_name}

# The only debug info comes from default trait method for copy.
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}linkageName = "copy($0)"

# We also have Moveinit in Movable and __deinit__ in Deinitable.
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}linkageName = "__init__(move:$0$)"
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}linkageName = "__deinit__($0$)"

# CHECK-NOT: #debuginfo.subprogram


@fieldwise_init
struct MyValueStruct(Copyable):
    var value: __mlir_type.index
