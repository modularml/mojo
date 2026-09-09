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

# Test that @__name sets the linkage name without exporting.


# CHECK: lit.fn @"my_func()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"custom_name" : !kgen.string, true>
# CHECK-NOT: export
@__name("custom_name")
def my_func():
    pass


# Test that @__name with a parametric expression produces a param.expr.

comptime a = 42


# CHECK: lit.fn @"parametric_name[::SIMD[DType.int, 1]]()"
# CHECK-SAME: linkageName = #kgen.linkage_name<#kgen.param.expr<data_to_str
@__name("prefix_" + String(T) + "_" + String(a))
def parametric_name[T: Int]():
    pass


# Test that the mangle option is stored in the IR as the boolean true.


# CHECK: lit.fn @"mangle_true()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_mangled" : !kgen.string, true>
@__name("my_mangled")
def mangle_true():
    pass


# Test that @__name combined with @export sets the linkage name and exports.
# Order doesn't matter: @__name sets the linkage name, @export marks exported.


# CHECK: lit.fn export @"name_then_export()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_export" : !kgen.string, true>
@__name("my_export")
@export
def name_then_export() abi("Mojo"):
    pass


# CHECK: lit.fn export @"name_then_c_export()"() cabi
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_c_export" : !kgen.string, true>
@export(ABI="C")
@__name("my_c_export")
def name_then_c_export():
    ...


# Same but with reversed order.


# CHECK: lit.fn export @"export_then_name()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_export2" : !kgen.string, true>
@export
@__name("my_export2")
def export_then_name() abi("Mojo"):
    pass


# CHECK: lit.fn export @"c_export_then_name()"() cabi
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_c_export2" : !kgen.string, true>
@export(ABI="C")
@__name("my_c_export2")
def c_export_then_name():
    ...
