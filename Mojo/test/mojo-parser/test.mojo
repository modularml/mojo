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

# ===----------------------------------------------------------------------=== #
#
# This tests that the builtin ImplicitlyCopyable and Movable traits are pulled from the
# Mojo Parser Tests stubs and not from the Mojo Stdlib package.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable = !lit.trait<
# CHECK-SAME: @std::@builtin::@stubs::@AnyType,
# CHECK-SAME: @std::@builtin::@stubs::@Copyable,
# CHECK-SAME: @std::@builtin::@stubs::@Deinitable,
# CHECK-SAME: @std::@builtin::@stubs::@ImplicitlyCopyable,
# CHECK-SAME: @std::@builtin::@stubs::@Movable>


# CHECK: lit.struct.decl @BoxedInt(!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable)
@fieldwise_init
struct BoxedInt(ImplicitlyCopyable):
    var value: Int
