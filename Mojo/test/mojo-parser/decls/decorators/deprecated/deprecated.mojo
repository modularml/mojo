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

# Tests for @deprecated decorator IR generation (LIT tests).
# For warning emission tests, see deprecated.mojo.

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


# ===----------------------------------------------------------------------=== #
# Test: Basic @deprecated IR generation
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @DeprecatedStruct
# CHECK-SAME: deprecationInfo = #lit.deprecation<"struct">
@deprecated("struct")
struct DeprecatedStruct(Movable where False):
    pass


# CHECK-LABEL: lit.fn @"deprecated_func
# CHECK-SAME: deprecationInfo = #lit.deprecation<"func">
@deprecated("func")
def deprecated_func():
    pass


# CHECK-LABEL: lit.trait.decl @DeprecatedTrait
# CHECK-SAME: deprecationInfo = #lit.deprecation<"trait">
@deprecated("trait")
trait DeprecatedTrait:
    pass


# CHECK-LABEL: lit.alias.decl *"deprecated_alias
# CHECK-SAME: deprecationInfo = #lit.deprecation<"alias">
@deprecated("alias")
comptime deprecated_alias = 1


# ===----------------------------------------------------------------------=== #
# Test: @deprecated(use=...) syntax
# ===----------------------------------------------------------------------=== #


struct DeprecatedStructTarget(Movable where False):
    pass


# CHECK-LABEL: lit.struct.decl @DeprecatedStructUse
# CHECK-SAME: deprecationInfo = #lit.deprecation<"'DeprecatedStructUse' is deprecated, use 'DeprecatedStructTarget' instead", "DeprecatedStructTarget">
@deprecated(use=DeprecatedStructTarget)
struct DeprecatedStructUse(Movable where False):
    pass


def deprecated_func_target():
    pass


# CHECK-LABEL: lit.fn @"deprecated_func_use
# CHECK-SAME: deprecationInfo = #lit.deprecation<"'deprecated_func_use' is deprecated, use 'deprecated_func_target' instead", "deprecated_func_target">
@deprecated(use=deprecated_func_target)
def deprecated_func_use():
    pass


trait DeprecatedTraitTarget:
    pass


# CHECK-LABEL: lit.trait.decl @DeprecatedTraitUse
# CHECK-SAME: deprecationInfo = #lit.deprecation<"'DeprecatedTraitUse' is deprecated, use 'DeprecatedTraitTarget' instead", "DeprecatedTraitTarget">
@deprecated(use=DeprecatedTraitTarget)
trait DeprecatedTraitUse:
    pass


comptime deprecated_alias_target = 1


# CHECK-LABEL: lit.alias.decl *"deprecated_alias_use
# CHECK-SAME: deprecationInfo = #lit.deprecation<"'deprecated_alias_use' is deprecated, use 'deprecated_alias_target' instead", "deprecated_alias_target">
@deprecated(use=deprecated_alias_target)
comptime deprecated_alias_use = 1


# ===----------------------------------------------------------------------=== #
# Test: @deprecated(use=...) for methods
# ===----------------------------------------------------------------------=== #


struct MethodDeprecationTest(Movable where False):
    def replacement_method(self):
        pass

    # CHECK-LABEL: lit.fn @"deprecated_method_use
    # CHECK-SAME: deprecationInfo = #lit.deprecation<"'deprecated_method_use' is deprecated, use 'replacement_method' instead", "replacement_method">
    @deprecated(use=replacement_method)
    def deprecated_method_use(self):
        pass


struct StaticMethodDeprecationTest(Movable where False):
    @staticmethod
    def replacement_static():
        pass

    # CHECK-LABEL: lit.fn @"deprecated_static_use
    # CHECK-SAME: deprecationInfo = #lit.deprecation<"'deprecated_static_use' is deprecated, use 'replacement_static' instead", "replacement_static">
    @staticmethod
    @deprecated(use=replacement_static)
    def deprecated_static_use():
        pass
