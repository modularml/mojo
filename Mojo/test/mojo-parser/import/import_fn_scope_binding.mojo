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

# Which declaration a name resolves to in the presence of function-scoped
# imports, observed through the emitted call symbols:
#  - a function-scoped import shadows a module-level import of the same name
#    without affecting sibling functions;
#  - an import inside a comptime block shadows the function-scope binding and
#    is restored after the block;
#  - module-level imports are order-independent (a def above the import line
#    can use the name), unlike function scope (import_fn_scope_ordering.mojo);
#  - exact duplicate imports are de-duplicated;
#  - `import a.b` also makes symbols of the parent package `a` reachable, and
#    overlapping chain imports share the prefix.

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

from wildcard_shadow_a import shadowed_fn


# CHECK-LABEL: lit.fn @"module_binding_user()"
# CHECK: lit.call {{.*}}@wildcard_shadow_a::@"shadowed_fn
def module_binding_user() -> Int:
    return shadowed_fn()


# CHECK-LABEL: lit.fn @"fn_shadows_module()"
# CHECK: lit.call {{.*}}@wildcard_shadow_b::@"shadowed_fn
def fn_shadows_module() -> Int:
    from wildcard_shadow_b import shadowed_fn

    return shadowed_fn()


# The sibling function still sees the module-level binding after
# fn_shadows_module shadowed it.
# CHECK-LABEL: lit.fn @"sibling_unaffected()"
# CHECK: lit.call {{.*}}@wildcard_shadow_a::@"shadowed_fn
def sibling_unaffected() -> Int:
    return shadowed_fn()


# The comptime-if block import (b) is used inside the block; the function
# scope binding (a) is restored after it.
# CHECK-LABEL: lit.fn @"block_shadows_fn()"
# CHECK: lit.call {{.*}}@wildcard_shadow_b::@"shadowed_fn
# CHECK: lit.call {{.*}}@wildcard_shadow_a::@"shadowed_fn
def block_shadows_fn() -> Int:
    from wildcard_shadow_a import shadowed_fn

    comptime if True:
        from wildcard_shadow_b import shadowed_fn

        _ = shadowed_fn()
    return shadowed_fn()


# CHECK-LABEL: lit.fn @"dedup_duplicate_import()"
# CHECK: lit.call {{.*}}@wildcard_shadow_b::@"b_only_fn
def dedup_duplicate_import() -> Int:
    from wildcard_shadow_b import b_only_fn
    from wildcard_shadow_b import b_only_fn

    return b_only_fn()


# CHECK-LABEL: lit.fn @"chain_parent_symbol_access()"
# CHECK: lit.call {{.*}}@test_package::@__init__::@"method_defined_in_init
# CHECK: lit.call {{.*}}@test_package::@test_nested_package::@module::@"nested_function
def chain_parent_symbol_access():
    import test_package.test_nested_package

    test_package.method_defined_in_init()
    test_package.test_nested_package.nested_function()


# CHECK-LABEL: lit.fn @"chain_prefix_reuse()"
# CHECK: lit.call {{.*}}@test_package::@__init__::@"method_defined_in_init
# CHECK: lit.call {{.*}}@test_package::@test_nested_package::@module::@"nested_function
def chain_prefix_reuse():
    import test_package
    import test_package.test_nested_package

    test_package.method_defined_in_init()
    test_package.test_nested_package.nested_function()


# Module-level imports are order-independent: late_fn is imported at the
# bottom of this file.
# CHECK-LABEL: lit.fn @"order_independent_module_user()"
# CHECK: lit.call {{.*}}@wildcard_shadow_b::@"b_only_fn
def order_independent_module_user() -> Int:
    return late_fn()


from wildcard_shadow_b import b_only_fn as late_fn
