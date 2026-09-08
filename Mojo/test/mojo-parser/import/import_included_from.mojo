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

# Regression test: a diagnostic about a declaration reached through imports
# prints an "Included from" stack rooted at the user's import and walking
# through each re-exporting module/package down to the definition.
#
# `parametric_fn` is defined in test_nested_package/module.mojo; using it
# without binding its parameter is a parameter-inference error whose note points
# at that declaration, so it exercises the include stack independently of
# overload resolution.

# Note: precompiled packages intentionally drop intra-package import
# locations as they are internal details of the package.

# Note: MLIR diagnostics print "Included from", user-facing Mojo diagnostics
# print "Imported from". Test for both.

# RUN: not %parse-mojo-isolated -split-input-file --use-mlir-diagnostics=true -I=%S/inputs %s 2>&1 | FileCheck %s --check-prefixes CHECK,MLIRDIAG,MLIRDIAG-SOURCE
# RUN: not %parse-mojo-isolated -split-input-file --use-mlir-diagnostics=false -I=%S/inputs %s 2>&1 | FileCheck %s --check-prefixes CHECK,MOJODIAG,MOJODIAG-SOURCE

# RUN: rm -rf %t && mkdir -p %t/src
# RUN: cp -r %S/inputs/test_package %t/src
# RUN: cp -r %S/inputs/star_reexport_package %t/src

# RUN: mojo precompile -I%S/../../test-packages -o %t/test_package.mojoc %t/src/test_package
# RUN: mojo precompile -I%S/../../test-packages -o %t/star_reexport_package.mojoc %t/src/star_reexport_package
# RUN: not %parse-mojo-isolated -split-input-file --use-mlir-diagnostics=true -I=%t %s 2>&1 | FileCheck %s --check-prefixes CHECK,MLIRDIAG
# RUN: not %parse-mojo-isolated -split-input-file --use-mlir-diagnostics=false -I=%t %s 2>&1 | FileCheck %s --check-prefixes CHECK,MOJODIAG

# Now remove the sources and run with Mojo diagnostics. We should see the same
# as in the precompiled source-available case: that we see "Imported from" into
# the user's module but not inside the package.
# RUN: rm -r %t/src
# RUN: not %parse-mojo-isolated -split-input-file --use-mlir-diagnostics=false -I=%t %s 2>&1 | FileCheck %s --check-prefixes CHECK,MOJODIAG

# Reached through an intermediate module: test_package/module.mojo re-exports
# `parametric_fn` from test_nested_package/module.mojo. The stack flows
#   this file -> test_package/module.mojo (re-export) -> the defining module.
from test_package.module import parametric_fn

def main():
    _ = parametric_fn()

# CHECK: error: invalid call to 'parametric_fn': failed to infer parameter 'n'
# MLIRDIAG: Included from {{.*}}import_included_from.mojo
# MLIRDIAG-SOURCE-NEXT: Included from {{.*}}test_package{{.*}}module.mojo:{{[0-9]+}}:
# MOJODIAG: Imported from {{.*}}import_included_from.mojo
# MOJODIAG-SOURCE-NEXT: Imported from {{.*}}test_package{{.*}}module.mojo:{{[0-9]+}}:
# CHECK-NEXT: test_nested_package{{.*}}module.mojo:{{[0-9]+}}:{{[0-9]+}}: note: function declared here

# // -----

# Reached through a package __init__ re-export: test_nested_package/__init__.mojo
# re-exports `parametric_fn`. The stack is rooted at this file's import (not at
# whatever lazy lookup first opened __init__).
from test_package.test_nested_package import parametric_fn

def main():
    _ = parametric_fn()

# CHECK: error: invalid call to 'parametric_fn': failed to infer parameter 'n'
# MLIRDIAG: Included from {{.*}}import_included_from.mojo
# MLIRDIAG-SOURCE-NEXT: Included from {{.*}}test_nested_package{{.*}}__init__.mojo:{{[0-9]+}}:
# MOJODIAG: Imported from {{.*}}import_included_from.mojo
# MOJODIAG-SOURCE-NEXT: Imported from {{.*}}test_nested_package{{.*}}__init__.mojo:{{[0-9]+}}:
# CHECK-NEXT: test_nested_package{{.*}}module.mojo:{{[0-9]+}}:{{[0-9]+}}: note: function declared here

# // -----

# Reached through a package __init__ that re-exports a sub-package with `*`:
# star_reexport_package/__init__.mojo does `from .subpkg import *`, so the stack
# flows through *both* package __init__s down to the definition.
#   this file -> star_reexport_package/__init__.mojo (import *)
#             -> star_reexport_package/subpkg/__init__.mojo (re-export) -> leaf
from star_reexport_package import needs_param

def main():
    _ = needs_param()

# CHECK: error: invalid call to 'needs_param': failed to infer parameter 'n'
# MLIRDIAG: Included from {{.*}}import_included_from.mojo
# MLIRDIAG-SOURCE-NEXT: Included from {{.*}}star_reexport_package{{.*}}__init__.mojo:{{[0-9]+}}:
# MLIRDIAG-SOURCE-NEXT: Included from {{.*}}star_reexport_package{{.*}}subpkg{{.*}}__init__.mojo:{{[0-9]+}}:
# MOJODIAG: Imported from {{.*}}import_included_from.mojo
# MOJODIAG-SOURCE-NEXT: Imported from {{.*}}star_reexport_package{{.*}}__init__.mojo:{{[0-9]+}}:
# MOJODIAG-SOURCE-NEXT: Imported from {{.*}}star_reexport_package{{.*}}subpkg{{.*}}__init__.mojo:{{[0-9]+}}:
# CHECK-NEXT: star_reexport_package{{.*}}subpkg{{.*}}leaf.mojo:{{[0-9]+}}:{{[0-9]+}}: note: function declared here
