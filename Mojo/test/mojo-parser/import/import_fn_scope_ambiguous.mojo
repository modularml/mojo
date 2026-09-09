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

# Function-scope variant of decls/conflicting_imports_errors.mojo: importing
# the same name from two different modules into one function scope. For
# functions this is a deprecation warning and the overload sets merge (the
# same-signature call below is then ambiguous); for values it is a hard
# ambiguity error at the second import statement.
#
# (Not -verify-diagnostics: the notes point into the shared input modules.)

# RUN: not %parse-mojo-isolated -split-input-file -I=%S/inputs %s 2>&1 | FileCheck %s


def merged_fn_overloads():
    from wildcard_shadow_a import shadowed_fn
    from wildcard_shadow_b import shadowed_fn

    _ = shadowed_fn()


# CHECK: error: import of 'shadowed_fn' is ambiguous
# CHECK: wildcard_shadow_a.mojo:{{[0-9]+}}:{{[0-9]+}}: note: 'shadowed_fn' declared here
# CHECK: wildcard_shadow_b.mojo:{{[0-9]+}}:{{[0-9]+}}: note: 'shadowed_fn' also declared here

# // -----


def ambiguous_value_import():
    from wildcard_shadow_a import shadowed_value
    # CHECK: error: import of 'shadowed_value' is ambiguous
    from wildcard_shadow_b import shadowed_value

    _ = shadowed_value


# CHECK: wildcard_shadow_a.mojo:{{[0-9]+}}:{{[0-9]+}}: note: 'shadowed_value' declared here
# CHECK: wildcard_shadow_b.mojo:{{[0-9]+}}:{{[0-9]+}}: note: 'shadowed_value' also declared here
