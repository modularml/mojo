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

# A function-scoped import is a declaration in the function scope: colliding
# with a local var or a parameter of the same name is a redefinition error, in
# either order. (Python instead silently rebinds.)

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# var first, import second.


def var_then_import():
    # expected-note @below {{previous definition here}}
    var shadowed_fn = 5
    # expected-error @below {{invalid redefinition of 'shadowed_fn'}}
    from wildcard_shadow_a import shadowed_fn

    _ = shadowed_fn


# // -----

# import first, var second.


def import_then_var():
    # expected-note @below {{previous definition here}}
    from wildcard_shadow_a import shadowed_fn

    # expected-error @below {{invalid redefinition of 'shadowed_fn'}}
    var shadowed_fn = 5
    _ = shadowed_fn


# // -----

# An argument of the same name also collides.


# expected-note @below {{previous definition here}}
def param_conflict(shadowed_fn: Int) -> Int:
    # expected-error @below {{invalid redefinition of 'shadowed_fn'}}
    from wildcard_shadow_a import shadowed_fn

    return 0
