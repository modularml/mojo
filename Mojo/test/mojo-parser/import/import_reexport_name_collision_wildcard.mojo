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

# Test that wildcard imports (`from pkg import *`) correctly re-export
# functions from __init__.mojo even when a submodule has the same name.
#
# This is the wildcard-import counterpart of
# import_reexport_name_collision.mojo (which tests explicit imports).

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

# Wildcard import should bring in the re-exported functions, not the modules.
from test_reexport_name_collision import *


# CHECK-LABEL: lit.fn @"main
def main():
    # Parametric function via wildcard import.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@foo::@"foo
    var a = foo[42]()

    # Non-parametric function via wildcard import.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@bar::@"bar
    var b = bar()

    # Struct via wildcard import.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@baz::@baz::@"__init__
    var c = baz()

    # Negative test: "qux" has no same-name re-export, so it should still
    # be accessible as a module via wildcard import.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@qux::@"qux_func
    var d = qux.qux_func()
