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

# Test that a function re-exported from a package's __init__.mojo is preferred
# over the submodule of the same name during name resolution.
#
# When a package re-exports a function whose name collides with a submodule
# (e.g. `test_reexport_name_collision/foo.mojo` defines `def foo`, and
# `__init__.mojo` does `from .foo import foo`), importing via the package
# should resolve to the function, not the module.

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

# The direct import path works:
from test_reexport_name_collision.foo import foo as foo_direct
from test_reexport_name_collision.bar import bar as bar_direct
from test_reexport_name_collision.baz import baz as baz_direct

# The re-export path should also work:
from test_reexport_name_collision import foo
from test_reexport_name_collision import bar
from test_reexport_name_collision import baz

# Negative test: "qux" is a submodule but __init__ does NOT re-export
# anything named "qux", so this should import the module.
from test_reexport_name_collision import qux


# CHECK-LABEL: lit.fn @"main
def main():
    # Parametric function: direct import works fine.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@foo::@"foo
    var a = foo_direct[42]()

    # Parametric function: re-exported import should also resolve.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@foo::@"foo
    var b = foo[42]()

    # Non-parametric function: direct import works fine.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@bar::@"bar
    var c = bar_direct()

    # Non-parametric function: re-exported import should also resolve.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@bar::@"bar
    var d = bar()

    # Struct: direct import works fine.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@baz::@baz::@"__init__
    var e = baz_direct()

    # Struct: re-exported import should also resolve.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@baz::@baz::@"__init__
    var f = baz()

    # Negative test: "qux" imported as module, access its member function.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@qux::@"qux_func
    var g = qux.qux_func()
