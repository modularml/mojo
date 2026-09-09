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

# Test that a package's __init__.mojo can re-export its submodules using the
# Python-style relative self-import `from . import foo`.

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

# Re-exported submodules are reachable through the package.
from test_from_relative_self_import import foo
from test_from_relative_self_import import bar

# CHECK-LABEL: lit.fn @"main
def main():
    # `foo` re-exported via `from . import foo` resolves to the submodule.
    # CHECK: lit.call {{.*}}@test_from_relative_self_import::@foo::@"hello
    var a = foo.hello()

    # `bar` re-exports a sibling with the same relative self-import and uses it,
    # exercising the recursive-but-acyclic lazy import path (bar -> foo).
    # CHECK: lit.call {{.*}}@test_from_relative_self_import::@bar::@"hello
    var b = bar.hello()
