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

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

# Bare `import pkg` does not expose submodules as attributes (gated ImportOp).
import visibility_pkg


def main():
    # expected-error @+1 {{package 'visibility_pkg' has no declaration 'sub'}}
    _ = visibility_pkg.sub


# // -----

# An explicitly imported submodule IS reachable (it is in the import tree).
import visibility_pkg.sub


def main():
    _ = visibility_pkg.sub


# // -----

# `import pkg.inner` gates inner's own submodules too (the `import std.package`
# case): the leaf package does not expose its submodules.
import visibility_pkg.inner


def main():
    _ = visibility_pkg.inner
    # expected-error @+1 {{package 'inner' has no declaration 'deep'}}
    _ = visibility_pkg.inner.deep


# // -----

# Explicitly importing the deep submodule makes it reachable.
import visibility_pkg.inner.deep


def main():
    _ = visibility_pkg.inner.deep


# // -----

# A re-exported submodule (`from . import shown` in __init__) IS reachable via
# attribute access after a bare import; a hidden sibling is still blocked.
import visibility_pkg


def main():
    _ = visibility_pkg.shown
    # expected-error @+1 {{package 'visibility_pkg' has no declaration 'sub'}}
    _ = visibility_pkg.sub

# // -----

# A submodule re-exported under an alias (`from . import sub as visible_sub` in
# __init__) is reachable through the alias, and access stays gated *through* it:
# the alias resolves the real module (`visibility_pkg.sub`), not the alias name,
# so a deeper call works.
#
# Keep this split free of any expected error. An aliased re-export reached in an
# otherwise error-free module regressed by crashing the referenced-decl
# resolution pass, which is skipped once any diagnostic is emitted — so an
# expected-error in the same split would mask the regression.
import visibility_pkg


def main():
    _ = visibility_pkg.visible_sub
    visibility_pkg.visible_sub.sub_fn()
