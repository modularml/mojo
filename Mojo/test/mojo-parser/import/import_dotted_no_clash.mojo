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

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# A module 'plain_mod' and a module 'plain_mod.dot' must never alias: each
# import form resolves to exactly its own file. The negative from-import in
# each split proves the *other* module's symbols are not reachable, i.e. the
# import did not accidentally resolve to (or merge with) the dotted twin.

import plain_mod


def use_plain() -> Int:
    return plain_mod.from_plain()


# expected-error @+1 {{module 'plain_mod' does not contain 'from_dotted'}}
from plain_mod import from_dotted

# // -----

import `plain_mod.dot`


def use_dotted() -> Int:
    return `plain_mod.dot`.from_dotted()


# expected-error @+1 {{module 'plain_mod.dot' does not contain 'from_plain'}}
from `plain_mod.dot` import from_plain

# // -----

# The same non-aliasing holds for siblings inside a package: 'module' vs
# 'module.with.dots'.
from `dotted.pkg` import module


def use_pkg_plain() -> Int:
    return module.in_module()


# expected-error @+1 {{module 'module' does not contain 'in_dotted_module'}}
from `dotted.pkg`.module import in_dotted_module
