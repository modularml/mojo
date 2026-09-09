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

# Packages and modules whose names contain periods are importable via escaped
# identifiers. Each split imports through a different path shape and invokes a
# function through the imported name, so both resolution and member access are
# exercised.

# Import a dotted package and call a function from its __init__.
import `dotted.pkg`


def use_init() -> Int:
    return `dotted.pkg`.pkg_init_fn()


# // -----

# Import a plain submodule of a dotted package and call through the chain.
import `dotted.pkg`.module


def use_module() -> Int:
    return `dotted.pkg`.module.in_module()


# // -----

# Import a dotted submodule of a dotted package.
import `dotted.pkg`.`module.with.dots`


def use_dotted_module() -> Int:
    return `dotted.pkg`.`module.with.dots`.in_dotted_module()


# // -----

# Leaf-bind a fully dotted path with 'as'.
import `dotted.pkg`.`sub.pkg`.deep as d


def use_leaf_binding() -> Int:
    return d.in_deep()


# // -----

# from-import a dotted submodule out of a dotted package.
from `dotted.pkg` import `module.with.dots`


def use_from_import() -> Int:
    return `module.with.dots`.in_dotted_module()


# // -----

# from-import a symbol through a dotted package and dotted sub-package.
from `dotted.pkg`.`sub.pkg`.deep import in_deep


def use_symbol_import() -> Int:
    return in_deep()


# // -----

# from-import a module out of a dotted sub-package, with a rename.
from `dotted.pkg`.`sub.pkg` import deep as renamed


def use_renamed() -> Int:
    return renamed.in_deep()


# // -----

# Wildcard import from a dotted submodule.
from `dotted.pkg`.`module.with.dots` import *


def use_wildcard() -> Int:
    return in_dotted_module()


# // -----

# from-import a *function* whose name contains dots: the construct name is a
# plain decl name, not a module path, so the dots need no splitting.
from `dotted.pkg`.module import `fn.with.dots`


def use_dotted_fn() -> Int:
    return `fn.with.dots`()


# // -----

# ...and the same with a rename to a plain identifier.
from `dotted.pkg`.module import `fn.with.dots` as plain_fn


def use_renamed_dotted_fn() -> Int:
    return plain_fn()
