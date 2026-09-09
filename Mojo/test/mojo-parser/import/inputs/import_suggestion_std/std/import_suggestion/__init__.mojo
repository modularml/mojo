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
"""Fixture package exercising the import-suggestion walk's matching rules.

Each public name below is part of this package's surface in a different way, so
the tests can assert the suggestion attributes them to `std.import_suggestion`:
"""

# A symbol re-exported from a submodule by name (explicit re-export).
from .explicit_mod import explicit_reexport

# A private re-export: underscore-prefixed names must never be suggested.
from .explicit_mod import _private_reexport

# A wildcard re-export of a submodule: every public symbol of wildcard_mod
# joins this surface.
from .wildcard_mod import *

# A wildcard re-export of a sub-package: its public surface (its __init__'s
# decls) joins this surface.
from .wild_pkg import *

# Unsupported wildcard forms, kept here to anchor the negative tests. Only
# single-component relative wildcards (`from .X import *`) are resolved:
#  - multi-component relative module: not single-component, so
#    `multicomp_wildcard_symbol` is not found.
from .deep.inner import *
#  - absolute module path (no leading dot): not relative, so
#    `abs_wildcard_symbol` is not found.
from std.import_suggestion.abs_wild_mod import *
#  - a wildcard whose module itself wildcard-re-exports: not followed (the walk
#    is one level deep), so `transitive_symbol` is not found.
from .wild_chain import *


# A symbol declared directly in the package __init__.
def direct_decl_symbol():
    pass
