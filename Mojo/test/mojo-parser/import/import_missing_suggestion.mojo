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

# Positive cases for the "did you mean to import" suggestion on an unknown
# declaration: a symbol that is part of a standard-library package's public
# surface but was not imported should produce a suggestion naming the *package*.
#
# The feature only runs against a prebuilt (bytecode) `std`, which is what we
# ship to users. So we precompile the tiny fixture std in
# inputs/import_suggestion_std to a `std.mojoc` and point the parser at it. With
# a bytecode std the whole package tree is materializable, so tests just
# reference a symbol directly — no need to pre-import anything.

# RUN: rm -rf %t && mkdir -p %t
# RUN: mojo precompile -o %t/std.mojoc %S/inputs/import_suggestion_std/std
# RUN: kgen-translate -import-mojo -mojo-enable-prebuilt-packages \
# RUN:   -mojo-search-paths=%t -split-input-file -verify-diagnostics %s

# NOTE: these tests deliberately use `def foo()` rather than `def main()`.
# Exporting `main` pulls in the startup-wrapper machinery (std.builtin._startup
# and a `Builtin.Startup` symbol), which the minimal fixture std does not
# provide. A non-main function avoids that entirely while still exercising the
# unknown-declaration path.


# A symbol re-exported by name in the package __init__ (`from .explicit_mod
# import explicit_reexport`) is attributed to the package.
def foo():
    # expected-error @+1 {{use of unknown declaration 'explicit_reexport'; did you mean to import it from 'std.import_suggestion'? Add 'from std.import_suggestion import explicit_reexport'}}
    _ = explicit_reexport

# // -----

# A symbol pulled into the package surface by a wildcard re-export
# (`from .wildcard_mod import *`) is attributed to the package.
def foo():
    # expected-error @+1 {{use of unknown declaration 'wildcard_reexport'; did you mean to import it from 'std.import_suggestion'? Add 'from std.import_suggestion import wildcard_reexport'}}
    _ = wildcard_reexport

# // -----

# A symbol declared directly in the package __init__.
def foo():
    # expected-error @+1 {{use of unknown declaration 'direct_decl_symbol'; did you mean to import it from 'std.import_suggestion'? Add 'from std.import_suggestion import direct_decl_symbol'}}
    _ = direct_decl_symbol

# // -----

# A symbol declared directly in a nested sub-package: the suggestion uses the
# full package path.
def foo():
    # expected-error @+1 {{use of unknown declaration 'nested_symbol'; did you mean to import it from 'std.import_suggestion.nested'? Add 'from std.import_suggestion.nested import nested_symbol'}}
    _ = nested_symbol

# // -----

# A symbol explicitly re-exported inside a nested sub-package's __init__
# (`from .detail import nested_reexport`): attributed to the nested package.
def foo():
    # expected-error @+1 {{use of unknown declaration 'nested_reexport'; did you mean to import it from 'std.import_suggestion.nested'? Add 'from std.import_suggestion.nested import nested_reexport'}}
    _ = nested_reexport

# // -----

# Ancestor/descendant chain: 'wildpkg_symbol' is declared in the sub-package
# std.import_suggestion.wild_pkg AND pulled into std.import_suggestion via
# `from .wild_pkg import *`, so two packages on one chain expose it. Rather than
# suppressing as ambiguous, the shortest (outermost, most public) path is
# suggested. This exercises both the sub-package wildcard branch and the
# chain-collapse-to-shortest logic (`uniqueImportPath`).
def foo():
    # expected-error @+1 {{use of unknown declaration 'wildpkg_symbol'; did you mean to import it from 'std.import_suggestion'? Add 'from std.import_suggestion import wildpkg_symbol'}}
    _ = wildpkg_symbol

# // -----

# Native owner preferred over a foreign re-exporter: 'native_owned_symbol' is
# declared in std.import_suggestion.owner and also re-exported by the sibling
# std.import_suggestion.aggregator via an absolute cross-package import (the way
# std.ffi re-exports std.os.abort). Two packages on diverging branches expose
# it, but only one owns it natively, so the suggestion prefers the native owner
# instead of suppressing as ambiguous.
def foo():
    # expected-error @+1 {{use of unknown declaration 'native_owned_symbol'; did you mean to import it from 'std.import_suggestion.owner'? Add 'from std.import_suggestion.owner import native_owned_symbol'}}
    _ = native_owned_symbol

# // -----

# Foreign-only fallback: 'foreign_only_symbol' is declared in the module
# std.import_suggestion.solo_mod, which no package re-exports from its own
# subtree; only std.import_suggestion.sole_reexporter surfaces it, via an
# absolute cross-package import. With no native owner, the sole foreign
# re-exporter is the fallback suggestion (mirrors `from std.ffi import X` when
# only ffi, not X's home package, exposes it).
def foo():
    # expected-error @+1 {{use of unknown declaration 'foreign_only_symbol'; did you mean to import it from 'std.import_suggestion.sole_reexporter'? Add 'from std.import_suggestion.sole_reexporter import foreign_only_symbol'}}
    _ = foreign_only_symbol
