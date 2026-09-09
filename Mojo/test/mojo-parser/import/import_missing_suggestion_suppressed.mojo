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

# Cases where the unknown-declaration diagnostic must NOT gain an import
# suggestion. We use FileCheck with an end-of-line anchor ({{$}}) to assert that
# nothing follows the bare message: -verify-diagnostics matches substrings and
# so cannot express the *absence* of the "; did you mean ..." suffix.
#
# See import_missing_suggestion.mojo for the fixture-std / bytecode setup and
# the reason these use `def foo()` instead of `def main()`.

# RUN: rm -rf %t && mkdir -p %t
# RUN: mojo precompile -o %t/std.mojoc %S/inputs/import_suggestion_std/std
# RUN: not kgen-translate -import-mojo -mojo-enable-prebuilt-packages \
# RUN:   -mojo-search-paths=%t -split-input-file %s 2>&1 | FileCheck %s


# Ambiguous: 'ambiguous_symbol' is re-exported by both
# std.import_suggestion.pkg_a and std.import_suggestion.pkg_b. These are sibling
# packages on diverging branches (neither is an ancestor of the other), so there
# is no single canonical import to suggest and it stays suppressed. Contrast the
# ancestor->descendant chain case (see import_missing_suggestion.mojo), which is
# collapsed to its shortest path instead.
def foo():
    # CHECK: use of unknown declaration 'ambiguous_symbol'{{$}}
    _ = ambiguous_symbol

# // -----

# Defined in a submodule but not re-exported by any __init__: reachable only via
# its internal submodule path, so it is not public API and is not suggested.
def foo():
    # CHECK: use of unknown declaration 'not_reexported_symbol'{{$}}
    _ = not_reexported_symbol

# // -----

# A private (underscore-prefixed) name, even though re-exported, is never
# suggested.
def foo():
    # CHECK: use of unknown declaration '_private_reexport'{{$}}
    _ = _private_reexport

# // -----

# A name that does not exist anywhere in std: no suggestion.
def foo():
    # CHECK: use of unknown declaration 'definitely_not_a_real_symbol'{{$}}
    _ = definitely_not_a_real_symbol

# // -----

# LIMITATION (tracked): a *multi-component* relative wildcard.
# std.import_suggestion's __init__ does `from .deep.inner import *`. Only
# single-component relative wildcards (`from .X import *`) are resolved, so this
# genuinely re-exported symbol is not found and gets no suggestion.
def foo():
    # CHECK: use of unknown declaration 'multicomp_wildcard_symbol'{{$}}
    _ = multicomp_wildcard_symbol

# // -----

# LIMITATION (tracked): an *absolute* wildcard re-export.
# std.import_suggestion's __init__ does `from std.import_suggestion.abs_wild_mod
# import *`. Only single-component *relative* wildcards are resolved (an absolute
# path has no leading dot), so this symbol is not found and gets no suggestion.
def foo():
    # CHECK: use of unknown declaration 'abs_wildcard_symbol'{{$}}
    _ = abs_wildcard_symbol

# // -----

# Leaf-collision guard: std.import_suggestion has a direct child module `inner`
# (declaring 'collision_symbol') whose leaf name matches the final component of
# the __init__'s `from .deep.inner import *`. Resolving that multi-component
# wildcard by leaf name alone would false-match this unrelated direct child and
# wrongly suggest `std.import_suggestion`. Multi-component wildcards are skipped
# up front, so 'collision_symbol' (not re-exported anywhere) gets no suggestion.
def foo():
    # CHECK: use of unknown declaration 'collision_symbol'{{$}}
    _ = collision_symbol

# // -----

# LIMITATION (tracked): a wildcard whose module itself wildcard-re-exports.
# std.import_suggestion does `from .wild_chain import *`, and wild_chain does
# `from .chained import *`. The walk is one level deep and does not follow the
# nested wildcard, so this symbol is not found and gets no suggestion.
def foo():
    # CHECK: use of unknown declaration 'transitive_symbol'{{$}}
    _ = transitive_symbol
