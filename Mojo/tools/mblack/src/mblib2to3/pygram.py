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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   src/mblib2to3/pygram.py
#
# ===----------------------------------------------------------------------=== #

# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Export the Python grammar and symbols."""

# Python imports
import os
from typing import Union

# Local imports
from .pgen2 import driver
from .pgen2.grammar import Grammar

# Moved into initialize because mypyc can't handle __file__ (XXX bug)
# # The grammar file
# _GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), "Grammar.txt")
# _PATTERN_GRAMMAR_FILE = os.path.join(os.path.dirname(__file__),
#                                      "PatternGrammar.txt")


class Symbols:
    def __init__(self, grammar: Grammar) -> None:
        """Initializer.

        Creates an attribute for each grammar symbol (nonterminal),
        whose value is the symbol's type (an int >= 256).
        """
        for name, symbol in grammar.symbol2number.items():
            setattr(self, name, symbol)


class _python_symbols(Symbols):
    and_expr: int
    and_test: int
    annassign: int
    arglist: int
    argument: int
    arith_expr: int
    asexpr_test: int
    assert_stmt: int
    async_funcdef: int
    async_stmt: int
    atom: int
    augassign: int
    break_stmt: int
    capture_item: int
    case_block: int
    classdef: int
    comp_for: int
    comp_if: int
    comp_iter: int
    comp_op: int
    comparison: int
    compound_stmt: int
    comptime_assert_stmt_body: int
    comptime_stmt: int
    convention: int
    continue_stmt: int
    decorated: int
    decorator: int
    decorators: int
    del_stmt: int
    dictsetmaker: int
    dotted_as_name: int
    dotted_as_names: int
    dotted_name: int
    encoding_decl: int
    eval_input: int
    except_clause: int
    exec_stmt: int
    expr: int
    expr_stmt: int
    exprlist: int
    factor: int
    file_input: int
    flow_stmt: int
    for_stmt: int
    funcdef: int
    functype: int
    generatortype: int
    global_stmt: int
    guard: int
    if_stmt: int
    import_as_name: int
    import_as_names: int
    import_from: int
    import_name: int
    import_stmt: int
    lambdef: int
    listmaker: int
    match_stmt: int
    metaparams: int
    named_effect: int
    namedexpr_test: int
    not_test: int
    old_comp_for: int
    old_comp_if: int
    old_comp_iter: int
    old_lambdef: int
    old_test: int
    or_test: int
    parameters: int
    pass_stmt: int
    pattern: int
    patterns: int
    power: int
    print_stmt: int
    raise_stmt: int
    raises_type: int
    return_stmt: int
    result_type: int
    shift_expr: int
    simple_stmt: int
    single_input: int
    sliceop: int
    small_stmt: int
    subject_expr: int
    star_expr: int
    stmt: int
    subscript: int
    subscriptlist: int
    suite: int
    term: int
    test: int
    testlist: int
    testlist1: int
    testlist_gexp: int
    testlist_safe: int
    testlist_star_expr: int
    tfpdef: int
    tfplist: int
    tname: int
    tname_star: int
    trailer: int
    try_stmt: int
    typedargslist: int
    varargslist: int
    var_stmt: int
    vfpdef: int
    vfplist: int
    vname: int
    where_clause: int
    while_stmt: int
    with_stmt: int
    xor_expr: int
    yield_arg: int
    yield_expr: int
    yield_stmt: int


class _pattern_symbols(Symbols):
    Alternative: int
    Alternatives: int
    Details: int
    Matcher: int
    NegatedUnit: int
    Repeater: int
    Unit: int


python_grammar: Grammar
python_grammar_no_print_statement: Grammar
python_grammar_no_print_statement_no_exec_statement: Grammar
python_grammar_no_print_statement_no_exec_statement_async_keywords: Grammar
python_grammar_no_exec_statement: Grammar
pattern_grammar: Grammar
python_grammar_soft_keywords: Grammar
mojo_grammar: Grammar

python_symbols: _python_symbols
pattern_symbols: _pattern_symbols


def initialize(cache_dir: Union[str, "os.PathLike[str]", None] = None) -> None:
    global python_grammar
    global python_grammar_no_print_statement
    global python_grammar_no_print_statement_no_exec_statement
    global python_grammar_no_print_statement_no_exec_statement_async_keywords
    global python_grammar_soft_keywords
    global mojo_grammar
    global python_symbols
    global pattern_grammar
    global pattern_symbols

    # The grammar file
    _GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), "Grammar.txt")
    _PATTERN_GRAMMAR_FILE = os.path.join(
        os.path.dirname(__file__), "PatternGrammar.txt"
    )

    # Python 2
    python_grammar = driver.load_packaged_grammar(
        "mblib2to3", _GRAMMAR_FILE, cache_dir
    )
    python_grammar.version = (2, 0)

    soft_keywords = python_grammar.soft_keywords.copy()
    python_grammar.soft_keywords.clear()

    python_symbols = _python_symbols(python_grammar)

    # Python 2 + from __future__ import print_function
    python_grammar_no_print_statement = python_grammar.copy()
    del python_grammar_no_print_statement.keywords["print"]

    # Python 3.0-3.6
    python_grammar_no_print_statement_no_exec_statement = python_grammar.copy()
    del python_grammar_no_print_statement_no_exec_statement.keywords["print"]
    del python_grammar_no_print_statement_no_exec_statement.keywords["exec"]
    python_grammar_no_print_statement_no_exec_statement.version = (3, 0)

    # Python 3.7+
    python_grammar_no_print_statement_no_exec_statement_async_keywords = (
        python_grammar_no_print_statement_no_exec_statement.copy()
    )
    python_grammar_no_print_statement_no_exec_statement_async_keywords.async_keywords = True
    python_grammar_no_print_statement_no_exec_statement_async_keywords.version = (
        3,
        7,
    )

    # Python 3.10+
    python_grammar_soft_keywords = python_grammar_no_print_statement_no_exec_statement_async_keywords.copy()
    python_grammar_soft_keywords.soft_keywords = soft_keywords
    python_grammar_soft_keywords.version = (3, 10)

    # MOJO+
    mojo_grammar = python_grammar_soft_keywords.copy()
    mojo_grammar.mojo_keywords = True
    mojo_grammar.declaration_keywords = [
        "def",
        "__mlir_region",
        "struct",
        "trait",
    ]
    mojo_grammar.version = (0, 1)

    pattern_grammar = driver.load_packaged_grammar(
        "mblib2to3", _PATTERN_GRAMMAR_FILE, cache_dir
    )
    pattern_symbols = _pattern_symbols(pattern_grammar)
