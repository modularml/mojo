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
#   Path:   tests/data/simple_cases/empty_lines.py
#
# ===----------------------------------------------------------------------=== #

"""Docstring."""


# leading comment
def f():
    NO = ""
    SPACE = " "
    DOUBLESPACE = "  "

    t = leaf.type
    p = leaf.parent  # trailing comment
    v = leaf.value

    if t in ALWAYS_NO_SPACE:
        pass
    if t == token.COMMENT:  # another trailing comment
        return DOUBLESPACE

    assert (
        p is not None
    ), f"INTERNAL ERROR: hand-made leaf without parent: {leaf!r}"

    prev = leaf.prev_sibling
    if not prev:
        prevp = preceding_leaf(p)
        if not prevp or prevp.type in OPENING_BRACKETS:

            return NO

        if prevp.type == token.EQUAL:
            if prevp.parent and prevp.parent.type in {
                syms.typedargslist,
                syms.varargslist,
                syms.parameters,
                syms.arglist,
                syms.argument,
            }:
                return NO

        elif prevp.type == token.DOUBLESTAR:
            if prevp.parent and prevp.parent.type in {
                syms.typedargslist,
                syms.varargslist,
                syms.parameters,
                syms.arglist,
                syms.dictsetmaker,
            }:
                return NO


###############################################################################
# SECTION BECAUSE SECTIONS
###############################################################################


def g():
    NO = ""
    SPACE = " "
    DOUBLESPACE = "  "

    t = leaf.type
    p = leaf.parent
    v = leaf.value

    # Comment because comments

    if t in ALWAYS_NO_SPACE:
        pass
    if t == token.COMMENT:
        return DOUBLESPACE

    # Another comment because more comments
    assert (
        p is not None
    ), f"INTERNAL ERROR: hand-made leaf without parent: {leaf!r}"

    prev = leaf.prev_sibling
    if not prev:
        prevp = preceding_leaf(p)

        if not prevp or prevp.type in OPENING_BRACKETS:
            # Start of the line or a bracketed expression.
            # More than one line for the comment.
            return NO

        if prevp.type == token.EQUAL:
            if prevp.parent and prevp.parent.type in {
                syms.typedargslist,
                syms.varargslist,
                syms.parameters,
                syms.arglist,
                syms.argument,
            }:
                return NO


# output


"""Docstring."""


# leading comment
def f():
    NO = ""
    SPACE = " "
    DOUBLESPACE = "  "

    t = leaf.type
    p = leaf.parent  # trailing comment
    v = leaf.value

    if t in ALWAYS_NO_SPACE:
        pass
    if t == token.COMMENT:  # another trailing comment
        return DOUBLESPACE

    assert (
        p is not None
    ), f"INTERNAL ERROR: hand-made leaf without parent: {leaf!r}"

    prev = leaf.prev_sibling
    if not prev:
        prevp = preceding_leaf(p)
        if not prevp or prevp.type in OPENING_BRACKETS:

            return NO

        if prevp.type == token.EQUAL:
            if prevp.parent and prevp.parent.type in {
                syms.typedargslist,
                syms.varargslist,
                syms.parameters,
                syms.arglist,
                syms.argument,
            }:
                return NO

        elif prevp.type == token.DOUBLESTAR:
            if prevp.parent and prevp.parent.type in {
                syms.typedargslist,
                syms.varargslist,
                syms.parameters,
                syms.arglist,
                syms.dictsetmaker,
            }:
                return NO


###############################################################################
# SECTION BECAUSE SECTIONS
###############################################################################


def g():
    NO = ""
    SPACE = " "
    DOUBLESPACE = "  "

    t = leaf.type
    p = leaf.parent
    v = leaf.value

    # Comment because comments

    if t in ALWAYS_NO_SPACE:
        pass
    if t == token.COMMENT:
        return DOUBLESPACE

    # Another comment because more comments
    assert (
        p is not None
    ), f"INTERNAL ERROR: hand-made leaf without parent: {leaf!r}"

    prev = leaf.prev_sibling
    if not prev:
        prevp = preceding_leaf(p)

        if not prevp or prevp.type in OPENING_BRACKETS:
            # Start of the line or a bracketed expression.
            # More than one line for the comment.
            return NO

        if prevp.type == token.EQUAL:
            if prevp.parent and prevp.parent.type in {
                syms.typedargslist,
                syms.varargslist,
                syms.parameters,
                syms.arglist,
                syms.argument,
            }:
                return NO
