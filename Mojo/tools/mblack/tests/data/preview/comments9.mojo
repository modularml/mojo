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
#   Path:   tests/data/preview/comments9.py
#
# ===----------------------------------------------------------------------=== #

# Test for https://github.com/psf/black/issues/246.

some = statement
# This comment should be split from the statement above by two lines.
def function():
    pass


some = statement
# This multiline comments section
# should be split from the statement
# above by two lines.
def function():
    pass


some = statement
# This comment should be split from the statement above by two lines.
async def async_function():
    pass


some = statement
# This comment should be split from the statement above by two lines.
class MyClass:
    pass


some = statement
# This should be stick to the statement above

# This should be split from the above by two lines
class MyClassWithComplexLeadingComments:
    pass


class ClassWithDocstring:
    """A docstring."""


# Leading comment after a class with just a docstring
class MyClassAfterAnotherClassWithDocstring:
    pass


some = statement
# leading 1
@deco1
# leading 2
# leading 2 extra
@deco2(with_args=True)
# leading 3
@deco3
# leading 4
def decorated():
    pass


some = statement
# leading 1
@deco1
# leading 2
@deco2(with_args=True)

# leading 3 that already has an empty line
@deco3
# leading 4
def decorated_with_split_leading_comments():
    pass


some = statement
# leading 1
@deco1
# leading 2
@deco2(with_args=True)
# leading 3
@deco3

# leading 4 that already has an empty line
def decorated_with_split_leading_comments():
    pass


def main():
    if a:
        # Leading comment before inline function
        def inline():
            pass

        # Another leading comment
        def another_inline():
            pass

    else:
        # More leading comments
        def inline_after_else():
            pass


if a:
    # Leading comment before "top-level inline" function
    def top_level_quote_inline():
        pass

    # Another leading comment
    def another_top_level_quote_inline_inline():
        pass

else:
    # More leading comments
    def top_level_quote_inline_after_else():
        pass


class MyClass:
    # First method has no empty lines between bare class def.
    # More comments.
    def first_method(self):
        pass


# output


# Test for https://github.com/psf/black/issues/246.

some = statement


# This comment should be split from the statement above by two lines.
def function():
    pass


some = statement


# This multiline comments section
# should be split from the statement
# above by two lines.
def function():
    pass


some = statement


# This comment should be split from the statement above by two lines.
async def async_function():
    pass


some = statement


# This comment should be split from the statement above by two lines.
class MyClass:
    pass


some = statement
# This should be stick to the statement above


# This should be split from the above by two lines
class MyClassWithComplexLeadingComments:
    pass


class ClassWithDocstring:
    """A docstring."""


# Leading comment after a class with just a docstring
class MyClassAfterAnotherClassWithDocstring:
    pass


some = statement


# leading 1
@deco1
# leading 2
# leading 2 extra
@deco2(with_args=True)
# leading 3
@deco3
# leading 4
def decorated():
    pass


some = statement


# leading 1
@deco1
# leading 2
@deco2(with_args=True)

# leading 3 that already has an empty line
@deco3
# leading 4
def decorated_with_split_leading_comments():
    pass


some = statement


# leading 1
@deco1
# leading 2
@deco2(with_args=True)
# leading 3
@deco3

# leading 4 that already has an empty line
def decorated_with_split_leading_comments():
    pass


def main():
    if a:
        # Leading comment before inline function
        def inline():
            pass

        # Another leading comment
        def another_inline():
            pass

    else:
        # More leading comments
        def inline_after_else():
            pass


if a:
    # Leading comment before "top-level inline" function
    def top_level_quote_inline():
        pass

    # Another leading comment
    def another_top_level_quote_inline_inline():
        pass

else:
    # More leading comments
    def top_level_quote_inline_after_else():
        pass


class MyClass:
    # First method has no empty lines between bare class def.
    # More comments.
    def first_method(self):
        pass
