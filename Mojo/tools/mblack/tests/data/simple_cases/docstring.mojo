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
#   Path:   tests/data/simple_cases/docstring.py
#
# ===----------------------------------------------------------------------=== #


class MyClass:
    """Multiline
    class docstring
    """

    def method(self):
        """Multiline
        method docstring
        """
        pass


def foo():
    """This is a docstring with
    some lines of text here
    """
    return


def bar():
    """This is another docstring
    with more lines of text
    """
    return


def baz():
    '''"This" is a string with some
    embedded "quotes"'''
    return


def troz():
    """Indentation with tabs
    is just as OK
    """
    return


def zort():
    """Another
    multiline
    docstring
    """
    pass


def poit():
    """
    Lorem ipsum dolor sit amet.

    Consectetur adipiscing elit:
     - sed do eiusmod tempor incididunt ut labore
     - dolore magna aliqua
       - enim ad minim veniam
       - quis nostrud exercitation ullamco laboris nisi
     - aliquip ex ea commodo consequat
    """
    pass


def under_indent():
    """
      These lines are indented in a way that does not
    make sense.
    """
    pass


def over_indent():
    """
    This has a shallow indent
      - But some lines are deeper
      - And the closing quote is too deep
    """
    pass


def single_line():
    """But with a newline after it!"""
    pass


def this():
    r"""
    'hey ho'
    """


def that():
    """ "hey yah" """


def and_that():
    """
    "hey yah" """


def and_this():
    '''
    "hey yah"'''


def multiline_whitespace():
    """ """


def oneline_whitespace():
    """ """


def empty():
    """"""


def single_quotes():
    "testing"


def believe_it_or_not_this_is_in_the_py_stdlib():
    '''
    "hey yah"'''


def ignored_docstring():
    """a => \
b"""


def single_line_docstring_with_whitespace():
    """This should be stripped"""


def docstring_with_inline_tabs_and_space_indentation():
    """hey

    tab	separated	value
        tab at start of line and then a tab	separated	value
                                multiple tabs at the beginning	and	inline
                        mixed tabs and spaces at beginning. next line has mixed tabs and spaces only.

    line ends with some tabs
    """


def docstring_with_inline_tabs_and_tab_indentation():
    """hey

    tab	separated	value
            tab at start of line and then a tab	separated	value
                                    multiple tabs at the beginning	and	inline
                            mixed tabs and spaces at beginning. next line has mixed tabs and spaces only.

    line ends with some tabs
    """
    pass


def backslash_space():
    """\ """


def multiline_backslash_1():
    """
  hey\there\
  \ """


def multiline_backslash_2():
    """
    hey there \ """


def multiline_backslash_3():
    """
    already escaped \\"""


def my_god_its_full_of_stars_1():
    "I'm sorry Dave\u2001"


# the space below is actually a \u2001, removed in output
def my_god_its_full_of_stars_2():
    "I'm sorry Dave"


def docstring_almost_at_line_limit():
    """long docstring................................................................."""


def docstring_almost_at_line_limit2():
    """long docstring.................................................................

    ..................................................................................
    """


def docstring_at_line_limit():
    """long docstring................................................................"""


def multiline_docstring_at_line_limit():
    """first line-----------------------------------------------------------------------

    second line----------------------------------------------------------------------"""


def stable_quote_normalization_with_immediate_inner_single_quote(self):
    """'<text here>

    <text here, since without another non-empty line black is stable>
    """


# output


class MyClass:
    """Multiline
    class docstring
    """

    def method(self):
        """Multiline
        method docstring
        """
        pass


def foo():
    """This is a docstring with
    some lines of text here
    """
    return


def bar():
    """This is another docstring
    with more lines of text
    """
    return


def baz():
    '''"This" is a string with some
    embedded "quotes"'''
    return


def troz():
    """Indentation with tabs
    is just as OK
    """
    return


def zort():
    """Another
    multiline
    docstring
    """
    pass


def poit():
    """
    Lorem ipsum dolor sit amet.

    Consectetur adipiscing elit:
     - sed do eiusmod tempor incididunt ut labore
     - dolore magna aliqua
       - enim ad minim veniam
       - quis nostrud exercitation ullamco laboris nisi
     - aliquip ex ea commodo consequat
    """
    pass


def under_indent():
    """
      These lines are indented in a way that does not
    make sense.
    """
    pass


def over_indent():
    """
    This has a shallow indent
      - But some lines are deeper
      - And the closing quote is too deep
    """
    pass


def single_line():
    """But with a newline after it!"""
    pass


def this():
    r"""
    'hey ho'
    """


def that():
    """ "hey yah" """


def and_that():
    """
    "hey yah" """


def and_this():
    '''
    "hey yah"'''


def multiline_whitespace():
    """ """


def oneline_whitespace():
    """ """


def empty():
    """"""


def single_quotes():
    "testing"


def believe_it_or_not_this_is_in_the_py_stdlib():
    '''
    "hey yah"'''


def ignored_docstring():
    """a => \
b"""


def single_line_docstring_with_whitespace():
    """This should be stripped"""


def docstring_with_inline_tabs_and_space_indentation():
    """hey

    tab	separated	value
        tab at start of line and then a tab	separated	value
                                multiple tabs at the beginning	and	inline
                        mixed tabs and spaces at beginning. next line has mixed tabs and spaces only.

    line ends with some tabs
    """


def docstring_with_inline_tabs_and_tab_indentation():
    """hey

    tab	separated	value
            tab at start of line and then a tab	separated	value
                                    multiple tabs at the beginning	and	inline
                            mixed tabs and spaces at beginning. next line has mixed tabs and spaces only.

    line ends with some tabs
    """
    pass


def backslash_space():
    """\ """


def multiline_backslash_1():
    """
  hey\there\
  \ """


def multiline_backslash_2():
    """
    hey there \ """


def multiline_backslash_3():
    """
    already escaped \\"""


def my_god_its_full_of_stars_1():
    "I'm sorry Dave\u2001"


# the space below is actually a \u2001, removed in output
def my_god_its_full_of_stars_2():
    "I'm sorry Dave"


def docstring_almost_at_line_limit():
    """long docstring................................................................."""


def docstring_almost_at_line_limit2():
    """long docstring.................................................................

    ..................................................................................
    """


def docstring_at_line_limit():
    """long docstring................................................................"""


def multiline_docstring_at_line_limit():
    """first line-----------------------------------------------------------------------

    second line----------------------------------------------------------------------"""


def stable_quote_normalization_with_immediate_inner_single_quote(self):
    """'<text here>

    <text here, since without another non-empty line black is stable>
    """
