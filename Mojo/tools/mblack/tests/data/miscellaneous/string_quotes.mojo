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
#   Path:   tests/data/miscellaneous/string_quotes.py
#
# ===----------------------------------------------------------------------=== #

''''''
"'"
'"'
"'"
'"'
"Hello"
"Don't do that"
'Here is a "'
"What's the deal here?"
'What\'s the deal "here"?'
'And "here"?'
"""Strings with "" in them"""
"""Strings with "" in them"""
'''Here's a "'''
"""Here's a " """
"""Just a normal triple
quote"""
f"just a normal {f} string"
f"""This is a triple-quoted {f}-string"""
f'MOAR {" ".join([])}'
f"MOAR {' '.join([])}"
r"raw string ftw"
r"Date d\'expiration:(.*)"
r'Tricky "quote'
r"Not-so-tricky \"quote"
rf"{yay}"
"\nThe \"quick\"\nbrown fox\njumps over\nthe 'lazy' dog.\n"
re.compile(r'[\\"]')
"x = ''; y = \"\""
"x = '''; y = \"\""
"x = ''''; y = \"\""
"x = '' ''; y = \"\""
'x = \'\'; y = """'
'x = \'\'\'; y = """"'
'x = \'\'\'\'; y = """""'
'x = \'\' \'\'; y = """""'
'unnecessary ""escaping'
"unnecessary ''escaping"
'\\""'
"\\''"
"Lots of \\\\\\\\'quotes'"
f'{y * " "} \'{z}\''
f"{{y * \" \"}} '{z}'"
f'\'{z}\' {y * " "}'
f"{y * x} '{z}'"
"'{z}' {y * \" \"}"
"{y * x} '{z}'"

# We must bail out if changing the quotes would introduce backslashes in f-string
# expressions. xref: https://github.com/psf/black/issues/2348
f"\"{b}\"{' ' * (long-len(b)+1)}: \"{sts}\",\n"
f"\"{a}\"{'hello' * b}\"{c}\""

# output

""""""
"'"
'"'
"'"
'"'
"Hello"
"Don't do that"
'Here is a "'
"What's the deal here?"
'What\'s the deal "here"?'
'And "here"?'
"""Strings with "" in them"""
"""Strings with "" in them"""
'''Here's a "'''
"""Here's a " """
"""Just a normal triple
quote"""
f"just a normal {f} string"
f"""This is a triple-quoted {f}-string"""
f'MOAR {" ".join([])}'
f"MOAR {' '.join([])}"
r"raw string ftw"
r"Date d\'expiration:(.*)"
r'Tricky "quote'
r"Not-so-tricky \"quote"
rf"{yay}"
"\nThe \"quick\"\nbrown fox\njumps over\nthe 'lazy' dog.\n"
re.compile(r'[\\"]')
"x = ''; y = \"\""
"x = '''; y = \"\""
"x = ''''; y = \"\""
"x = '' ''; y = \"\""
'x = \'\'; y = """'
'x = \'\'\'; y = """"'
'x = \'\'\'\'; y = """""'
'x = \'\' \'\'; y = """""'
'unnecessary ""escaping'
"unnecessary ''escaping"
'\\""'
"\\''"
"Lots of \\\\\\\\'quotes'"
f'{y * " "} \'{z}\''
f"{{y * \" \"}} '{z}'"
f'\'{z}\' {y * " "}'
f"{y * x} '{z}'"
"'{z}' {y * \" \"}"
"{y * x} '{z}'"

# We must bail out if changing the quotes would introduce backslashes in f-string
# expressions. xref: https://github.com/psf/black/issues/2348
f"\"{b}\"{' ' * (long-len(b)+1)}: \"{sts}\",\n"
f"\"{a}\"{'hello' * b}\"{c}\""
