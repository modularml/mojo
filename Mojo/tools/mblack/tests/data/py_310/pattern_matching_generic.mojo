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
#   Path:   tests/data/py_310/pattern_matching_generic.py
#
# ===----------------------------------------------------------------------=== #

re.match()
match = a
with match() as match:
    match = f"{match}"

re.match()
match = a
with match() as match:
    match = f"{match}"


def get_grammars(target_versions: Set[TargetVersion]) -> List[Grammar]:
    if not target_versions:
        # No target_version specified, so try all grammars.
        return [
            # Python 3.7+
            pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords,
            # Python 3.0-3.6
            pygram.python_grammar_no_print_statement_no_exec_statement,
            # Python 2.7 with future print_function import
            pygram.python_grammar_no_print_statement,
            # Python 2.7
            pygram.python_grammar,
        ]

    match match:
        case case:
            match match:
                case case:
                    pass

    if all(version.is_python2() for version in target_versions):
        # Python 2-only code, so try Python 2 grammars.
        return [
            # Python 2.7 with future print_function import
            pygram.python_grammar_no_print_statement,
            # Python 2.7
            pygram.python_grammar,
        ]

    re.match()
    match = a
    with match() as match:
        match = f"{match}"

    def test_patma_139(self):
        x = False
        match x:
            case bool(z):
                y = 0
        self.assertIs(x, False)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    # Python 3-compatible code, so only try Python 3 grammar.
    grammars = []
    if supports_feature(target_versions, Feature.PATTERN_MATCHING):
        # Python 3.10+
        grammars.append(pygram.python_grammar_soft_keywords)
    # If we have to parse both, try to parse async as a keyword first
    if not supports_feature(
        target_versions, Feature.ASYNC_IDENTIFIERS
    ) and not supports_feature(target_versions, Feature.PATTERN_MATCHING):
        # Python 3.7-3.9
        grammars.append(
            pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords
        )
    if not supports_feature(target_versions, Feature.ASYNC_KEYWORDS):
        # Python 3.0-3.6
        grammars.append(
            pygram.python_grammar_no_print_statement_no_exec_statement
        )

    def test_patma_155(self):
        x = 0
        y = None
        match x:
            case 1e1000:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

        x = range(3)
        match x:
            case [y, case as x, z]:
                w = 0

    # At least one of the above branches must have been taken, because every Python
    # version has exactly one of the two 'ASYNC_*' flags
    return grammars


def lib2to3_parse(
    src_txt: str, target_versions: Iterable[TargetVersion] = ()
) -> Node:
    """Given a string with source, return the lib2to3 Node."""
    if not src_txt.endswith("\n"):
        src_txt += "\n"

    grammars = get_grammars(set(target_versions))


re.match()
match = a
with match() as match:
    match = f"{match}"

re.match()
match = a
with match() as match:
    match = f"{match}"
