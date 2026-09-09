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
#   Path:   tests/test_trans.py
#
# ===----------------------------------------------------------------------=== #

from typing import List, Tuple

from mblack.trans import iter_fexpr_spans


def test_fexpr_spans() -> None:
    def check(
        string: str,
        expected_spans: List[Tuple[int, int]],
        expected_slices: List[str],
    ) -> None:
        spans = list(iter_fexpr_spans(string))

        # Checking slices isn't strictly necessary, but it's easier to verify at
        # a glance than only spans
        assert len(spans) == len(expected_slices)
        for (i, j), slice in zip(spans, expected_slices):
            assert len(string[i:j]) == j - i
            assert string[i:j] == slice

        assert spans == expected_spans

    # Most of these test cases omit the leading 'f' and leading / closing quotes
    # for convenience
    # Some additional property-based tests can be found in
    # https://github.com/psf/black/pull/2654#issuecomment-981411748
    check("""{var}""", [(0, 5)], ["{var}"])
    check("""f'{var}'""", [(2, 7)], ["{var}"])
    check(
        """f'{1 + f() + 2 + "asdf"}'""",
        [(2, 24)],
        ["""{1 + f() + 2 + "asdf"}"""],
    )
    check("""text {var} text""", [(5, 10)], ["{var}"])
    check("""text {{ {var} }} text""", [(8, 13)], ["{var}"])
    check("""{a} {b} {c}""", [(0, 3), (4, 7), (8, 11)], ["{a}", "{b}", "{c}"])
    check("""f'{a} {b} {c}'""", [(2, 5), (6, 9), (10, 13)], ["{a}", "{b}", "{c}"])
    check("""{ {} }""", [(0, 6)], ["{ {} }"])
    check("""{ {{}} }""", [(0, 8)], ["{ {{}} }"])
    check("""{ {{{}}} }""", [(0, 10)], ["{ {{{}}} }"])
    check("""{{ {{{}}} }}""", [(5, 7)], ["{}"])
    check("""{{ {{{var}}} }}""", [(5, 10)], ["{var}"])
    check("""{f"{0}"}""", [(0, 8)], ["""{f"{0}"}"""])
    check("""{"'"}""", [(0, 5)], ["""{"'"}"""])
    check("""{"{"}""", [(0, 5)], ["""{"{"}"""])
    check("""{"}"}""", [(0, 5)], ["""{"}"}"""])
    check("""{"{{"}""", [(0, 6)], ["""{"{{"}"""])
    check("""{''' '''}""", [(0, 9)], ["""{''' '''}"""])
    check("""{'''{'''}""", [(0, 9)], ["""{'''{'''}"""])
    check("""{''' {'{ '''}""", [(0, 13)], ["""{''' {'{ '''}"""])
    check(
        '''f\'\'\'-{f"""*{f"+{f'.{x}.'}+"}*"""}-'y\\'\'\'\'''',
        [(5, 33)],
        ['''{f"""*{f"+{f'.{x}.'}+"}*"""}'''],
    )
    check(r"""{}{""", [(0, 2)], ["{}"])
    check("""f"{'{'''''''''}\"""", [(2, 15)], ["{'{'''''''''}"])
