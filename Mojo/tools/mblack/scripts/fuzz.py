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
#   Path:   scripts/fuzz.py
#
# ===----------------------------------------------------------------------=== #

"""Property-based tests for Black.

By Zac Hatfield-Dodds, based on my Hypothesmith tool for source code
generation.  You can run this file with `python`, `pytest`, or (soon)
a coverage-guided fuzzer I'm working on.
"""

import re

import hypothesmith
import mblack
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from mblib2to3.pgen2.tokenize import TokenError


# This test uses the Hypothesis and Hypothesmith libraries to generate random
# syntactically-valid Python source code and run Black in odd modes.
@settings(
    max_examples=1000,  # roughly 1k tests/minute, or half that under coverage
    derandomize=True,  # deterministic mode to avoid CI flakiness
    deadline=None,  # ignore Hypothesis' health checks; we already know that
    suppress_health_check=HealthCheck.all(),  # this is slow and filter-heavy.
)
@given(
    # Note that while Hypothesmith might generate code unlike that written by
    # humans, it's a general test that should pass for any *valid* source code.
    # (so e.g. running it against code scraped of the internet might also help)
    src_contents=hypothesmith.from_grammar() | hypothesmith.from_node(),
    # Using randomly-varied modes helps us to exercise less common code paths.
    mode=st.builds(
        mblack.FileMode,
        line_length=st.just(88) | st.integers(0, 200),
        string_normalization=st.booleans(),
        preview=st.booleans(),
        is_pyi=st.booleans(),
        magic_trailing_comma=st.booleans(),
    ),
)
def test_idempotent_any_syntactically_valid_python(
    src_contents: str, mode: mblack.FileMode
) -> None:
    # Before starting, let's confirm that the input string is valid Python:
    compile(src_contents, "<string>", "exec")  # else the bug is in hypothesmith

    # Then format the code...
    try:
        dst_contents = mblack.format_str(src_contents, mode=mode)
    except mblack.InvalidInput:
        # This is a bug - if it's valid Python code, as above, Black should be
        # able to cope with it.  See issues #970, #1012
        # TODO: remove this try-except block when issues are resolved.
        return
    except TokenError as e:
        if (  # Special-case logic for backslashes followed by newlines or end-of-input
            e.args[0] == "EOF in multi-line statement"
            and re.search(r"\\($|\r?\n)", src_contents) is not None
        ):
            # This is a bug - if it's valid Python code, as above, Black should be
            # able to cope with it.  See issue #1012.
            # TODO: remove this block when the issue is resolved.
            return
        raise

    # And check that we got equivalent and stable output.
    mblack.assert_equivalent(src_contents, dst_contents)
    mblack.assert_stable(src_contents, dst_contents, mode=mode)

    # Future test: check that pure-python and mypyc versions of black
    # give identical output for identical input?


if __name__ == "__main__":
    # Run tests, including shrinking and reporting any known failures.
    test_idempotent_any_syntactically_valid_python()

    # If Atheris is available, run coverage-guided fuzzing.
    # (if you want only bounded fuzzing, just use `pytest fuzz.py`)
    try:
        import sys

        import atheris
    except ImportError:
        pass
    else:
        test = test_idempotent_any_syntactically_valid_python
        atheris.Setup(
            sys.argv,
            test.hypothesis.fuzz_one_input,  # type: ignore[attr-defined]
        )
        atheris.Fuzz()
