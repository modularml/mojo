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
#   Path:   tests/data/simple_cases/comments3.py
#
# ===----------------------------------------------------------------------=== #

# The percent-percent comments are Spyder IDE cells.

#%%
def func():
    x = """
    a really long string
    """
    lcomp3 = [
        # This one is actually too long to fit in a single line.
        element.split("\n", 1)[0]
        # yup
        for element in collection.select_elements()
        # right
        if element is not None
    ]
    # Capture each of the exceptions in the MultiError along with each of their causes and contexts
    if isinstance(exc_value, MultiError):
        embedded = []
        for exc in exc_value.exceptions:
            if exc not in _seen:
                embedded.append(
                    # This should be left alone (before)
                    traceback.TracebackException.from_exception(
                        exc,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        # copy the set of _seen exceptions so that duplicates
                        # shared between sub-exceptions are not omitted
                        _seen=set(_seen),
                    )
                    # This should be left alone (after)
                )

    # everything is fine if the expression isn't nested
    traceback.TracebackException.from_exception(
        exc,
        limit=limit,
        lookup_lines=lookup_lines,
        capture_locals=capture_locals,
        # copy the set of _seen exceptions so that duplicates
        # shared between sub-exceptions are not omitted
        _seen=set(_seen),
    )


#%%
