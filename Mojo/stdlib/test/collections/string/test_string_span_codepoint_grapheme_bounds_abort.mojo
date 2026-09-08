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
#
# Verifies that invalid `codepoint=`/`grapheme=` `ContiguousSlice` indexing
# (out of bounds, reversed, or negative) aborts on `StringSlice` and `String`
# instead of silently clamping. Mirrors
# `test_string_slice_bounds_abort.mojo`'s coverage of `byte=` slicing.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s
# RUN: not %mojo -D test=3 %s 2>&1 | FileCheck --check-prefix CHECK_3 %s
# RUN: not %mojo -D test=4 %s 2>&1 | FileCheck --check-prefix CHECK_4 %s
# RUN: not %mojo -D test=5 %s 2>&1 | FileCheck --check-prefix CHECK_5 %s
# RUN: not %mojo -D test=6 %s 2>&1 | FileCheck --check-prefix CHECK_6 %s
# RUN: not %mojo -D test=7 %s 2>&1 | FileCheck --check-prefix CHECK_7 %s
# RUN: not %mojo -D test=8 %s 2>&1 | FileCheck --check-prefix CHECK_8 %s
# RUN: not %mojo -D test=9 %s 2>&1 | FileCheck --check-prefix CHECK_9 %s

from std.sys import get_defined_int


def main() raises:
    comptime TEST = get_defined_int["test"]()
    comptime if TEST == 1:
        # 3 codepoints: one flag emoji per codepoint slot.
        var s = StringSlice("🔄🔥🔄")
        # CHECK_1: {{.*}}: Assert Error: slice end index 100 is out of bounds, valid range is 0 to 3
        _ = s[codepoint=0:100]
    elif TEST == 2:
        var s = StringSlice("🔄🔥🔄")
        # CHECK_2: {{.*}}: Assert Error: slice start index 3 is greater than slice end index 1
        _ = s[codepoint=3:1]
    elif TEST == 3:
        var s = StringSlice("🔄🔥🔄")
        # CHECK_3: {{.*}}: Assert Error: slice start index -1 is out of bounds, valid range is 0 to 3
        _ = s[codepoint= -1:]
    elif TEST == 4:
        # Through `String`'s `codepoint=` accessor, which delegates to
        # `StringSlice`.
        var s = String("🔄🔥🔄")
        # CHECK_4: {{.*}}: Assert Error: slice end index 100 is out of bounds, valid range is 0 to 3
        _ = s[codepoint=0:100]
    elif TEST == 5:
        # "cafe" + combining acute accent -- 4 graphemes, 5 codepoints.
        var s = StringSlice("café")
        # CHECK_5: {{.*}}: Assert Error: slice end index 100 is out of bounds, valid range is 0 to 4
        _ = s[grapheme=0:100]
    elif TEST == 6:
        var s = StringSlice("café")
        # CHECK_6: {{.*}}: Assert Error: slice start index 3 is greater than slice end index 1
        _ = s[grapheme=3:1]
    elif TEST == 7:
        var s = StringSlice("café")
        # CHECK_7: {{.*}}: Assert Error: slice start index -1 is out of bounds, valid range is 0 to 4
        _ = s[grapheme= -1:]
    elif TEST == 8:
        var s = StringSlice("café")
        # An out-of-range start with no end still requires a full scan to
        # discover the real grapheme count for the abort message.
        # CHECK_8: {{.*}}: Assert Error: slice start index 100 is out of bounds, valid range is 0 to 4
        _ = s[grapheme=100:]
    elif TEST == 9:
        var s = StringSlice("café")
        # CHECK_9: {{.*}}: Assert Error: slice end index -1 is out of bounds, valid range is 0 to 4
        _ = s[grapheme=0:-1]
    else:
        comptime assert False, "unreachable!"
