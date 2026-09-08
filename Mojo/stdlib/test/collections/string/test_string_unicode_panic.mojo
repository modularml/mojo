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
# RUN: %mojo-build %s -o %t
# RUN: not not %t 1 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not not %t 2 2>&1 | FileCheck --check-prefix CHECK_2 %s
# RUN: %t 3 2>&1 | FileCheck --check-prefix CHECK_3 %s
# RUN: not not %t 4 2>&1 | FileCheck --check-prefix CHECK_4 %s
# RUN: not not %t 5 2>&1 | FileCheck --check-prefix CHECK_5 %s
# RUN: not not %t 6 2>&1 | FileCheck --check-prefix CHECK_6 %s
# RUN: not not %t 7 2>&1 | FileCheck --check-prefix CHECK_7 %s


from std.sys.arg import argv


def main() raises:
    if len(argv()) <= 1:
        return
    var test = argv()[1]
    if test == "1":
        var s = String("😀longlonglonglong")
        # CHECK_1: does not lie on a codepoint boundary.
        s.resize(1)
    elif test == "2":
        var s = String("😀longlonglonglong")
        # CHECK_2: does not lie on a codepoint boundary.
        s.resize(unsafe_uninit_length=1)
    elif test == "3":
        var s = String("😀😃")
        s.resize(4)
        s.resize(4)
        s.resize(5, 127)
        var s2 = String("😀😃")
        s2.resize(unsafe_uninit_length=4)
        s2.resize(unsafe_uninit_length=4)
        # CHECK_3: OK
        print("OK")
    elif test == "4":
        var s = String("😀😃")
        s.resize(7)
        # CHECK_4: does not lie on a codepoint boundary.
    elif test == "5":
        var s = String()
        s.resize(10, 128)
        # CHECK_5: Fill byte is the start of a multi-byte character.
    elif test == "6":
        var s = String("😌😃")
        # CHECK_6: does not lie on a codepoint boundary.
        var _y = s[byte=1]
    elif test == "7":
        var s = String("😌😃")
        # CHECK_7: is not a codepoint boundary.
        var _y = s[byte=0:5]
