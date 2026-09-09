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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-LABEL: lit.fn @"test_raw_t_strings()"
def test_raw_t_strings():
    var name = "Alice"
    var x = 42

    # Basic raw t-string with rt prefix — backslash is literal
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Hello\\nWorld {}"
    var s1 = rt"Hello\nWorld {name}"

    # Basic raw t-string with tr prefix
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Hello\\tWorld {}"
    var s2 = tr"Hello\tWorld {name}"

    # Raw t-string with no interpolations
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "raw\\nstring"
    var s3 = rt"raw\nstring"

    # Raw t-string with multiple interpolations
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Path: C:\\Users\\{}"
    var s4 = rt"Path: C:\Users\{name}"

    # Raw t-string with escaped braces (still work in raw mode)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{[{][{]}}braces{{[}][}]}}"
    var s5 = rt"{{braces}}"

    # Triple-quoted raw t-string
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "\0A    raw\\n{}\0A    "
    var s6 = rt"""
    raw\n{x}
    """

    # Uppercase prefix variants
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Rt {}"
    var s7 = Rt"Rt {x}"

    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "rT {}"
    var s8 = rT"rT {x}"

    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "RT {}"
    var s9 = RT"RT {x}"

    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "tR {}"
    var s10 = tR"tR {x}"

    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Tr {}"
    var s11 = Tr"Tr {x}"

    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "TR {}"
    var s12 = TR"TR {x}"

    # Raw t-string with hex escape (stays literal)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "A\\x42 {}"
    var s13 = rt"A\x42 {x}"

    # Raw t-string with octal escape (stays literal)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "A\\012 {}"
    var s14 = rt"A\012 {x}"

    # Raw t-string with double backslash
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "AB\\\\ {}"
    var s15 = rt"AB\\ {x}"

    # Adjacent interpolations in raw t-string
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}{}"
    var s16 = rt"{name}{x}"

    # Raw t-string with single quotes
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "raw\\n {}"
    var s17 = rt'raw\n {x}'

    # Raw t-string concatenation
    # fmt: off
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "raw\\n{} also\\t{}"
    var s18 = rt"raw\n{x}" rt" also\t{name}"
    # fmt: on

    # Mixed raw + cooked t-string concatenation
    # fmt: off
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "raw\\n{} cooked\0A{}"
    var s19 = rt"raw\n{x}" t" cooked\n{name}"
    # fmt: on
