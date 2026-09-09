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
#   Path:   tests/data/py_310/pattern_matching_complex.py
#
# ===----------------------------------------------------------------------=== #

# Cases sampled from Lib/test/test_patma.py

# case black_test_patma_098
match x:
    case -0j:
        y = 0
# case black_test_patma_142
match x:
    case bytes(z):
        y = 0
# case black_test_patma_073
match x:
    case 0 if 0:
        y = 0
    case 0 if 1:
        y = 1
# case black_test_patma_006
match 3:
    case 0 | 1 | 2 | 3:
        x = True
# case black_test_patma_049
match x:
    case [0, 1] | [1, 0]:
        y = 0
# case black_check_sequence_then_mapping
match x:
    case [...]:
        return "seq"
    case {}:
        return "map"
# case black_test_patma_035
match x:
    case {0: [1, 2, {}]}:
        y = 0
    case {0: [1, 2, {}] | True} | {1: [[]]} | {0: [1, 2, {}]} | [] | "X" | {}:
        y = 1
    case []:
        y = 2
# case black_test_patma_107
match x:
    case 0.25 + 1.75j:
        y = 0
# case black_test_patma_097
match x:
    case -0j:
        y = 0
# case black_test_patma_007
match 4:
    case 0 | 1 | 2 | 3:
        x = True
# case black_test_patma_154
match x:
    case 0 if x:
        y = 0
# case black_test_patma_134
match x:
    case {1: 0}:
        y = 0
    case {0: 0}:
        y = 1
    case {**z}:
        y = 2
# case black_test_patma_185
match Seq():
    case [...]:
        y = 0
# case black_test_patma_063
match x:
    case 1:
        y = 0
    case 1:
        y = 1
# case black_test_patma_248
match x:
    case {"foo": bar}:
        y = bar
# case black_test_patma_019
match (0, 1, 2):
    case [0, 1, *x, 2]:
        y = 0
# case black_test_patma_052
match x:
    case [0]:
        y = 0
    case [1, 0] if (x := x[:0]):
        y = 1
    case [1, 0]:
        y = 2
# case black_test_patma_191
match w:
    case [x, y, ...]:
        z = 0
# case black_test_patma_110
match x:
    case -0.25 - 1.75j:
        y = 0
# case black_test_patma_151
match (x,):
    case [y]:
        z = 0
# case black_test_patma_114
match x:
    case A.B.C.D:
        y = 0
# case black_test_patma_232
match x:
    case None:
        y = 0
# case black_test_patma_058
match x:
    case 0:
        y = 0
# case black_test_patma_233
match x:
    case False:
        y = 0
# case black_test_patma_078
match x:
    case []:
        y = 0
    case [""]:
        y = 1
    case "":
        y = 2
# case black_test_patma_156
match x:
    case z:
        y = 0
# case black_test_patma_189
match w:
    case [x, y, *rest]:
        z = 0
# case black_test_patma_042
match x:
    case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
        y = 0
# case black_test_patma_034
match x:
    case {0: [1, 2, {}]}:
        y = 0
    case {0: [1, 2, {}] | False} | {1: [[]]} | {0: [1, 2, {}]} | [] | "X" | {}:
        y = 1
    case []:
        y = 2
