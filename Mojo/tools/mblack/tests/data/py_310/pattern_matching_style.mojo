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
#   Path:   tests/data/py_310/pattern_matching_style.py
#
# ===----------------------------------------------------------------------=== #

match something:
    case b():
        print(1 + 1)
    case c(
        very_complex=True,
        perhaps_even_loooooooooooooooooooooooooooooooooooooong=-1,
    ):
        print(1)
    case c(
        very_complex=True,
        perhaps_even_loooooooooooooooooooooooooooooooooooooong=-1,
    ):
        print(2)
    case a:
        pass

match(arg)  # comment

match()

match()

case(arg)  # comment

case()

case()


re.match(something)  # fast
re.match()
match match():
    case case(
        arg,  # comment
    ):
        pass

# output

match something:
    case b():
        print(1 + 1)
    case c(
        very_complex=True,
        perhaps_even_loooooooooooooooooooooooooooooooooooooong=-1,
    ):
        print(1)
    case c(
        very_complex=True,
        perhaps_even_loooooooooooooooooooooooooooooooooooooong=-1,
    ):
        print(2)
    case a:
        pass

match(arg)  # comment

match()

match()

case(arg)  # comment

case()

case()


re.match(something)  # fast
re.match()
match match():
    case case(
        arg,  # comment
    ):
        pass
