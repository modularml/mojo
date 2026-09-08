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
#   Path:   tests/data/preview/remove_for_brackets.py
#
# ===----------------------------------------------------------------------=== #

# Only remove tuple brackets after `for`
for (k, v) in d.items():
    print(k, v)

# Don't touch tuple brackets after `in`
for module in (core, _unicodefun):
    if hasattr(module, "_verify_python3_env"):
        module._verify_python3_env = lambda: None

# Brackets remain for long for loop lines
for (
    why_would_anyone_choose_to_name_a_loop_variable_with_a_name_this_long,
    i_dont_know_but_we_should_still_check_the_behaviour_if_they_do,
) in d.items():
    print(k, v)

for (
    k,
    v,
) in (
    dfkasdjfldsjflkdsjflkdsjfdslkfjldsjfgkjdshgkljjdsfldgkhsdofudsfudsofajdslkfjdslkfjldisfjdffjsdlkfjdlkjjkdflskadjldkfjsalkfjdasj.items()
):
    print(k, v)

# Test deeply nested brackets
for (k, v) in d.items():
    print(k, v)

# output
# Only remove tuple brackets after `for`
for k, v in d.items():
    print(k, v)

# Don't touch tuple brackets after `in`
for module in (core, _unicodefun):
    if hasattr(module, "_verify_python3_env"):
        module._verify_python3_env = lambda: None

# Brackets remain for long for loop lines
for (
    why_would_anyone_choose_to_name_a_loop_variable_with_a_name_this_long,
    i_dont_know_but_we_should_still_check_the_behaviour_if_they_do,
) in d.items():
    print(k, v)

for (
    k,
    v,
) in (
    dfkasdjfldsjflkdsjflkdsjfdslkfjldsjfgkjdshgkljjdsfldgkhsdofudsfudsofajdslkfjdslkfjldisfjdffjsdlkfjdlkjjkdflskadjldkfjsalkfjdasj.items()
):
    print(k, v)

# Test deeply nested brackets
for k, v in d.items():
    print(k, v)
