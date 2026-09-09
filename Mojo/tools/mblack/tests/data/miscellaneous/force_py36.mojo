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
#   Path:   tests/data/miscellaneous/force_py36.py
#
# ===----------------------------------------------------------------------=== #

# The input source must not contain any Py36-specific syntax (e.g. argument type
# annotations, trailing comma after *rest) or this test becomes invalid.
def long_function_name(
    argument_one,
    argument_two,
    argument_three,
    argument_four,
    argument_five,
    argument_six,
    *rest,
):
    ...


# output
# The input source must not contain any Py36-specific syntax (e.g. argument type
# annotations, trailing comma after *rest) or this test becomes invalid.
def long_function_name(
    argument_one,
    argument_two,
    argument_three,
    argument_four,
    argument_five,
    argument_six,
    *rest,
):
    ...
