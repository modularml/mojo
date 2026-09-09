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
#   Path:   tests/data/miscellaneous/docstring_preview_no_string_normalization.py
#
# ===----------------------------------------------------------------------=== #


def do_not_touch_this_prefix():
    R"""There was a bug where docstring prefixes would be normalized even with -S.
    """


def do_not_touch_this_prefix2():
    Rf"There was a bug where docstring prefixes would be normalized even with -S."


def do_not_touch_this_prefix3():
    """There was a bug where docstring prefixes would be normalized even with -S.
    """
