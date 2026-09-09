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
"""Wildcard-re-exported by the package __init__ via an *absolute* path
(`from std.import_suggestion.abs_wild_mod import *`). Absolute wildcards are not
resolved, so `abs_wildcard_symbol` is not found and gets no suggestion."""


def abs_wildcard_symbol():
    pass
