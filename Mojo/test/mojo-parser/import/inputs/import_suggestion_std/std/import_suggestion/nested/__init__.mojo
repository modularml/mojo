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
"""Nested sub-package: verifies the suggestion uses the deep package path."""

# An explicit re-export inside a nested sub-package's __init__; suggestable as
# the nested package path `std.import_suggestion.nested`.
from .detail import nested_reexport


# A direct declaration in a nested sub-package's __init__.
def nested_symbol():
    pass
