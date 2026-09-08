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
"""Convenience aggregator: re-exports `native_owned_symbol` from the foreign
package `std.import_suggestion.owner` (an absolute cross-package import), the
way `std.ffi` re-exports `std.os.abort`. The suggestion must prefer the native
owner (`std.import_suggestion.owner`), not this aggregator."""

from std.import_suggestion.owner import native_owned_symbol
