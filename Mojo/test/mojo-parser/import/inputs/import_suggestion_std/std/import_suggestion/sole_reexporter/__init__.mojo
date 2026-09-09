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
"""Sole exposer of `foreign_only_symbol`: re-exports it from the foreign sibling
module `std.import_suggestion.solo_mod` (an absolute cross-package import). No
package owns the name natively, so this re-exporter is the fallback suggestion."""

from std.import_suggestion.solo_mod import foreign_only_symbol
