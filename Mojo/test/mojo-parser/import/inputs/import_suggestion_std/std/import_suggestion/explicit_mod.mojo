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
"""Submodule whose symbols are (or are not) re-exported by the package."""


# Re-exported by __init__ -> suggestable as `std.import_suggestion`.
def explicit_reexport():
    pass


# Re-exported by __init__ but private -> never suggested.
def _private_reexport():
    pass


# Defined here but NOT re-exported -> not public API, so not suggested even
# though it is reachable via the full submodule path.
def not_reexported_symbol():
    pass
