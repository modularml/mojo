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
"""Sub-package wildcard-re-exported by the parent (`from .wild_pkg import *`).
Exercises the sub-package branch of the wildcard resolver (its public surface is
its own __init__'s decls)."""


# Pulled into std.import_suggestion's surface via the sub-package wildcard.
def wildpkg_symbol():
    pass
