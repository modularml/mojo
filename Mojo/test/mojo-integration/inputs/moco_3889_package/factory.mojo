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

from .inner import Inner


struct Factory:
    def __init__(out self):
        pass

    # make() references Inner both in its return type and body, causing
    # Inner.__init__'s FuncSymbolAttr to be walked when Factory is
    # body-resolved.  Before the fix, a stale MLIR SymbolTable cache (built
    # before Inner's parent package was materialized) could permanently mark
    # that FuncSymbolAttr as unresolvable, leaving Inner.__init__ absent from
    # declForFuncSymbol and triggering a KGEN verifier error.
    def make(self) -> Inner:
        return Inner(42)
