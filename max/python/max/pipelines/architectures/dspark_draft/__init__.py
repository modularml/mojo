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
"""Model-agnostic DSpark draft: block nn.Module + registry placeholder.

Speculators-format DSpark drafters declare the generic HuggingFace
architectures ``["DSparkDraftModel"]`` shared across target model lines,
and the draft nn.Module is likewise target-agnostic: each unified target
architecture imports it and drives it inside its compiled graph. The
format helpers (config parsing, d2t vocabulary map, checkpoint weight
handling) live in the ``speculators_common`` package.
"""

from .arch import dspark_speculators_draft_arch
from .dspark_speculators_draft import DSparkSpeculatorsDraft
from .model import DSparkDraftPlaceholderModel

__all__ = [
    "DSparkDraftPlaceholderModel",
    "DSparkSpeculatorsDraft",
    "dspark_speculators_draft_arch",
]
