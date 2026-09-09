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
"""FLUX.2 ModuleV3 diffusion architecture (single ``Module`` implementation)."""

from .arch import flux2_modulev3_arch
from .flux2_executor import FLUXModule
from .flux2_inputs import Flux2ModuleV3Inputs

__all__ = [
    "FLUXModule",
    "Flux2ModuleV3Inputs",
    "flux2_modulev3_arch",
]
