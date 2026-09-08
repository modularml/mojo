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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence

import max._core

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

def MOToMOGG(
    kernel_library_paths: Sequence[str] = [], force_sync: bool = False
) -> max._core.Pass:
    """
    This pass performs a conversion from known `mo` dialect ops to registered
    `mogg` dialect ops, performing kernel selection along the way.

    TODO: Expand this to also perform lowering search
    """
