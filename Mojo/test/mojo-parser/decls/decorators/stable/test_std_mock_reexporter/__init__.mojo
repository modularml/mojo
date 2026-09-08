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

# Mock reexporter package for testing that @stable(recursive=True) does not
# bleed across module boundaries.
#
# This module imports UnstableStruct with @stable(recursive=True), suppressing
# stability warnings within THIS file. That override must not propagate to
# files that import from this module.

@stable(recursive=True)
from test_std_mock import UnstableStruct
