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

# https://linear.app/modularml/issue/MOCO-773/error-init-result-type-must-be-elided-or-none-does-not-have-source
# The bug is that the location that should be in this file, but ends up on the package __init__.mojo instead.


struct LocTest:
    def __init__(out self) -> LocTest:
        pass
