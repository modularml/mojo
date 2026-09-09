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

from other.pkg.m import Nameable, Thing


# A parameter constrained by an imported trait: the constraint reaches the
# artifact as rendered text inside a debug-info source name, not as a symbol
# reference.
struct Wrap[T: Nameable]:
    def get(self) -> Int:
        return 0


def make() -> Int:
    var thing = Thing(42)
    return thing.get()
