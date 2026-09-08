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

import timing_pkg.sub
import timing_pkg.nested.leaf


# Body use of the dotted imports.
def body_use():
    var _x = timing_pkg.sub.VALUE
    var _y = timing_pkg.nested.leaf.DEEP


# Default-argument use of the same dotted imports. The full path is navigated,
# so every segment (timing_pkg, timing_pkg.nested, timing_pkg.nested.leaf) must
# be bound — including the middle `nested`.
def default_use(
    x: Int = timing_pkg.sub.VALUE, y: Int = timing_pkg.nested.leaf.DEEP
):
    pass
