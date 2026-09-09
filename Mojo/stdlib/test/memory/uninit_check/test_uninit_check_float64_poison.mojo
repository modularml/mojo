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
# Tests that loading a Float64 with the debug allocator poison pattern
# (largest finite value, DBL_MAX = 0x7FEFFFFFFFFFFFFF) triggers abort.

from std.memory import Pointer


# CHECK: UNINIT_READ at {{.*}}: dtype={{.*}}: load matched debug allocator poison sentinel
def main():
    var value = UInt64(0x7FEFFFFFFFFFFFFF)
    var ptr = Pointer(to=value).unsafe_bitcast[Float64]()
    _ = ptr.unsafe_load()
