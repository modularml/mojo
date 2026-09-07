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
# Tests that the poison check is disabled by default
# (without -D MOJO_STDLIB_SIMD_UNINIT_CHECK).
# Loading the poison pattern should NOT crash when the check is disabled.

from std.memory import Pointer


def test_poison_ignored_when_disabled():
    """With MOJO_STDLIB_SIMD_UNINIT_CHECK not set, loading the poison
    pattern should not abort."""
    var value = UInt32(0x7F7FFFFF)
    var ptr = Pointer(to=value).unsafe_bitcast[Float32]()
    _ = ptr.unsafe_load()


def main():
    test_poison_ignored_when_disabled()
