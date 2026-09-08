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

from std.testing import assert_equal
from std.ffi import external_call


def test_tracy_bridge_symbols_exist_and_disabled_by_default() raises:
    # Check that the TRACY_ENABLE query symbol exists and is callable.
    _ = external_call["KGEN_CompilerRT_TracyIsEnabled", Int]()

    # Begin/End should behave based on `enabled`. Tracy is built with
    # TRACY_ON_DEMAND and no profiler client is connected in this test, so
    # the zone is inactive and packs to 0 even when Tracy is enabled; when
    # Tracy is disabled the bridge is a no-op returning 0. Either way End(0)
    # must be safe.
    var name = "tracy_bridge_test"
    var ctx = external_call["KGEN_CompilerRT_TracyZoneBegin", UInt64](
        name.unsafe_ptr(), name.byte_length(), 0
    )
    assert_equal(UInt64(0), ctx)
    external_call["KGEN_CompilerRT_TracyZoneEnd", NoneType](ctx)


def main() raises:
    test_tracy_bridge_symbols_exist_and_disabled_by_default()
