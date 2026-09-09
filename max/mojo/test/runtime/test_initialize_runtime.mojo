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

from max.algorithm import parallelize
from std.runtime import initialize_runtime, parallelism_level
from std.testing import assert_equal, assert_true, TestSuite


# The runtime is already initialized by the test's main wrapper, so this only
# exercises the idempotent path. The cold path (no Mojo main at all) is
# covered by the C-host lit test in KGEN/test/mojo-integration/shared-lib-c-host.
def test_initialize_runtime_idempotent() raises:
    initialize_runtime()
    initialize_runtime()
    assert_true(parallelism_level() >= 1)

    comptime N = 64
    var results = List[Int](length=N, fill=0)

    def fill(i: Int) {mut results}:
        results[i] = i

    parallelize(fill, N)

    for i in range(N):
        assert_equal(results[i], i)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
