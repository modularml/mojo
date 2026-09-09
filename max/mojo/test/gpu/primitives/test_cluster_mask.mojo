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
"""Tests for `cluster_mask_base`.

The base mask for axis 0 covers the first row of the cluster (CTA ranks are
contiguous along axis 0); the base mask for axis 1 covers the first column,
i.e. exactly one bit every `cluster_shape[0]` ranks. Bits at ranks >= the
cluster size are illegal in multicast/mbarrier-arrive masks and trap on
device, so shapes with `cluster_shape[1] > 2` (e.g. (2, 4, 1)) guard against
the mask over-extending past the cluster.
"""

from max.gpu.primitives.cluster import cluster_mask_base
from std.testing import assert_equal, TestSuite
from std.utils import Index


def test_axis0_masks() raises:
    assert_equal(cluster_mask_base[Index(1, 1, 1), 0](), 0b1)
    assert_equal(cluster_mask_base[Index(2, 1, 1), 0](), 0b11)
    assert_equal(cluster_mask_base[Index(4, 2, 1), 0](), 0b1111)
    assert_equal(cluster_mask_base[Index(2, 4, 1), 0](), 0b11)
    assert_equal(cluster_mask_base[Index(4, 4, 1), 0](), 0b1111)


def test_axis1_masks() raises:
    assert_equal(cluster_mask_base[Index(1, 1, 1), 1](), 0b1)
    assert_equal(cluster_mask_base[Index(4, 1, 1), 1](), 0b1)
    assert_equal(cluster_mask_base[Index(2, 2, 1), 1](), 0b101)
    assert_equal(cluster_mask_base[Index(4, 2, 1), 1](), 0b1_0001)


def test_axis1_masks_tall_clusters() raises:
    # cluster_shape[1] > 2 previously compounded the whole mask each
    # iteration, setting bits past the cluster size (0x1555 for (2, 4, 1))
    # which traps on device when used as a multicast mask.
    assert_equal(cluster_mask_base[Index(2, 4, 1), 1](), 0b0101_0101)
    assert_equal(cluster_mask_base[Index(1, 4, 1), 1](), 0b1111)
    assert_equal(cluster_mask_base[Index(4, 4, 1), 1](), 0b0001_0001_0001_0001)
    assert_equal(cluster_mask_base[Index(1, 8, 1), 1](), 0b1111_1111)
    assert_equal(cluster_mask_base[Index(2, 8, 1), 1](), 0b0101_0101_0101_0101)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
