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

from std.sys import Vendor
from std.testing import TestSuite, assert_equal, assert_not_equal


def test_vendor_write_to() raises:
    assert_equal(String(Vendor.NO_GPU), "no_gpu")
    assert_equal(String(Vendor.AMD_GPU), "amd_gpu")
    assert_equal(String(Vendor.NVIDIA_GPU), "nvidia_gpu")
    assert_equal(String(Vendor.APPLE_GPU), "apple_gpu")


def test_vendor_equality() raises:
    assert_equal(Vendor.AMD_GPU, Vendor.AMD_GPU)
    assert_not_equal(Vendor.AMD_GPU, Vendor.NVIDIA_GPU)
    assert_not_equal(Vendor.NO_GPU, Vendor.APPLE_GPU)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
