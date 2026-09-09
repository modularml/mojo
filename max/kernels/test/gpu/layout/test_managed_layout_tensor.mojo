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

from max.gpu.host import DeviceContext
from layout import IntTuple, Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from std.testing import assert_equal

from std.utils import IndexList

# Tests for the ManagedLayoutTensor
# Verifies that device_tensor() and tensor() methods work correctly for various ranks


def test_managed_layout_tensor_1d() raises:
    """Test 1D ManagedLayoutTensor tensor operations."""
    comptime layout_1d = Layout(IntTuple(10))

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[.float32, layout_1d]()
    var host_tensor_1d = cpu_tensor.tensor[update=False]()
    assert_equal(comptime (host_tensor_1d.layout.rank()), 1)
    assert_equal(host_tensor_1d.dim[0](), 10)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[.float32, layout_1d](gpu_ctx)
    var device_tensor_1d = gpu_tensor.device_tensor[update=False]()
    assert_equal(comptime (device_tensor_1d.layout.rank()), 1)
    assert_equal(device_tensor_1d.dim[0](), 10)


def test_managed_layout_tensor_2d() raises:
    """Test 2D ManagedLayoutTensor tensor operations."""
    comptime layout_2d = Layout(IntTuple(4, 6))

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[.float32, layout_2d]()
    var host_tensor_2d = cpu_tensor.tensor[update=False]()
    assert_equal(comptime (host_tensor_2d.layout.rank()), 2)
    assert_equal(host_tensor_2d.dim[0](), 4)
    assert_equal(host_tensor_2d.dim[1](), 6)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[.float32, layout_2d](gpu_ctx)
    var device_tensor_2d = gpu_tensor.device_tensor[update=False]()
    assert_equal(comptime (device_tensor_2d.layout.rank()), 2)
    assert_equal(device_tensor_2d.dim[0](), 4)
    assert_equal(device_tensor_2d.dim[1](), 6)


def test_managed_layout_tensor_3d() raises:
    """Test 3D ManagedLayoutTensor tensor operations."""
    comptime layout_3d = Layout(IntTuple(2, 3, 4))

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[.float32, layout_3d]()
    var host_tensor_3d = cpu_tensor.tensor[update=False]()
    assert_equal(comptime (host_tensor_3d.layout.rank()), 3)
    assert_equal(host_tensor_3d.dim[0](), 2)
    assert_equal(host_tensor_3d.dim[1](), 3)
    assert_equal(host_tensor_3d.dim[2](), 4)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[.float32, layout_3d](gpu_ctx)
    var device_tensor_3d = gpu_tensor.device_tensor[update=False]()
    assert_equal(comptime (device_tensor_3d.layout.rank()), 3)
    assert_equal(device_tensor_3d.dim[0](), 2)
    assert_equal(device_tensor_3d.dim[1](), 3)
    assert_equal(device_tensor_3d.dim[2](), 4)


def test_managed_layout_tensor_dynamic() raises:
    """Test ManagedLayoutTensor with dynamic dimensions."""
    # Create layout with some dynamic dimensions
    comptime layout_dynamic = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 4)

    # Define runtime shape with actual values
    var runtime_shape = IndexList[3](5, 8, 4)
    var runtime_layout = RuntimeLayout[layout_dynamic].row_major(runtime_shape)

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[.float32, layout_dynamic](
        runtime_layout
    )
    var host_tensor_dynamic = cpu_tensor.tensor[update=False]()
    assert_equal(comptime (host_tensor_dynamic.layout.rank()), 3)
    assert_equal(host_tensor_dynamic.dim[0](), 5)
    assert_equal(host_tensor_dynamic.dim[1](), 8)
    assert_equal(host_tensor_dynamic.dim[2](), 4)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[.float32, layout_dynamic](
        runtime_layout, gpu_ctx
    )
    var device_tensor_dynamic = gpu_tensor.device_tensor[update=False]()
    assert_equal(comptime (device_tensor_dynamic.layout.rank()), 3)
    assert_equal(device_tensor_dynamic.dim[0](), 5)
    assert_equal(device_tensor_dynamic.dim[1](), 8)
    assert_equal(device_tensor_dynamic.dim[2](), 4)


def main() raises:
    """Main test function that runs all ManagedLayoutTensor tests."""
    test_managed_layout_tensor_1d()
    test_managed_layout_tensor_2d()
    test_managed_layout_tensor_3d()
    test_managed_layout_tensor_dynamic()
