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
"""Real by-value GPU launch test for `Coord`'s `DevicePassable` conformance.

`test_device_passable.mojo` covers `Coord`'s host-side encoding (bytes into a
host buffer via `DefaultDeviceTypeEncoder`). This test closes the remaining gap
by launching a kernel that receives a `Coord` and reads it back on the device,
so the host->device transfer is exercised end to end.

`Coord` reaches a launch embedded in a `DevicePassable` composite (as
`TileTensor.layout` carries `Coord` `_shape`/`_stride`), so the wrapper here
mirrors that path via `encode_fields`.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite
from std.utils.coord import Coord


# A `DevicePassable` composite carrying a `Coord` field, encoding via
# `encode_fields` exactly as `TileTensor` does for its `Layout` (whose
# `_shape`/`_stride` are `Coord`s).
@fieldwise_init
struct CoordBox(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    var dims: Coord[Int64, Int64]
    var tag: Int64

    comptime device_type = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String("CoordBox")


# Launched single-threaded (grid=1/block=1), so the kernel writes
# unconditionally and needs no `global_idx`.
def _read_back_kernel(box: CoordBox, dst: Pointer[Int64, MutAnyOrigin]):
    dst[] = Int64(Int(box.dims[0].value()))
    dst[unsafe_offset=1] = Int64(Int(box.dims[1].value()))
    dst[unsafe_offset=2] = Int64(box.tag)


def test_coord_field_survives_device_launch() raises:
    with DeviceContext() as ctx:
        var box = CoordBox(
            dims=Coord[Int64, Int64](Int64(11), Int64(22)), tag=33
        )
        var out_dev = ctx.enqueue_create_buffer[.int64](3)
        ctx.enqueue_function[_read_back_kernel](
            box, out_dev.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        var out_host = ctx.enqueue_create_host_buffer[.int64](3)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        # The `Coord` field transferred intact through the launch boundary.
        assert_equal(Int(out_host[0]), 11)
        assert_equal(Int(out_host[1]), 22)
        assert_equal(Int(out_host[2]), 33)
        _ = out_dev^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
