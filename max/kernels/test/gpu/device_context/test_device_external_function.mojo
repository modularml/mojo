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
"""Test loading and executing an external cubin binary."""

from max.gpu.host import (
    DeviceContext,
)
from max.gpu.host import DeviceGraph, DeviceGraphBuilder
from max.gpu.host.device_context import DeviceExternalFunction
from std.os import getenv
from std.testing import assert_equal


def test_external_cubin_vec_add(ctx: DeviceContext) raises:
    """Test loading and executing an external cubin for vector addition."""
    var cubin_data: List[UInt8]
    with open(getenv("CUBIN_PATH"), "r") as file:
        cubin_data = file.read_bytes()

    var external_func = DeviceExternalFunction(
        ctx,
        function_name="vec_add",  # matches extern "C" name
        # DeviceExternalFunction takes a StringSlice, which is probably wrong.
        # The cubin is [very, very likely] invalid UTF8.
        asm=String(StringSlice(unsafe_from_utf8=cubin_data)),
    )

    comptime length = 1024
    var block_dim = 32
    var grid_dim = length // block_dim

    var in0 = ctx.enqueue_create_buffer[.float32](length)
    var in1 = ctx.enqueue_create_buffer[.float32](length)
    var out = ctx.enqueue_create_buffer[.float32](length)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = 1

    ctx.enqueue_function(
        external_func,
        in0,
        in1,
        out,
        length,
        grid_dim=(grid_dim,),
        block_dim=(block_dim,),
    )

    ctx.synchronize()

    with out.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(i + 1))


def test_external_cubin_vec_add_graph(ctx: DeviceContext) raises:
    """Test adding an external cubin function as a device graph node."""
    var cubin_data: List[UInt8]
    with open(getenv("CUBIN_PATH"), "r") as file:
        cubin_data = file.read_bytes()

    var external_func = DeviceExternalFunction(
        ctx,
        function_name="vec_add",  # matches extern "C" name
        # DeviceExternalFunction takes a StringSlice, which is probably wrong.
        # The cubin is [very, very likely] invalid UTF8.
        asm=String(StringSlice(unsafe_from_utf8=cubin_data)),
    )

    comptime length = 1024
    var block_dim = 32
    var grid_dim = length // block_dim

    var in0 = ctx.enqueue_create_buffer[.float32](length)
    var in1 = ctx.enqueue_create_buffer[.float32](length)
    var out = ctx.enqueue_create_buffer[.float32](length)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = 2

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            external_func,
            in0,
            in1,
            out,
            length,
            grid_dim=(grid_dim,),
            block_dim=(block_dim,),
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with out.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(i + 2))


def main() raises:
    with DeviceContext() as ctx:
        test_external_cubin_vec_add(ctx)
        test_external_cubin_vec_add_graph(ctx)
