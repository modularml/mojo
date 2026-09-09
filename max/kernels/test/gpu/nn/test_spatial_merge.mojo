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
from layout import Idx, TileTensor, row_major
from nn.spatial_merge import spatial_merge
from std.testing import assert_equal


def test_spatial_merge(ctx: DeviceContext) raises:
    comptime dtype = DType.float32
    comptime merge_size = 2
    comptime hidden_size = 4
    comptime batch_size = 2

    # Batch item 0: t=1, h=4, w=4
    # Batch item 1: t=2, h=2, w=2
    var grid_thw_list: List[Int64] = [1, 4, 4, 2, 2, 2]

    var total_input_patches = 20
    var total_output_patches = 6
    var C_out = hidden_size * merge_size * merge_size

    var input_size = total_input_patches * hidden_size
    var output_size = total_output_patches * C_out
    var grid_thw_size = batch_size * 3

    # Create device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](input_size)
    var output_device = ctx.enqueue_create_buffer[dtype](output_size)
    var grid_thw_device = ctx.enqueue_create_buffer[.int64](grid_thw_size)

    # Initialize input data on host
    with input_device.map_to_host() as input_host:
        var input_host_tensor = TileTensor(
            input_host,
            row_major(total_input_patches, hidden_size),
        )
        for patch_idx in range(total_input_patches):
            for feat_idx in range(hidden_size):
                input_host_tensor[patch_idx, feat_idx] = Float32(
                    patch_idx * 100 + feat_idx
                )

    # Initialize grid_thw data on host
    with grid_thw_device.map_to_host() as grid_thw_host:
        var grid_thw_host_tensor = TileTensor(
            grid_thw_host,
            row_major[batch_size, 3](),
        )
        for i in range(batch_size):
            for j in range(3):
                grid_thw_host_tensor[i, j] = grid_thw_list[i * 3 + j]

    # Create TileTensors for GPU operations
    var input_tensor = TileTensor(
        input_device,
        row_major(total_input_patches, hidden_size),
    )
    var output_tensor = TileTensor(
        output_device,
        row_major(total_output_patches, C_out),
    )
    var grid_thw_tensor = TileTensor(
        grid_thw_device,
        row_major[batch_size, 3](),
    )

    spatial_merge[dtype](
        output_tensor,
        input_tensor,
        grid_thw_tensor,
        hidden_size,
        merge_size,
        ctx,
    )

    ctx.synchronize()

    # Verify batch 0: t=1, h=4, w=4, merge_size=2
    # Output has 1 × 2 × 2 = 4 merged patches.
    # Each merged patch contains 2×2 spatial positions.

    # Expected mapping for batch 0:
    # Output patch 0 (ho=0, wo=0) contains input patches: [0,1,4,5]
    # Output patch 1 (ho=0, wo=1) contains input patches: [2,3,6,7]
    # Output patch 2 (ho=1, wo=0) contains input patches: [8,9,12,13]
    # Output patch 3 (ho=1, wo=1) contains input patches: [10,11,14,15]

    var batch0_expected_list: List[Int] = [
        0,
        1,
        4,
        5,
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        13,
        10,
        11,
        14,
        15,
    ]
    var batch0_expected: Pointer[
        batch0_expected_list.T, origin_of(batch0_expected_list)
    ] = batch0_expected_list.unsafe_ptr()

    # Verify results
    with output_device.map_to_host() as output_host:
        # Verify batch 0 (4 output patches × 4 spatial positions × 4 features).
        for out_patch in range(4):
            for spatial_pos in range(4):
                var expected_input_patch = batch0_expected[
                    out_patch * 4 + spatial_pos
                ]
                for feat in range(hidden_size):
                    var output_idx = (
                        out_patch * C_out + spatial_pos * hidden_size + feat
                    )
                    var actual_val = output_host[output_idx]
                    var expected_val = Float32(
                        expected_input_patch * 100 + feat
                    )
                    assert_equal(
                        actual_val,
                        expected_val,
                        "Batch 0: patch="
                        + String(out_patch)
                        + " pos="
                        + String(spatial_pos)
                        + " feat="
                        + String(feat),
                    )

        # Verify batch 1: t=2, h=2, w=2, merge_size=2
        # Output has 2 × 1 × 1 = 2 merged patches (one per frame).
        # Each merged patch contains 2×2 spatial positions (entire spatial grid).
        # Input patches are 16, 17, 18, 19.

        var batch1_expected_list: List[Int] = [
            16,
            17,
            18,
            19,
            16,
            17,
            18,
            19,  # Frame 0 and Frame 1 (repeated)
        ]
        var batch1_expected: Pointer[
            batch1_expected_list.T, origin_of(batch1_expected_list)
        ] = batch1_expected_list.unsafe_ptr()

        var batch1_start = 4 * C_out  # Batch 0 had 4 output patches

        # Verify batch 1 (2 output patches × 4 spatial positions × 4 features)
        for out_patch in range(2):
            for spatial_pos in range(4):
                var expected_input_patch = batch1_expected[
                    out_patch * 4 + spatial_pos
                ]
                for feat in range(hidden_size):
                    var output_idx = (
                        batch1_start
                        + out_patch * C_out
                        + spatial_pos * hidden_size
                        + feat
                    )
                    var actual_val = output_host[output_idx]
                    var expected_val = Float32(
                        expected_input_patch * 100 + feat
                    )
                    assert_equal(
                        actual_val,
                        expected_val,
                        "Batch 1: patch="
                        + String(out_patch)
                        + " pos="
                        + String(spatial_pos)
                        + " feat="
                        + String(feat),
                    )


def test_spatial_merge_no_out_of_bounds(ctx: DeviceContext) raises:
    # spatial_merge must launch one GPU block per merged output patch. The output
    # uses the [total_patches, hidden_size] shape, with a sentinel-filled canary
    # after it: a block that writes past the output overwrites the canary, and a
    # merged patch left unwritten stays sentinel.
    comptime dtype = DType.float32
    comptime merge_size = 2
    comptime hidden_size = 4
    comptime batch_size = 1

    # t=1, h=2, w=4: 8 input patches, 2 merged patches (width spans 2 blocks).
    var grid_thw_list: List[Int64] = [1, 2, 4]
    var total_input_patches = 8
    var num_merged_patches = 2
    var C_out = hidden_size * merge_size * merge_size

    var input_size = total_input_patches * hidden_size
    # Output uses the production shape [total_patches, hidden_size].
    var output_rows = total_input_patches
    var output_size = output_rows * hidden_size  # == num_merged_patches * C_out
    var canary_size = output_size
    var grid_thw_size = batch_size * 3

    var sentinel = Float32(-99999.0)

    var input_device = ctx.enqueue_create_buffer[dtype](input_size)
    var output_device = ctx.enqueue_create_buffer[dtype](
        output_size + canary_size
    )
    var grid_thw_device = ctx.enqueue_create_buffer[.int64](grid_thw_size)

    with input_device.map_to_host() as input_host:
        var input_host_tensor = TileTensor(
            input_host,
            row_major(total_input_patches, hidden_size),
        )
        for patch_idx in range(total_input_patches):
            for feat_idx in range(hidden_size):
                input_host_tensor[patch_idx, feat_idx] = Float32(
                    patch_idx * 100 + feat_idx
                )

    # Sentinel-fill output and canary so any stray or missing write is visible.
    with output_device.map_to_host() as output_host:
        for i in range(output_size + canary_size):
            output_host[i] = sentinel

    with grid_thw_device.map_to_host() as grid_thw_host:
        var grid_thw_host_tensor = TileTensor(
            grid_thw_host,
            row_major[batch_size, 3](),
        )
        for i in range(batch_size):
            for j in range(3):
                grid_thw_host_tensor[i, j] = grid_thw_list[i * 3 + j]

    var input_tensor = TileTensor(
        input_device,
        row_major(total_input_patches, hidden_size),
    )
    var output_tensor = TileTensor(
        output_device,
        row_major(output_rows, hidden_size),
    )
    var grid_thw_tensor = TileTensor(
        grid_thw_device,
        row_major[batch_size, 3](),
    )

    spatial_merge[dtype](
        output_tensor,
        input_tensor,
        grid_thw_tensor,
        hidden_size,
        merge_size,
        ctx,
    )

    ctx.synchronize()

    # Expected merge of grid [1, 2, 4]:
    # merged patch 0 (wo=0) contains input patches [0, 1, 4, 5]
    # merged patch 1 (wo=1) contains input patches [2, 3, 6, 7]
    var expected_list: List[Int] = [0, 1, 4, 5, 2, 3, 6, 7]
    var expected: Pointer[
        expected_list.T, origin_of(expected_list)
    ] = expected_list.unsafe_ptr()

    with output_device.map_to_host() as output_host:
        for patch in range(num_merged_patches):
            for pos in range(merge_size * merge_size):
                var expected_input_patch = expected[patch * 4 + pos]
                for feat in range(hidden_size):
                    var idx = patch * C_out + pos * hidden_size + feat
                    assert_equal(
                        output_host[idx],
                        Float32(expected_input_patch * 100 + feat),
                        "merged patch="
                        + String(patch)
                        + " pos="
                        + String(pos)
                        + " feat="
                        + String(feat),
                    )

        # A correct kernel never writes into the canary region.
        for i in range(output_size, output_size + canary_size):
            assert_equal(
                output_host[i],
                sentinel,
                "spatial_merge wrote out of bounds at index " + String(i),
            )


def main() raises:
    with DeviceContext() as ctx:
        test_spatial_merge(ctx)
        test_spatial_merge_no_out_of_bounds(ctx)
