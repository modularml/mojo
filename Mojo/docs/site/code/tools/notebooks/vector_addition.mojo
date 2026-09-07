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

# Tested on T4 GPU 2 Dec 2025


from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.sys import has_accelerator

comptime VECTOR_WIDTH = 10
comptime layout = row_major[VECTOR_WIDTH]()
comptime active_dtype = DType.uint8


# Elementwise vector addition on GPU threads
def vector_addition(
    left: TileTensor[active_dtype, type_of(layout), MutAnyOrigin],
    right: TileTensor[active_dtype, type_of(layout), MutAnyOrigin],
    output: TileTensor[active_dtype, type_of(layout), MutAnyOrigin],
):
    var idx = thread_idx.x
    output[idx] = left[idx] + right[idx]


def main() raises:
    # Ensure a supported GPU (NVIDIA or AMD) is available
    comptime assert has_accelerator(), "This example requires a supported GPU"

    # Create GPU device context
    var ctx = DeviceContext()

    # Allocate buffers and tensors for left and right operands, and output
    var left_buffer = ctx.enqueue_create_buffer[active_dtype](VECTOR_WIDTH)
    var left_tensor = TileTensor(left_buffer, layout)

    var right_buffer = ctx.enqueue_create_buffer[active_dtype](VECTOR_WIDTH)
    var right_tensor = TileTensor(right_buffer, layout)

    var output_buffer = ctx.enqueue_create_buffer[active_dtype](VECTOR_WIDTH)
    var output_tensor = TileTensor(output_buffer, layout)

    # Initialize input buffers with sample data
    var message_bytes: List[UInt8] = [
        71,
        100,
        107,
        107,
        110,
        31,
        76,
        110,
        105,
        110,
    ]
    with left_buffer.map_to_host() as mapped_buffer:
        var mapped_tensor = TileTensor(mapped_buffer, layout)
        for idx in range(VECTOR_WIDTH):
            mapped_tensor[idx] = message_bytes[idx]
    _ = right_buffer.enqueue_fill(1)

    # Launch GPU kernel
    ctx.enqueue_function[vector_addition](
        left_tensor,
        right_tensor,
        output_tensor,
        grid_dim=1,
        block_dim=VECTOR_WIDTH,
    )
    ctx.synchronize()

    # Read results back and print as ASCII
    with output_buffer.map_to_host() as mapped_buffer:
        var mapped_tensor = TileTensor(mapped_buffer, layout)
        for idx in range(VECTOR_WIDTH):
            print(chr(Int(mapped_tensor[idx])), end="")
        print()
