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

from pathlib import Path

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def draw_mandelbrot(
    tensor: Buffer, width: int, height: int, iterations: int
) -> None:
    """A helper function to visualize the Mandelbrot set in ASCII art."""
    sr = "....,c8M@jawrpogOQEPGJ"
    for row in range(height):
        for col in range(width):
            v = tensor[row, col].item()
            if v < iterations:
                idx = int(v % len(sr))
                p = sr[idx]
                print(p, end="")
            else:
                print(" ", end="")
        print()


def create_mandelbrot_graph(
    width: int,
    height: int,
    min_x: float,
    min_y: float,
    scale_x: float,
    scale_y: float,
    max_iterations: int,
    device: DeviceRef,
) -> Graph:
    """Configure a graph to run a Mandelbrot kernel."""
    output_dtype = DType.int32
    mojo_kernels = Path(__file__).parent / "kernels"

    with Graph(
        "mandelbrot",
        custom_extensions=[mojo_kernels],
    ) as graph:
        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        result = ops.custom(
            name="mandelbrot",
            device=device,
            values=[
                ops.constant(
                    min_x, dtype=DType.float32, device=DeviceRef.CPU()
                ),
                ops.constant(
                    min_y, dtype=DType.float32, device=DeviceRef.CPU()
                ),
                ops.constant(
                    scale_x, dtype=DType.float32, device=DeviceRef.CPU()
                ),
                ops.constant(
                    scale_y, dtype=DType.float32, device=DeviceRef.CPU()
                ),
                ops.constant(
                    max_iterations, dtype=DType.int32, device=DeviceRef.CPU()
                ),
            ],
            out_types=[
                TensorType(
                    dtype=output_dtype, shape=[height, width], device=device
                )
            ],
        )[0].tensor

        # Return the result of the custom operation as the output of the graph.
        graph.output(result)
        return graph


if __name__ == "__main__":
    # Establish Mandelbrot set ranges.
    WIDTH = 60
    HEIGHT = 25
    MAX_ITERATIONS = 100
    MIN_X = -2.0
    MAX_X = 0.7
    MIN_Y = -1.12
    MAX_Y = 1.12

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Configure our simple graph.
    scale_x = (MAX_X - MIN_X) / WIDTH
    scale_y = (MAX_Y - MIN_Y) / HEIGHT
    graph = create_mandelbrot_graph(
        WIDTH,
        HEIGHT,
        MIN_X,
        MIN_Y,
        scale_x,
        scale_y,
        MAX_ITERATIONS,
        DeviceRef.from_device(device),
    )

    # Set up an inference session that runs the graph on a GPU, if available.
    session = InferenceSession(
        devices=[device],
    )
    # Compile the graph.
    compiled = session.compile(graph)
    model = session.init(compiled)

    # Perform the calculation on the target device.
    result = model.execute()[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Buffer)
    result = result.to(CPU())

    draw_mandelbrot(result, WIDTH, HEIGHT, MAX_ITERATIONS)
