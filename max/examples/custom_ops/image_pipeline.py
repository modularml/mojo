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

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from PIL import Image


def main() -> None:
    device = CPU() if accelerator_count() == 0 else Accelerator()
    img = np.array(Image.open(Path(__file__).parent / "dogs.jpg"))

    color_tensor_type = TensorType(
        DType.uint8,
        shape=img.shape,
        device=DeviceRef.from_device(device),
    )

    gray_tensor_type = TensorType(
        DType.uint8,
        shape=[img.shape[0], img.shape[1]],
        device=DeviceRef.from_device(device),
    )

    graph = Graph(
        "image_pipeline",
        input_types=[color_tensor_type],
        custom_extensions=[Path(__file__).parent / "kernels"],
    )

    def grayscale(x: TensorValue) -> TensorValue:
        return ops.custom(
            name="grayscale",
            device=DeviceRef.from_device(device),
            values=[x],
            out_types=[gray_tensor_type],
        )[0].tensor

    def brightness(x: TensorValue, brightness: float) -> TensorValue:
        return ops.custom(
            name="brightness",
            device=DeviceRef.from_device(device),
            values=[
                x,
                ops.constant(brightness, DType.float32, DeviceRef.CPU()),
            ],
            out_types=[x.type],
        )[0].tensor

    def blur(x: TensorValue, blur_size: int) -> TensorValue:
        return ops.custom(
            name="blur",
            device=DeviceRef.from_device(device),
            values=[
                x,
                ops.constant(blur_size, DType.int64, DeviceRef.CPU()),
            ],
            out_types=[x.type],
        )[0].tensor

    with graph:
        grayed = grayscale(graph.inputs[0].tensor)
        brightened = brightness(grayed, brightness=1.5)
        blurred = blur(brightened, blur_size=8)
        graph.output(blurred)

    session = InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)

    img_dev = Buffer.from_numpy(img).to(device)

    result = model.execute(img_dev)[0]
    assert isinstance(result, Buffer)
    result = result.to(CPU())

    Image.fromarray(result.to_numpy()).save("dogs_out.jpg")


if __name__ == "__main__":
    main()
