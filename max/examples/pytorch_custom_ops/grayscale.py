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

# DOC: max/develop/custom-kernels-pytorch.mdx

from pathlib import Path

import numpy as np
import torch
from max.experimental.torch import CustomOpLibrary
from PIL import Image

# Load the Mojo custom operations from the `operations` directory.
mojo_kernels = Path(__file__).parent / "operations"
ops = CustomOpLibrary(mojo_kernels)


@torch.compile
def grayscale(pic: torch.Tensor):  # noqa: ANN201
    output = pic.new_empty(pic.shape[:-1])  # Remove color channel dimension
    ops.grayscale(output, pic)  # Call our custom Mojo operation
    return output


def create_test_image():  # noqa: ANN201
    # Create a synthetic RGB test image (64x64 pixels)
    # Create a simple gradient pattern
    test_array = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            test_array[i, j] = [i * 4, j * 4, (i + j) * 2]

    return Image.fromarray(test_array, mode="RGB")


def main() -> None:
    # Create test image
    image = create_test_image()
    # Convert to numpy array and then to PyTorch tensor
    image_array = np.array(image)

    # Convert to PyTorch tensor (CPU only for compatibility)
    image_tensor = torch.from_numpy(image_array)

    # Apply our custom grayscale operation
    gray_image_tensor = grayscale(image_tensor)

    # Convert back to PIL Image for potential saving/display
    gray_array = gray_image_tensor.numpy()
    result_image = Image.fromarray(gray_array, mode="L")

    # Save the result
    result_image.save("grayscale_output.png")
    print("Grayscale image saved as 'grayscale_output.png'")


if __name__ == "__main__":
    main()
