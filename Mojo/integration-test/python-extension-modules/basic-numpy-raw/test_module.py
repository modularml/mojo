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

# Imports from 'mojo_module.so'
import mojo_module  # type: ignore[import-not-found]
import numpy as np


def test_mojo_numpy() -> None:
    print("Hello from Basic Numpy Example!")

    enumerated = np.empty((5, 5), dtype=np.int32)
    for i, j in np.ndindex(enumerated.shape):
        enumerated[i, j] = 10 * i + j

    print(f"The original array has contents: \n{enumerated}")

    expected = np.array(
        [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
        ]
    )
    assert np.array_equal(enumerated, expected)

    mojo_module.mojo_incr_np_array(enumerated)

    print(f"The altered array has contents: \n{enumerated}")

    expected = np.array(
        [
            [1, 2, 3, 4, 5],
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
            [31, 32, 33, 34, 35],
            [41, 42, 43, 44, 45],
        ]
    )
    assert np.array_equal(enumerated, expected)

    print("🎉🎉🎉 Mission Success! 🎉🎉🎉")
