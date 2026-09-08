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

from std.random import seed

from layout import TileTensor, row_major
from nn.randn import random_normal


def test_random_normal():
    seed(0)

    comptime out_shape = row_major[2, 2]()
    var output_stack = Array[Float32, 4](fill=0)
    var output = TileTensor(output_stack, out_shape)

    random_normal[.float32, 0.0, 1.0](output)
    # CHECK-LABEL: == test_random_normal
    print("== test_random_normal")


def main():
    test_random_normal()
