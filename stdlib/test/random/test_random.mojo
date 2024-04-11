# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s | FileCheck %s

from random import randn_float64, random_float64, random_si64, random_ui64, seed


# CHECK-LABEL: test_random
fn test_random():
    print("== test_random")

    # CHECK-LABEL: random_float64 =
    print("random_float64 = ", random_float64(0, 1))

    # CHECK-LABEL: random_si64 =
    print("random_si64 = ", random_si64(-255, 255))

    # CHECK-LABEL: random_ui64 =
    print("random_ui64 = ", random_ui64(0, 255))

    # CHECK-LABEL: randn_float64 =
    print("randn_float64 = ", randn_float64(0, 1))


# CHECK-LABEL: test_seed
fn test_seed():
    print("== test_seed")

    seed(5)

    # CHECK: random_seed_float64 = [[FLOAT64:.*]]
    print("random_seed_float64 = ", random_float64(0, 1))

    # CHECK: random_seed_si64 = [[SI64:.*]]
    print("random_seed_si64 = ", random_si64(-255, 255))

    # CHECK: random_seed_ui64 = [[UI64:.*]]
    print("random_seed_ui64 = ", random_ui64(0, 255))

    seed(5)

    # CHECK: random_seed_float64 = [[FLOAT64]]
    print("random_seed_float64 = ", random_float64(0, 1))

    # CHECK: random_seed_si64 = [[SI64]]
    print("random_seed_si64 = ", random_si64(-255, 255))

    # CHECK: random_seed_ui64 = [[UI64]]
    print("random_seed_ui64 = ", random_ui64(0, 255))


fn main():
    test_random()
    test_seed()
