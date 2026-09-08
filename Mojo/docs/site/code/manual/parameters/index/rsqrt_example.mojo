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

# start-rsqrt-example
from std.math import sqrt


def rsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return 1 / sqrt(x)


def main() raises:
    var v = Scalar[.float16](42)
    print(rsqrt(v))


# end-rsqrt-example
