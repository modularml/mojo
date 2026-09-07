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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that a non-literal float exponent of a different dtype is rejected,
# so `SIMD.__pow__` never narrows a runtime exponent to the base dtype. Only
# float literals are converted, via the `FloatLiteral` overload.


# CHECK: constraint failed: unsupported type combination
def main():
    _ = Float32(4.0) ** Float64(0.5)
