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
#
# Regression: `--debug-level=full` crash on Pointer[Scalar[dtype]].
#
# RUN: %mojo -debug-level full %s | FileCheck %s

from std.memory import Pointer


def load_generic[
    dtype: DType
](p: Pointer[mut=False, Scalar[dtype], ...]) -> Scalar[dtype]:
    var val = p[]
    return val


def main():
    var x: Float32 = 3.5
    # CHECK: 3.5
    print(load_generic(Pointer(to=x)))
