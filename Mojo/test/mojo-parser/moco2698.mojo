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

# RUN: %parse-mojo-isolated --verify-diagnostics %s

comptime float = __mlir_type.`!kgen.scalar<f64>`

# expected-error @below {{'def' statement must be on its own line}}
# expected-warning @below {{transfer from a value of trivial register type 'float' has no effect and can be removed}}
# expected-warning @below {{'float' value is unused; assign to '_' to discard the result}}
# expected-note @below {{'float' is aka '__mlir_type.`!kgen.scalar<f64>`'}}
struct a(Movable where False): def b(c, d : float) : d^
