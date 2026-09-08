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
# RUN: %parse-mojo-isolated -verify-diagnostics %s


def test1():
    # expected-error @below {{invalid MLIR attribute: `#kgen.deferred` can only be used for non-typed attributes}}
    # expected-note @below {{attempting to parse: '#kgen.deferred 0 : index'}}
    _ = __mlir_attr.`#kgen.deferred 0 : index`


struct DType(Movable where False):
    comptime type = __mlir_type.`!kgen.dtype`
    var value: Self.type


def test3(x: __mlir_type.`!kgen.struct<(!kgen.scalar<f32>)>`):
    # expected-error @+1 {{element index 1 out of bounds (>=1)}}
    _ = __mlir_op.`kgen.struct.extract`[_type = __mlir_type.`!kgen.scalar<f32>`, index=__mlir_attr.`1:index`](x)
