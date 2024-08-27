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


fn _minmax[
    type: DType, //
](x: UnsafePointer[Scalar[type]], N: Int) -> Tuple[Scalar[type], Scalar[type]]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return (min_val, max_val)


fn compare[
    dtype: DType, verbose: Bool = True
](
    x: UnsafePointer[Scalar[dtype]],
    y: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
    *,
    msg: String = "",
) -> SIMD[dtype, 4]:
    var atol = UnsafePointer[Scalar[dtype]].alloc(num_elements)
    var rtol = UnsafePointer[Scalar[dtype]].alloc(num_elements)

    for i in range(num_elements):
        var d = abs(x[i] - y[i])
        var e = abs(d / y[i])
        atol[i] = d
        rtol[i] = e

    var atol_minmax = _minmax(atol, num_elements)
    var rtol_minmax = _minmax(rtol, num_elements)
    if verbose:
        if msg:
            print(msg)
        print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
        print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
        print("==========================================================")
    atol.free()
    rtol.free()
    return SIMD[dtype, 4](
        atol_minmax[0], atol_minmax[1], rtol_minmax[0], rtol_minmax[1]
    )
