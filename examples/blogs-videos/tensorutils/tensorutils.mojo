# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

from tensor import Tensor
from math import trunc, mod


fn tensorprint[type: DType](t: Tensor[type]) -> None:
    let rank = t.rank()
    var dim0: Int = 0
    var dim1: Int = 0
    var dim2: Int = 0
    if rank == 0 or rank > 3:
        print("Error: Tensor rank should be: 1,2, or 3. Tensor rank is ", rank)
        return
    if rank == 1:
        dim0 = 1
        dim1 = 1
        dim2 = t.dim(0)
    if rank == 2:
        dim0 = 1
        dim1 = t.dim(0)
        dim2 = t.dim(1)
    if rank == 3:
        dim0 = t.dim(0)
        dim1 = t.dim(1)
        dim2 = t.dim(2)
    var val: SIMD[type, 1] = 0.0
    for i in range(dim0):
        if i == 0 and rank == 3:
            print("[")
        else:
            if i > 0:
                print()
        for j in range(dim1):
            if rank != 1:
                if j == 0:
                    print_no_newline("  [")
                else:
                    print_no_newline("\n   ")
            print_no_newline("[")
            for k in range(dim2):
                if rank == 1:
                    val = t[k]
                if rank == 2:
                    val = t[j, k]
                if rank == 3:
                    val = t[i, j, k]
                let int_str = String(trunc(val).cast[DType.int32]())
                let float_str = String(mod(val, 1))
                let s = int_str + "." + float_str[2:6]
                if k == 0:
                    print_no_newline(s)
                else:
                    print_no_newline("  ", s)
            print_no_newline("]")
        if rank > 1:
            print_no_newline("]")
        print()
    if rank == 3:
        print("]")
    print(
        "Tensor shape:",
        t.shape().__str__(),
        ", Tensor rank:",
        rank,
        ",",
        "DType:",
        type.__str__(),
    )
    print()
