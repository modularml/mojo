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
# RUN: %mojo %s
# Test for https://github.com/modularml/mojo/issues/1505

from random import random_ui64

from testing import assert_equal

from utils import IndexList


fn gen_perm() -> IndexList[64]:
    var result = IndexList[64]()

    for i in range(64):
        result[i] = 64 - i - 1
    return result


def main():
    alias p = gen_perm()

    # generate random data to prevent that everything gets simplified
    var data1 = SIMD[DType.uint8, 64]()
    for i in range(64):
        data1[i] = random_ui64(0, 100).cast[DType.uint8]()

    var data2 = data1.shuffle[
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        p[10],
        p[11],
        p[12],
        p[13],
        p[14],
        p[15],
        p[16],
        p[17],
        p[18],
        p[19],
        p[20],
        p[21],
        p[22],
        p[23],
        p[24],
        p[25],
        p[26],
        p[27],
        p[28],
        p[29],
        p[30],
        p[31],
        p[32],
        p[33],
        p[34],
        p[35],
        p[36],
        p[37],
        p[38],
        p[39],
        p[40],
        p[41],
        p[42],
        p[43],
        p[44],
        p[45],
        p[46],
        p[47],
        p[48],
        p[49],
        p[50],
        p[51],
        p[52],
        p[53],
        p[54],
        p[55],
        p[56],
        p[57],
        p[58],
        p[59],
        p[60],
        p[61],
        p[62],
        p[63],
    ]()

    for i in range(64):
        assert_equal(data1[p[i]], data2[i])
