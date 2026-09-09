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

# RUN: %mojo %s | FileCheck %s

# Folding a `contains` / `all_satisfies` chain over a variadic trait pack
# reaches the same conformance query once per prefix at each step. Re-deriving
# those repeats instead of resolving each once makes the work grow as 2^N.
# This test guards that by wall clock: 32 elements compiles in about a second
# and otherwise exceeds the test timeout.

comptime T = Copyable & Deinitable


@fieldwise_init
struct C[i: Int](T & ImplicitlyCopyable):
    var v: Int


struct M[*ts: T]:
    comptime _Has[t: T] = Self.ts.contains[t]()
    comptime _HasAll[*qs: T] = qs.all[Self._Has]()

    @staticmethod
    def check[*qs: T]():
        comptime assert Self._HasAll[*qs], "missing"


def main():
    M[
        C[0],
        C[1],
        C[2],
        C[3],
        C[4],
        C[5],
        C[6],
        C[7],
        C[8],
        C[9],
        C[10],
        C[11],
        C[12],
        C[13],
        C[14],
        C[15],
        C[16],
        C[17],
        C[18],
        C[19],
        C[20],
        C[21],
        C[22],
        C[23],
        C[24],
        C[25],
        C[26],
        C[27],
        C[28],
        C[29],
        C[30],
        C[31],
    ].check[C[0]]()
    # CHECK: ok
    print("ok")
