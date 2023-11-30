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

# Simple program demonstrating a naive matrix multiplication in Python
from timeit import timeit

import check_mod

check_mod.install_if_missing("numpy")
import numpy as np


class PyMatrix:
    def __init__(self, value, rows, cols):
        self.value = value
        self.rows = rows
        self.cols = cols

    def __getitem__(self, idxs):
        return self.value[idxs[0]][idxs[1]]

    def __setitem__(self, idxs, value):
        self.value[idxs[0]][idxs[1]] = value


def matmul_python(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


def benchmark_matmul_python(M, N, K):
    A = PyMatrix(list(np.random.rand(M, K)), M, K)
    B = PyMatrix(list(np.random.rand(K, N)), K, N)
    C = PyMatrix(list(np.zeros((M, N))), M, N)
    secs = timeit(lambda: matmul_python(C, A, B), number=2) / 2
    gflops = ((2 * M * N * K) / secs) / 1e9
    return gflops


def benchmark_matmul_numpy(M, N, K):
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    secs = timeit(lambda: A @ B, number=10) / 10
    gflops = ((2 * M * N * K) / secs) / 1e9
    return gflops


if __name__ == "__main__":
    print("Throughput of a 128x128 matrix multiplication in Python:")
    print(benchmark_matmul_python(128, 128, 128))
    print("Throughput of a 128x128 matrix multiplication in Python numpy:")
    print(benchmark_matmul_numpy(128, 128, 128))
