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

"""Figure 19.4: Forward propagation implementation for batched convolutional
layer.

Based on Programming Massively Parallel Processors Chapter 19.

This is the CPU reference implementation for a batched convolution.
X: input feature maps [N, C, H, W]
F: filter weights [M, C, K, K]
Y: output feature maps [N, M, H_out, W_out]
"""


def conv_layer_batched(
    N: Int,
    M: Int,
    C: Int,
    H: Int,
    W: Int,
    K: Int,
    X: UnsafePointer[mut=False, Float32, _],
    F: UnsafePointer[mut=False, Float32, _],
    Y: UnsafePointer[mut=True, Float32, _],
):
    """Batched convolution (Figure 19.4).

    Performs a basic batched convolution with no padding and stride=1.

    Args:
        N: Batch size.
        M: Output channels.
        C: Input channels.
        H: Input height.
        W: Input width.
        K: Kernel size (K x K).
        X: Input tensor pointer [N, C, H, W].
        F: Filter tensor pointer [M, C, K, K].
        Y: Output tensor pointer [N, M, H_out, W_out].
    """
    var H_out = H - K + 1
    var W_out = W - K + 1

    for n in range(N):  # for each sample in the batch
        for m in range(M):  # for each output feature map
            for h in range(H_out):  # for each output element
                for w in range(W_out):
                    var out_idx = (
                        n * M * H_out * W_out
                        + m * H_out * W_out
                        + h * W_out
                        + w
                    )
                    Y[out_idx] = 0.0

                    for c in range(C):  # sum over all input feature maps
                        for p in range(K):  # K x K filter
                            for q in range(K):
                                var x_idx = (
                                    n * C * H * W
                                    + c * H * W
                                    + (h + p) * W
                                    + (w + q)
                                )
                                var f_idx = (
                                    m * C * K * K + c * K * K + p * K + q
                                )
                                Y[out_idx] += X[x_idx] * F[f_idx]


def main():
    print("Figure 19.4: Forward propagation for batched convolution")

    # Test parameters
    var N = 2  # batch size
    var M = 2  # output channels
    var C = 1  # input channels
    var H = 5  # height
    var W = 5  # width
    var K = 3  # kernel size

    var H_out = H - K + 1
    var W_out = W - K + 1

    # Allocate memory
    var X = alloc[Float32](N * C * H * W)
    var F = alloc[Float32](M * C * K * K)
    var Y = alloc[Float32](N * M * H_out * W_out)

    # Initialize with simple values (all 1.0)
    for i in range(N * C * H * W):
        X[i] = 1.0

    for i in range(M * C * K * K):
        F[i] = 1.0

    # Initialize output to 0
    for i in range(N * M * H_out * W_out):
        Y[i] = 0.0

    # Test batched convolution
    print("Testing batched convolution (Figure 19.4)...")
    conv_layer_batched(N, M, C, H, W, K, X, F, Y)

    print("Input shape: [", N, ",", C, ",", H, ",", W, "]")
    print("Filter shape: [", M, ",", C, ",", K, ",", K, "]")
    print("Output shape: [", N, ",", M, ",", H_out, ",", W_out, "]")

    print("\nSample output values:")
    for n in range(N):
        for m in range(M):
            print("Batch", n, ", Channel", m, ":", end=" ")
            var base = n * M * H_out * W_out + m * H_out * W_out
            var count = 0
            for i in range(H_out * W_out):
                if count < 3:
                    print(Y[base + i], end=" ")
                    count += 1
            print("...")

    # Verify: With all 1s input and filter, each output element should be C*K*K
    var expected = Float32(C * K * K)  # 1 * 3 * 3 = 9.0
    var passed = True
    for i in range(N * M * H_out * W_out):
        if Y[i] != expected:
            print("Mismatch at index", i, ": expected", expected, ", got", Y[i])
            passed = False
            break

    if passed:
        print("\nFigure 19.4 Test Passed!")
    else:
        print("\nFigure 19.4 Test Failed!")

    # Cleanup
    X.free()
    F.free()
    Y.free()
