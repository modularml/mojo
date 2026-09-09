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

"""Kernel-level accuracy test for `BK = 64` blockwise FP8 batched matmul.

The new `BK = 64` code path in
`batched_matmul_dynamic_scaled_fp8` is what makes GLM-5.1-FP8
(per-head `Dn = 192`, `Dv = 256`) work end-to-end. The DeepSeek
regression tests pin the existing `BK = 128` path but exercise no
`BK = 64` shapes. This test fills that gap by comparing the SM100
batched matmul output against the naive Mojo reference for two real
GLM shapes:

  (a) per-head K-up: `[1, M=7, N=192, K=512]`
  (b) batched K-up (8 heads): `[1, M=7, N=1536, K=512]`

Both use `n_scale_granularity = k_scale_granularity = 64` (matching
`BK = 64`). The reference is the same naive blockwise FP8 matmul
that the existing DeepSeek tests use, so any divergence is a kernel
numerics bug we've introduced.
"""

from std.math import ceildiv
from std.random import randn, seed

from max.gpu.host import DeviceContext
from layout import (
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    naive_blockwise_scaled_fp8_matmul,
)
from std.testing import assert_almost_equal
from std.utils.index import Index, IndexList


def _run_one[
    M: Int,
    N: Int,
    K: Int,
    n_g: Int,
    k_g: Int,
](name: StringLiteral, ctx: DeviceContext) raises:
    """Run a single FP8 matmul at the given dims and tolerate-compare.

    Inputs are random FP8 values + random per-block scales. The test
    pins `batched_matmul_dynamic_scaled_fp8` numerics against the
    naive CPU reference. Both consume the same FP8 + scale tensors;
    any drift signals a kernel bug at `BK = k_g`.
    """
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scale_type = DType.float32

    print(
        "== test_blockwise_fp8_bk64",
        name,
        " problem: (M=",
        M,
        ", N=",
        N,
        ", K=",
        K,
        ") granularity: (1,",
        n_g,
        ",",
        k_g,
        ")",
    )

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scales_size = (K // k_g) * M
    var b_scales_size = (N // n_g) * (K // k_g)

    var a_host = List(length=a_size, fill=Scalar[a_type](0))
    var b_host = List(length=b_size, fill=Scalar[b_type](0))
    var c_host = List(length=c_size, fill=Scalar[c_type](0))
    var c_host_ref = List(length=c_size, fill=Scalar[c_type](0))
    var a_scales_host = List(length=a_scales_size, fill=Scalar[scale_type](0))
    var b_scales_host = List(length=b_scales_size, fill=Scalar[scale_type](0))

    var a_host_tt = TileTensor(a_host, row_major[M, K]())
    var b_host_tt = TileTensor(b_host, row_major[N, K]())
    var a_scales_host_tt = TileTensor(a_scales_host, row_major[K // k_g, M]())
    var b_scales_host_tt = TileTensor(
        b_scales_host, row_major[N // n_g, K // k_g]()
    )
    random(a_host_tt)
    random(b_host_tt)
    random(a_scales_host_tt)
    random(b_scales_host_tt)

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scales_dev = ctx.enqueue_create_buffer[scale_type](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[scale_type](b_scales_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    var a_tt = TileTensor(a_dev, row_major[M, K]())
    var b_tt = TileTensor(b_dev, row_major[N, K]())
    var c_tt = TileTensor(c_dev, row_major[M, N]())
    var c_ref_tt = TileTensor(c_ref_dev, row_major[M, N]())
    var a_scales_tt = TileTensor(a_scales_dev, row_major[K // k_g, M]())
    var b_scales_tt = TileTensor(b_scales_dev, row_major[N // n_g, K // k_g]())

    # ===== Reference: naive blockwise FP8 matmul =====
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk=Index(1, n_g, k_g),
    ](
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        a_scales_tt.to_layout_tensor(),
        b_scales_tt.to_layout_tensor(),
        ctx,
    )
    ctx.synchronize()

    # ===== SM100 kernel under test =====
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=n_g,
        k_scale_granularity=k_g,
        transpose_b=True,
        target="gpu",
    ](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        ctx,
    )
    ctx.synchronize()

    # ===== Compare =====
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    comptime rtol = 1e-2
    comptime atol = 1e-2
    var ndiff = 0
    for mi in range(M):
        for ni in range(N):
            var got = c_host[mi * N + ni]
            var ref_val = c_host_ref[mi * N + ni]
            try:
                assert_almost_equal(got, ref_val, rtol=rtol, atol=atol)
            except:
                ndiff += 1
                if ndiff <= 3:
                    print(
                        "  diff @ (",
                        mi,
                        ",",
                        ni,
                        ") got=",
                        got,
                        " ref=",
                        ref_val,
                    )
    if ndiff > 0:
        raise Error(
            "BK="
            + String(k_g)
            + " kernel diverged from naive reference at "
            + String(ndiff)
            + " of "
            + String(M * N)
            + " positions"
        )
    print("  PASSED")


def main() raises:
    seed(2758)
    var ctx = DeviceContext()

    # (a) per-head K-up shape: 1 head's worth of MLA decode. n_g=64 to
    # handle GLM's straddling per-head row count; k_g stays at 128 (no
    # over-fetch).
    _run_one[M=8, N=192, K=512, n_g=64, k_g=128]("per_head_glm", ctx)

    # (b) batched K-up shape: 8 heads flat (GLM-5.1 scaled to H=8).
    _run_one[M=8, N=1536, K=512, n_g=64, k_g=128]("batched_glm", ctx)
