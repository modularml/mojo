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
"""Test calling FlashInfer FP4 GEMM from a MAX custom op.

Tests both that the MAX custom op integration works and that the numerical
output is correct."""

import os
from pathlib import Path

import torch
import tvm_ffi
from flashinfer.aot import gen_gemm_sm100_module_cutlass_fp4
from flashinfer.quantization.fp4_quantization import (
    SfLayout,
    e2m1_and_ufp8sf_scale_to_float,
    nvfp4_quantize,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor, TensorType


def _set_up_ninja_path() -> None:
    """Add ninja binary to PATH for FlashInfer JIT compilation.

    FlashInfer relies on ninja to JIT-compile kernels. In Bazel's
    pycross_wheel_library environment, ninja.BIN_DIR can be empty, so we
    locate the binary relative to the installed ninja package and prepend
    it to PATH. This must run before FlashInfer is imported or initialized.
    """
    try:
        import ninja  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        # ninja not available: let flashinfer import fail separately.
        return

    ninja_bin_dir = ninja.BIN_DIR
    if not ninja_bin_dir:
        # In Bazel pycross_wheel_library, bin is at ../../bin relative to
        # the package location.
        ninja_bin_dir = os.path.normpath(
            os.path.join(os.path.dirname(ninja.__file__), "..", "..", "bin")
        )
    if ninja_bin_dir and os.path.isdir(ninja_bin_dir):
        if ninja_bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = (
                ninja_bin_dir + os.pathsep + os.environ.get("PATH", "")
            )


def test_flashinfer_fp4_gemm_custom_op() -> None:
    """Test the FlashInfer FP4 GEMM Mojo custom op."""
    ops_path = Path(os.environ["FLASHINFER_FP4_GEMM_OPS_PATH"])

    # Build the FlashInfer FP4 GEMM kernel.
    _set_up_ninja_path()
    spec = gen_gemm_sm100_module_cutlass_fp4()
    spec.build(verbose=True)
    lib_path = str(spec.jit_library_path)

    M, N, K = 128, 256, 512
    accelerator = Accelerator()

    mat1_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    mat2_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # `448` is the max representable FP8 (E4M3) value, while `6` is the max
    # representable FP4 (E2M1) value, so `448 * 6 = 2688` is the largest value
    # the quantized value can take on.
    mat1_global_sf = (
        (448 * 6) / mat1_bf16.float().abs().nan_to_num().max()
    ).float()
    mat2_global_sf = (
        (448 * 6) / mat2_bf16.float().abs().nan_to_num().max()
    ).float()

    mat1_fp4, mat1_sf = nvfp4_quantize(
        mat1_bf16,
        mat1_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    mat2_fp4, mat2_sf = nvfp4_quantize(
        mat2_bf16,
        mat2_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )

    # Combined global scale: the GEMM kernel applies this to compensate
    # for the global scaling applied during quantization.
    alpha = 1.0 / (mat1_global_sf * mat2_global_sf)

    # The custom op expects uint8 tensors (packed FP4 data) with shape
    # [M, K/2] and [N, K/2] respectively. nvfp4_quantize already returns
    # packed uint8 data in that shape.
    mat1 = Tensor.from_dlpack(Buffer.from_dlpack(mat1_fp4.view(torch.uint8)))
    mat2 = Tensor.from_dlpack(Buffer.from_dlpack(mat2_fp4.view(torch.uint8)))

    mat1_scale = Tensor.from_dlpack(
        Buffer.from_dlpack(mat1_sf.view(torch.uint8).reshape(-1).contiguous())
    )
    mat2_scale = Tensor.from_dlpack(
        Buffer.from_dlpack(mat2_sf.view(torch.uint8).reshape(-1).contiguous())
    )
    global_scale = Tensor.from_dlpack(
        Buffer.from_dlpack(alpha.reshape(1).contiguous())
    )

    workspace = Tensor.from_dlpack(
        Buffer(shape=[1024 * 1024], dtype=DType.int8, device=accelerator)
    )

    out_type = TensorType(
        shape=[M, N], dtype=DType.bfloat16, device=accelerator
    )

    assert mat1.device.api == "cuda", "Expected CUDA device"

    dev = tvm_ffi.device("cuda:0")
    stream = torch.cuda.current_stream().cuda_stream

    with tvm_ffi.use_raw_stream(dev, stream):
        result = F.custom(
            "flashinfer_fp4_gemm",
            device=mat1.device,
            values=[
                mat1,
                mat2,
                mat1_scale,
                mat2_scale,
                global_scale,
                workspace,
            ],
            out_types=[out_type],
            parameters={"lib_path": lib_path},
            custom_extensions=[ops_path],
        )[0]

    assert result.real
    assert result.type.shape == [M, N]
    assert result.type.dtype == DType.bfloat16

    # Dequantize the same quantized inputs back to float32 for reference.
    # e2m1_and_ufp8sf_scale_to_float multiplies by global_scale_tensor,
    # so we pass the inverse to undo the scaling applied during quantization.
    mat1_deq = e2m1_and_ufp8sf_scale_to_float(
        mat1_fp4.cpu().view(torch.uint8),
        mat1_sf.cpu().view(torch.uint8).reshape(-1),
        (1.0 / mat1_global_sf).cpu(),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    )
    mat2_deq = e2m1_and_ufp8sf_scale_to_float(
        mat2_fp4.cpu().view(torch.uint8),
        mat2_sf.cpu().view(torch.uint8).reshape(-1),
        (1.0 / mat2_global_sf).cpu(),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    )

    ref_output = (mat1_deq @ mat2_deq.T).to(torch.bfloat16)

    result_torch = torch.from_dlpack(result)

    assert not torch.isnan(result_torch).any(), "Output contains NaN"
    assert not torch.isinf(result_torch).any(), "Output contains Inf"
    assert result_torch.abs().max() > 0, "Output is all zeros"

    torch.testing.assert_close(
        result_torch.cpu().float(),
        ref_output.cpu().float(),
        atol=0.05,
        rtol=0.05,
    )
