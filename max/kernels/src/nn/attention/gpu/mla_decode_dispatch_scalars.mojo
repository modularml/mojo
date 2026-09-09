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
"""Device-dispatched MLA decode dispatch-metadata scalars.

This module hosts a single device-generic entry point,
`mla_decode_dispatch_scalars`, that computes the packed 3-int MLA decode
dispatch metadata `(batch_size, q_max_seq_len, num_partitions)`. It mirrors
the MHA pattern (`mha_decoding_num_partitions`), which hides the HIP-vs-SM100
device dispatch behind one function so callers (in particular the
Mojo->Python binding `mla_dispatch_args_scalar`) stay device-agnostic and
carry no `if ctx.api()` branch.

The file sits *above* both per-device heuristics in the dependency graph and
imports from both: the device-generic AMD/MHA heuristic
(`mha_decode_partition_heuristic`) and the SM100/NVIDIA runtime heuristic
(`nvidia/sm100/mla_decode_dispatch`). Co-locating the unified function in
either heuristic file would invert the dependency direction (an SM100 import
in a generic heuristic, or an AMD import in an SM100 file), so it lives in a
device-generic location instead.
"""

from max.gpu.host import DeviceAttribute, DeviceContext

from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    compute_mla_dispatch_scalars_runtime,
)


def mla_decode_dispatch_scalars(
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    ctx: DeviceContext,
) raises -> Tuple[Int, Int, Int]:
    """Compute the packed MLA decode dispatch metadata for the active device.

    Returns `(batch_size, q_max_seq_len, num_partitions)`. Dispatches on
    `ctx.api()` internally, mirroring `mha_decoding_num_partitions`:

    - **HIP (AMD):** the AMD MLA decode kernel bakes its split-K grid from
      `mha_decoding_num_partitions(..., is_mla=True)` at dispatch time and
      never reads `num_partitions` from the scalar
      buffer; the metadata `num_partitions` is used only as the HIP
      device-graph capture/replay selection key. To keep the capture key
      byte-for-byte equal to the baked grid, compute the SAME value the kernel
      bakes: pass RAW `max_cache_valid_length` and `num_heads` (`kv_heads == 1`
      for MLA so `heads_per_group == num_heads`), with NO `q_max_seq_len`
      adjustment.
    - **CUDA (NVIDIA/SM100):** delegate to the SM100 runtime heuristic
      `compute_mla_dispatch_scalars_runtime`, which reads the device SM count.
    - **Any other api (e.g. Metal):** raise: there is no MLA decode dispatch
      path for non-HIP/non-CUDA devices, so route them to a clear error rather
      than silently into the SM100 heuristic.

    Args:
        batch_size: Current decode batch size.
        max_cache_valid_length: Max valid KV-cache length across the batch.
        q_max_seq_len: Max query sequence length (1 for decode).
        num_heads: Number of Q attention heads.
        is_fp8_kv: Whether the KV cache is FP8 (selects the SM100 sub-heuristic).
        ctx: Device context used to read `api()` and the SM count.

    Returns:
        The packed `(batch_size, q_max_seq_len, num_partitions)` metadata.
    """
    if ctx.api() == "hip":
        # MLA: kv_num_heads == 1, so heads_per_group == num_heads. Pass raw
        # max_cache_valid_length — byte-for-byte the same value
        # `mha_decoding_num_partitions` bakes into the AMD split-K grid.
        var np = mha_decoding_num_partitions(
            batch_size,
            max_cache_valid_length,
            num_heads,
            ctx,
            is_mla=True,
        )
        return (batch_size, q_max_seq_len, np)
    elif ctx.api() == "cuda":
        # SM100/NVIDIA runtime heuristic (reads the device SM count). CUDA is
        # the only non-HIP device with an MLA decode dispatch path.
        return compute_mla_dispatch_scalars_runtime(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
        )
    else:
        raise Error(
            "mla_decode_dispatch_scalars supports only HIP (AMD) and CUDA"
            " (NVIDIA/SM100) device contexts."
        )
