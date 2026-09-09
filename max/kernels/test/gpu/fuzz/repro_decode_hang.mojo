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
#
# Minimal repro probe for the decode-path hang found by fuzz_mha_causal (the
# Slice 0 fuzzer). Runs ONE decode (seq_len=1) flash_attention call with a
# CausalPaddingMask, with `num_keys` and `valid_length` taken from argv so a
# single build can probe many configs:
#
#   ./bazelw test //max/kernels/test/gpu/fuzz:repro_decode_hang.mojo.test \
#     --test_arg=<num_keys> --test_arg=<valid_length> \
#     --test_timeout=60 --local_resources=gpu-memory=1000 \
#     --test_env=CUDA_VISIBLE_DEVICES=0
#
# A clean config prints "start" then "done" and passes; a hanging config prints
# "start" and then TIMES OUT.

from std.sys import argv
from std.random import rand
from max.gpu.host import DeviceContext
from layout import Idx, Layout, LayoutTensor, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalPaddingMask

comptime qkv_type = DType.bfloat16
comptime depth = 128
comptime num_heads = 32
comptime group = 1
comptime kv_num_heads = num_heads // group
comptime scale = Float32(0.125)


def _parse_two_ints(default_a: Int, default_b: Int) -> Tuple[Int, Int]:
    var ints = List[Int]()
    for a in argv():
        try:
            ints.append(Int(a))
        except:
            pass
    if len(ints) >= 2:
        return (ints[len(ints) - 2], ints[len(ints) - 1])
    return (default_a, default_b)


def main() raises:
    var cfg = _parse_two_ints(1, 0)
    var num_keys = cfg[0]
    var valid_length = cfg[1]
    comptime batch_size = 1
    comptime seq_len = 1

    print(
        "start: seq_len=",
        seq_len,
        "num_keys=",
        num_keys,
        "valid_length=",
        valid_length,
    )

    with DeviceContext() as ctx:
        var q_size = batch_size * num_heads * seq_len * depth
        var kv_size = batch_size * kv_num_heads * num_keys * depth

        var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
        var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
        rand(q_host.as_span())
        rand(k_host.as_span())
        rand(v_host.as_span())

        var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
        var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
        var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
        var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
        ctx.enqueue_copy(q_dev, q_host)
        ctx.enqueue_copy(k_dev, k_host)
        ctx.enqueue_copy(v_dev, v_host)

        var q = TileTensor(
            q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
        )
        var k = TileTensor(
            k_dev,
            row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
        )
        var v = TileTensor(
            v_dev,
            row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
        )
        var o = TileTensor(
            o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
        )

        var vl_dev = ctx.enqueue_create_buffer[.uint32](1)
        ctx.enqueue_memset(vl_dev, UInt32(valid_length))
        var vl = LayoutTensor[.uint32, Layout.row_major(1)](vl_dev.unsafe_ptr())
        var mask = CausalPaddingMask(vl)

        flash_attention(o, q, k, v, mask, scale, ctx)
        ctx.synchronize()

        _ = q_dev
        _ = k_dev
        _ = v_dev
        _ = o_dev
        _ = vl_dev

    print("done")
