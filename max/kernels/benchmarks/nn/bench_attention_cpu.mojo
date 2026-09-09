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

from std.random import rand

from std.benchmark import *
from std.memory import alloc, dealloc
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.attention.cpu.mha import flash_attention

from std.utils import IndexList
from std.utils.index import Index


@fieldwise_init
struct AttentionSpec(ImplicitlyCopyable, Writable):
    var batch_size: Int
    var seq_len: Int
    var kv_seq_len: Int
    var depth_dim: Int

    # fmt: off
    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the attention spec.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "batch_size=", self.batch_size,
            ",seq_len=", self.seq_len,
            ",kv_seq_len=", self.kv_seq_len,
            ",depth_dim=", self.depth_dim,
        )
    # fmt: on


def bench_attention[dtype: DType](mut m: Bench, spec: AttentionSpec) raises:
    var q_shape = Index(spec.batch_size, spec.seq_len, spec.depth_dim)
    var kv_shape = Index(spec.batch_size, spec.kv_seq_len, spec.depth_dim)
    var mask_shape = Index(spec.batch_size, spec.seq_len, spec.kv_seq_len)

    var q_alloc = alloc[Scalar[dtype]](
        {count = q_shape.flattened_length()}
    ).into_managed()
    var k_alloc = alloc[Scalar[dtype]](
        {count = kv_shape.flattened_length()}
    ).into_managed()
    var v_alloc = alloc[Scalar[dtype]](
        {count = kv_shape.flattened_length()}
    ).into_managed()
    var mask_alloc = alloc[Scalar[dtype]](
        {count = mask_shape.flattened_length()}
    ).into_managed()
    var output_alloc = alloc[Scalar[dtype]](
        {count = q_shape.flattened_length()}
    ).into_managed()

    rand(q_alloc.unsafe_span())
    rand(k_alloc.unsafe_span())
    rand(v_alloc.unsafe_span())
    rand(mask_alloc.unsafe_span())

    comptime layout = Layout.row_major[3]()
    var q = LayoutTensor[dtype, layout](
        q_alloc.unsafe_ptr(), RuntimeLayout[layout].row_major(q_shape)
    )
    var k = LayoutTensor[dtype, layout](
        k_alloc.unsafe_ptr(), RuntimeLayout[layout].row_major(kv_shape)
    )
    var v = LayoutTensor[dtype, layout](
        v_alloc.unsafe_ptr(), RuntimeLayout[layout].row_major(kv_shape)
    )
    var mask = LayoutTensor[dtype, layout](
        mask_alloc.unsafe_ptr(), RuntimeLayout[layout].row_major(mask_shape)
    )
    var output = LayoutTensor[dtype, layout](
        output_alloc.unsafe_ptr(), RuntimeLayout[layout].row_major(q_shape)
    )

    @always_inline
    def input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) capturing -> SIMD[dtype, simd_width]:
        return k.load[width=simd_width](rebind[IndexList[3]](idx))

    @always_inline
    def input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) capturing -> SIMD[dtype, simd_width]:
        return v.load[width=simd_width](rebind[IndexList[3]](idx))

    @always_inline
    def mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) capturing -> SIMD[dtype, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[3]](idx))

    comptime scale = 0.25

    @always_inline
    def flash_bench_fn(mut b: Bencher) {imm}:
        @always_inline
        def iter_fn[depth_static_dim: Int]() {imm}:
            comptime output_static_shape = IndexList[3](
                UNKNOWN_VALUE, UNKNOWN_VALUE, depth_static_dim
            )
            flash_attention[input_k_fn, input_v_fn, mask_fn](
                q,
                k.runtime_layout.shape.value.canonicalize(),
                v.runtime_layout.shape.value.canonicalize(),
                mask.runtime_layout.shape.value.canonicalize(),
                output,
                scale=scale,
            )

        comptime depth_static_dims = [40, 64, 80, 128, 160]

        comptime for idx in range(len(depth_static_dims)):
            comptime dim = depth_static_dims[idx]
            if dim == spec.depth_dim:
                # `iter` takes a closure value, and a parametric closure only
                # names an overload set, so instantiate it behind a
                # non-parametric one.
                @always_inline
                def iter_static() {imm}:
                    iter_fn[dim]()

                b.iter(iter_static)
                return

        # Fallback to dispatch with a dynamic shape.
        @always_inline
        def iter_dynamic() {imm}:
            iter_fn[UNKNOWN_VALUE]()

        b.iter(iter_dynamic)

    m.bench_function(flash_bench_fn, BenchId("flash", String(spec)))

    dealloc(q_alloc^)
    dealloc(k_alloc^)
    dealloc(v_alloc^)
    dealloc(mask_alloc^)
    dealloc(output_alloc^)


def main() raises:
    var specs = [
        # bert-base-uncased-seqlen-16-onnx.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=16,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # BERT/bert-base-uncased-seqlen-128-onnx.yaml
        # GPT-2/gpt2-small-seqlen-128.yaml
        # RoBERTa/roberta-base-hf-onnx.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=128,
            kv_seq_len=128,
            depth_dim=64,
        ),
        # CLIP-ViT/clip-vit-large-patch14-onnx.yaml
        AttentionSpec(
            batch_size=16,
            seq_len=257,
            kv_seq_len=257,
            depth_dim=64,
        ),
        # Llama2/llama2-7B-MS-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=100,
            kv_seq_len=100,
            depth_dim=128,
        ),
        # Llama2/llama2-7B-MS-token-gen-onnx.yaml
        # Mistral/mistral-7b-hf-onnx-LPTG.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=1,
            kv_seq_len=1025,
            depth_dim=128,
        ),
        # Mistral/mistral-7b-hf-onnx-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=1024,
            kv_seq_len=1024,
            depth_dim=128,
        ),
        # OpenCLIP/clip-dynamic-per-tensor-weight-type-quint8-onnx-optimized.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=50,
            kv_seq_len=50,
            depth_dim=64,
        ),
        AttentionSpec(
            batch_size=24,
            seq_len=77,
            kv_seq_len=77,
            depth_dim=64,
        ),
        # ReplitV1.5/replitv15-3B-hf-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=24,
            seq_len=1024,
            kv_seq_len=1024,
            depth_dim=128,
        ),
        # ReplitV1.5/replitv15-3B-hf-LPTG-onnx.yaml
        AttentionSpec(
            batch_size=24,
            seq_len=1,
            kv_seq_len=1025,
            depth_dim=128,
        ),
        # StableDiffusion-1.x/text_encoder/text_encoder-onnx.yaml
        AttentionSpec(
            batch_size=24,
            seq_len=16,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # StableDiffusion-1.x/unet/unet-onnx.yaml
        AttentionSpec(
            batch_size=16,
            seq_len=64,
            kv_seq_len=16,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=64,
            kv_seq_len=64,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=256,
            kv_seq_len=16,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=256,
            kv_seq_len=256,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=1024,
            kv_seq_len=16,
            depth_dim=80,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=1024,
            kv_seq_len=1024,
            depth_dim=80,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=4096,
            kv_seq_len=16,
            depth_dim=40,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=4096,
            kv_seq_len=4096,
            depth_dim=40,
        ),
        # StableDiffusion-1.x/vae_decoder/vae_decoder-onnx.yaml
        # StableDiffusion-1.x/vae_encoder/vae_encoder-onnx.yaml
        AttentionSpec(
            batch_size=2,
            seq_len=4096,
            kv_seq_len=4096,
            depth_dim=512,
        ),
        # StarCoder/starcoder-7b-hf-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=1,
            seq_len=32768,
            kv_seq_len=1024,
            depth_dim=128,
        ),
        # StarCoder/starcoder-7b-hf-token-gen-onnx.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=16,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # WavLM/wavlm-large-onnx.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=49,
            kv_seq_len=49,
            depth_dim=64,
        ),
        # Whisper/decoder_model_merged/decoder_model_merged-onnx.yaml
        AttentionSpec(
            batch_size=16,
            seq_len=1,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # Whisper/encoder_model/encoder_model-onnx.yaml
        AttentionSpec(
            batch_size=8,
            seq_len=1500,
            kv_seq_len=1500,
            depth_dim=64,
        ),
    ]

    var m = Bench()
    for i in range(len(specs)):
        bench_attention[.float32](m, specs[i])
    m.dump_report()
