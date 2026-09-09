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

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer.transformer import logits_postprocess

from ..llama3.model_config import create_rope_embedding
from .full_attention import LFM2FullAttention
from .model_config import LFM2Config


class LFM2MLP(Module):
    def __init__(
        self, config: LFM2Config, linear_cls: functools.partial[Linear]
    ):
        super().__init__()
        self.w1 = linear_cls(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )
        self.w3 = linear_cls(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )
        self.w2 = linear_cls(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class LFM2ShortConv(Module):
    def __init__(
        self, config: LFM2Config, linear_cls: functools.partial[Linear]
    ):
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.kernel_size = config.conv_L_cache
        self.in_proj = linear_cls(
            in_dim=hidden,
            out_dim=3 * hidden,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=config.conv_bias,
        )
        self.out_proj = linear_cls(
            in_dim=hidden,
            out_dim=hidden,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=config.conv_bias,
        )
        self.conv_weight = Weight(
            "conv_weight",
            config.dtype,
            [hidden, 1, self.kernel_size],
            config.devices[0],
        )
        self.conv_bias: Weight | None
        if config.conv_bias:
            self.conv_bias = Weight(
                "conv_bias", config.dtype, [hidden], config.devices[0]
            )
        else:
            self.conv_bias = None

    def __call__(
        self,
        x: TensorValue,
        conv_state: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        bcx = self.in_proj(x)
        b, c, xv = ops.split(
            bcx,
            [self.hidden_size, self.hidden_size, self.hidden_size],
            axis=-1,
        )
        bx = b * xv
        conv_w = self.conv_weight.reshape([self.hidden_size, self.kernel_size])
        conv_w = ops.unsqueeze(conv_w, 0)
        conv_b = self.conv_bias

        out_init = bx * ops.constant(0.0, dtype=bx.dtype, device=bx.device)
        t_init = ops.constant(0, DType.int32, device=DeviceRef.CPU())
        seq_len = ops.cast(ops.shape_to_tensor(x.shape)[0], DType.int32).to(
            DeviceRef.CPU()
        )
        positions = ops.range(
            start=ops.constant(0, DType.int32, device=DeviceRef.CPU()),
            stop=seq_len,
            step=ops.constant(1, DType.int32, device=DeviceRef.CPU()),
            out_dim=bx.shape[0],
            dtype=DType.int32,
            device=x.device,
        )

        # Map every token to its request index in the ragged batch by counting
        # how many request end-offsets it is greater-than-or-equal to. The
        # end-offsets are input_row_offsets[1:], one entry per request.
        end_offsets = ops.cast(input_row_offsets[1:], positions.dtype).to(
            x.device
        )
        pos_2d = ops.unsqueeze(positions, 1)
        end_2d = ops.unsqueeze(end_offsets, 0)
        cmp = ops.greater_equal(pos_2d, end_2d)
        req_idx_per_token = ops.squeeze(
            ops.sum(ops.cast(cmp, DType.int32), axis=1), -1
        )

        def pred(t: TensorValue, *_: TensorValue) -> TensorValue:
            return t < seq_len

        def body(
            t: TensorValue,
            state_stack: TensorValue,
            out: TensorValue,
            pos: TensorValue,
            bx_full: TensorValue,
            cw: TensorValue,
            req_idx_full: TensorValue,
        ) -> list[TensorValue]:
            t_dev = t.to(x.device)
            t_ix = t_dev.reshape([1])
            r_ix = ops.gather(req_idx_full, t_ix, axis=0)

            # Slice this request's [1, hidden, K] state out of the [N, ...]
            # stack, slide in the new token, then scatter the result back.
            state_r = ops.gather(state_stack, r_ix, axis=0)
            bx_t = ops.gather(bx_full, t_ix, axis=0)
            bx_t_k = ops.unsqueeze(ops.permute(bx_t, [1, 0]), 0)
            state_r_next = ops.concat((state_r[:, :, 1:], bx_t_k), axis=2)

            conv_out = ops.squeeze(ops.sum(state_r_next * cw, axis=2), -1)
            if conv_b is not None:
                conv_out = conv_out + conv_b

            # scatter_nd is GPU-friendly (unlike ops.scatter, which round-trips
            # through CPU via TODO(GEX-2197)), so use it for the per-request
            # state stack update inside the loop.
            indices = ops.reshape(r_ix, [1, 1])
            state_stack_next = ops.scatter_nd(
                state_stack, state_r_next, indices
            )

            eq = ops.equal(pos, t_ix)
            mask = ops.unsqueeze(ops.cast(eq, out.dtype), -1)
            one = ops.constant(1.0, out.dtype, device=x.device)
            out_next = out * (one - mask) + conv_out * mask
            return [
                t + ops.constant(1, DType.int32, device=DeviceRef.CPU()),
                state_stack_next,
                out_next,
                pos,
                bx_full,
                cw,
                req_idx_full,
            ]

        _, new_state, conv_out, *_ = ops.while_loop(
            [
                t_init,
                conv_state,
                out_init,
                positions,
                bx,
                conv_w,
                req_idx_per_token,
            ],
            pred,
            body,
        )
        y = c * conv_out
        return self.out_proj(y), new_state


class LFM2DecoderLayer(Module):
    def __init__(
        self,
        config: LFM2Config,
        layer_idx: int,
        kv_layer_idx: int,
        rope: RotaryEmbedding,
        linear_cls: functools.partial[Linear],
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.kv_layer_idx = kv_layer_idx
        self.operator_norm = RMSNorm(
            config.hidden_size, config.dtype, config.norm_eps
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, config.dtype, config.norm_eps
        )
        self.feed_forward = LFM2MLP(config, linear_cls)
        self.self_attn: LFM2FullAttention | None
        self.conv: LFM2ShortConv | None
        if self.layer_type == "full_attention":
            self.self_attn = LFM2FullAttention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=kv_layer_idx,
                dtype=config.dtype,
                devices=config.devices,
                linear_cls=linear_cls,
                scale=config.attention_multiplier,
                has_bias=False,
                qk_norm_eps=config.norm_eps,
                norm_dtype=config.dtype,
                attn_output_gate=False,
            )
            self.conv = None
        else:
            self.conv = LFM2ShortConv(config, linear_cls)
            self.self_attn = None

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        conv_state: TensorValue | None,
    ) -> tuple[TensorValue, TensorValue | None]:
        residual = x
        normed = self.operator_norm(x)
        new_conv_state = conv_state
        if self.self_attn is not None:
            kv_layer_idx = ops.constant(
                self.kv_layer_idx, DType.uint32, device=DeviceRef.CPU()
            )
            op_out = self.self_attn(
                kv_layer_idx,
                normed,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        else:
            if self.conv is None or conv_state is None:
                raise ValueError(
                    f"Layer {self.layer_type!r} requires conv and conv_state "
                    "to be set, but one or both are None."
                )
            op_out, new_conv_state = self.conv(
                normed, conv_state, input_row_offsets
            )

        h = residual + op_out
        h = h + self.feed_forward(self.ffn_norm(h))
        return h, new_conv_state


class LFM2(Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        assert len(config.devices) == 1
        self.config = config
        self.num_conv_layers = sum(
            1 for t in config.layer_types if t != "full_attention"
        )
        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=False,
            rope_scaling_params=None,
            longrope_scaling_params=None,
            device=config.devices[0],
        )
        linear_cls = functools.partial(Linear, quant_config=config.quant_config)
        self.rope = rope
        kv_idx = 0
        layers: list[LFM2DecoderLayer] = []
        for i, t in enumerate(config.layer_types):
            layer = LFM2DecoderLayer(
                config=config,
                layer_idx=i,
                kv_layer_idx=kv_idx,
                rope=rope,
                linear_cls=linear_cls,
            )
            if t == "full_attention":
                kv_idx += 1
            layers.append(layer)
        self.layers = LayerList(layers)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )
        self.norm = RMSNorm(config.hidden_size, config.dtype, config.norm_eps)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            config.devices[0],
            has_bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.set_shared_weight("weight", self.embed_tokens.weight)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.config.devices[0]
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        kv_inputs = kv_params.flattened_kv_inputs()
        conv_state_types = [
            TensorType(
                self.config.dtype,
                [
                    "conv_batch",
                    self.config.hidden_size,
                    self.config.conv_L_cache,
                ],
                device=device_ref,
            )
            for _ in range(self.num_conv_layers)
        ]
        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs,
            *conv_state_types,
        )

    def __call__(
        self,
        tokens: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        conv_states: list[TensorValue],
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens)
        freqs_cis = self.rope.freqs_cis.to(h.device)
        conv_idx = 0
        new_states = []
        for layer in self.layers:
            is_conv = layer.layer_type != "full_attention"
            conv_state = conv_states[conv_idx] if is_conv else None
            h, new_state = layer(
                h,
                kv_collection=kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
                conv_state=conv_state,
            )
            if is_conv:
                assert new_state is not None
                new_states.append(new_state)
                conv_idx += 1

        outputs = logits_postprocess(
            h=h,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            norm=self.norm,
            lm_head=self.lm_head,
            return_logits=self.config.return_logits,
            return_hidden_states=self.config.return_hidden_states,
            logits_scaling=self.config.logits_scaling,
        )
        return outputs + tuple(new_states)
