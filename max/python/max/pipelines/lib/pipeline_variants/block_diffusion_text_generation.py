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
"""Block-diffusion text generation pipeline.

One scheduler step is one full canvas: an encoder pass commits the pending
tokens (the prompt on the first step, the previously accepted canvas
afterwards) into the paged KV cache, then an inner denoising loop runs the
decoder graph up to ``max_denoising_steps`` times with entropy-bound
acceptance, a linear temperature schedule, self-conditioning feedback, and
stable-and-confident early stopping. The accepted canvas, truncated at the
first EOS, is appended to each request context, so the serving frontend
streams up to ``canvas_length`` tokens per step.

Decoder steps never advance the committed context length: canvas K/V land in
the cache slots after each request's committed length and are overwritten on
every denoise step, then once more by the next encoder pass with causal
values. Architectures selecting this pipeline should set
``required_arguments={"enable_prefix_caching": False}`` until block reuse
across requests is audited against that pattern.

Architectures select this pipeline via the ``pipeline_cls`` field on
:class:`~max.pipelines.lib.registry.SupportedArchitecture`.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.pipelines.context import TextContext, TextGenerationOutput
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.modeling.types import RequestID, TextGenerationInputs
from transformers import GenerationConfig

from .text_generation import TextGenerationPipeline

logger = logging.getLogger(__name__)

# HF defaults from DiffusionGemmaGenerationConfig._get_default_generation_params.
# Overridden per-field from the checkpoint's generation_config.json when present.
_DEFAULT_GENERATION_PARAMS = {
    "max_denoising_steps": 48,
    "t_min": 0.4,
    "t_max": 0.8,
    "entropy_bound": 0.1,
    "stability_threshold": 1,
    "confidence_threshold": 0.005,
    "pad_token_id": 0,
}


class _CanvasTokenView:
    """Context proxy standing in for a request during denoise steps.

    Presents the committed tokens as processed and the canvas as the active
    window, leaving the real request context untouched until the canvas is
    accepted. Every attribute other than ``tokens`` (and ``set_canvas``)
    delegates to the real context, so cache-manager paths that read fields
    like ``spec_decoding_state`` or ``lora_name`` see the request's real
    state.
    """

    def __init__(self, ctx: TextContext, canvas: np.ndarray) -> None:
        object.__setattr__(self, "_ctx", ctx)
        committed = np.asarray(ctx.tokens.all, dtype=np.int64)
        tokens = TokenBuffer(
            np.concatenate([committed, canvas.astype(np.int64)])
        )
        if len(committed):
            tokens.skip_processing(len(committed))
        object.__setattr__(self, "tokens", tokens)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_ctx"), name)

    def set_canvas(self, canvas: np.ndarray) -> None:
        """Overwrites the active (canvas) token values in place."""
        active = self.tokens.active
        active[:] = canvas


class BlockDiffusionTextGenerationPipeline(TextGenerationPipeline[TextContext]):
    """Drives block-diffusion generation over the compiled graph pair."""

    def _generation_params(self) -> dict[str, Any]:
        if not hasattr(self, "_diffusion_params"):
            params = dict(_DEFAULT_GENERATION_PARAMS)
            try:
                gc = GenerationConfig.from_pretrained(
                    self._pipeline_config.model.model_path
                )
                for key in params:
                    value = getattr(gc, key, None)
                    if value is not None:
                        params[key] = value
                sampler_cfg = getattr(gc, "sampler_config", None)
                if isinstance(sampler_cfg, dict):
                    params["entropy_bound"] = sampler_cfg.get(
                        "entropy_bound", params["entropy_bound"]
                    )
            except Exception:
                logger.warning(
                    "Could not read generation_config.json; using HF defaults.",
                    exc_info=True,
                )
            self._diffusion_params = params
        return self._diffusion_params

    def _device_zeros_bf16(self, rows: int, cols: int) -> Buffer:
        host = Buffer.from_numpy(np.zeros((rows * cols,), dtype=np.uint16))
        return host.view(DType.bfloat16, [rows, cols]).to(self._devices[0])

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        """Runs one canvas: encoder commit, denoising loop, finalize."""
        model_inputs, bitmask, flat_batch = self.prepare_batch(inputs.batches)
        if bitmask is not None:
            raise ValueError(
                "Structured output is not supported by the block-diffusion"
                " pipeline yet."
            )
        if len(flat_batch) == 0:
            return {}

        # --- 1. Encoder pass: commit pending tokens into the KV cache. ---
        self._pipeline_model.execute(model_inputs=model_inputs)

        # Chunked-prefill contexts have more prompt to process before any
        # canvas can be generated; advance their chunk and emit nothing.
        canvas_batch: list[TextContext] = []
        for ctx in flat_batch:
            if ctx.tokens.actively_chunked:
                ctx.advance_token_buffer(new_token=0)
            else:
                canvas_batch.append(ctx)

        res: dict[RequestID, TextGenerationOutput] = {}
        if canvas_batch:
            res = self._generate_canvas(canvas_batch)

        # Commit prefix-cache bookkeeping after context updates.
        for ctx in inputs.flat_batch:
            self._kv_manager.step(ctx)
        return res

    def _sample_canvas(
        self,
        current: np.ndarray,
        sc_logits: Buffer,
        sc_enabled: Buffer,
        temperature: float,
        row_offsets: Buffer,
        views: list[_CanvasTokenView],
        rng: np.random.Generator,
    ) -> tuple[Buffer, np.ndarray, np.ndarray, np.ndarray]:
        """Runs one decoder forward and samples a candidate canvas.

        Returns ``(sc_logits_out, argmax, sampled, entropy)``: the device
        self-conditioning logits to feed back, and host arrays for the
        argmax canvas, the categorical sample, and per-token entropy.
        """
        pm = self._pipeline_model
        device = self._devices[0]
        batch, canvas_len = current.shape
        n_tokens = batch * canvas_len

        canvas_buf = Buffer.from_numpy(current.reshape(-1)).to(device)
        temp_buf = Buffer.from_numpy(np.array([temperature], np.float32)).to(
            device
        )
        # _CanvasTokenView delegates to a real TextContext via __getattr__;
        # cast at the cache-manager boundary (a proxy can't pass isinstance).
        kv_inputs = self._kv_manager.runtime_inputs(
            [cast("list[TextContext]", views)]
        )
        sc_out, argmax_d, topk_p_d, topk_i_d, entropy_d = (
            pm.execute_decoder_step(  # type: ignore[attr-defined]
                canvas_tokens=canvas_buf,
                input_row_offsets=row_offsets,
                sc_logits=sc_logits,
                sc_enabled=sc_enabled,
                temperature=temp_buf,
                kv_cache_inputs=kv_inputs,
            )
        )

        argmax = argmax_d.to_numpy().reshape(batch, canvas_len)
        entropy = entropy_d.to_numpy().reshape(batch, canvas_len)
        # Host categorical over the device top-64 (renormalized); the
        # truncated tail mass is negligible at schedule temperatures and
        # avoids a [N, vocab] cumsum that has no GPU kernel.
        topk_p = topk_p_d.to_numpy().reshape(n_tokens, -1)
        topk_i = topk_i_d.to_numpy().reshape(n_tokens, -1)
        cdf = np.cumsum(topk_p / topk_p.sum(axis=-1, keepdims=True), axis=-1)
        u = rng.random(n_tokens, dtype=np.float32)
        choice = (cdf < u[:, None]).sum(axis=-1).clip(0, topk_i.shape[1] - 1)
        sampled = topk_i[np.arange(n_tokens), choice].reshape(batch, canvas_len)
        return sc_out, argmax, sampled, entropy

    @staticmethod
    def _accept_and_renoise(
        current: np.ndarray,
        sampled: np.ndarray,
        entropy: np.ndarray,
        entropy_bound: float,
        vocab: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Entropy-bound acceptance + renoise (HF EntropyBoundSampler).

        Accepts the k lowest-entropy tokens such that
        ``cumsum(H_sorted) - H_sorted <= entropy_bound``; the rest are
        re-sampled uniformly from the vocabulary.
        """
        order = np.argsort(entropy, axis=-1)
        h_sorted = np.take_along_axis(entropy, order, axis=-1)
        cum = np.cumsum(h_sorted, axis=-1)
        sorted_mask = (cum - h_sorted) <= entropy_bound
        accept_mask = np.zeros_like(sorted_mask)
        np.put_along_axis(accept_mask, order, sorted_mask, axis=-1)
        return np.where(
            accept_mask,
            np.where(accept_mask, sampled, current),
            rng.integers(0, vocab, size=current.shape, dtype=np.int64),
        )

    def _finalize(
        self,
        canvas_batch: list[TextContext],
        final_argmax: np.ndarray,
    ) -> dict[RequestID, TextGenerationOutput]:
        """Appends each accepted canvas to its context (EOS-truncated).

        ``advance_token_buffer`` sets EOS / max-length status via the eos
        tracker. The appended canvas tokens are re-exposed for processing so
        the next step's encoder pass covers the whole block.
        """
        res: dict[RequestID, TextGenerationOutput] = {}
        for i, ctx in enumerate(canvas_batch):
            appended = 0
            for tok in final_argmax[i]:
                ctx.advance_token_buffer(new_token=int(tok))
                appended += 1
                if ctx.is_done:
                    break
            if appended > 1 and not ctx.is_done:
                ctx.tokens.rewind_processing(appended - 1)
            output = ctx.to_generation_output()
            if output.tokens:
                res[ctx.request_id] = output
        return res

    def _generate_canvas(
        self, canvas_batch: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        pm = self._pipeline_model
        config = pm.config  # type: ignore[attr-defined]
        params = self._generation_params()
        canvas_len: int = config.canvas_length
        vocab: int = config.text_config.vocab_size
        max_steps: int = params["max_denoising_steps"]
        t_min: float = params["t_min"]
        t_max: float = params["t_max"]
        entropy_bound: float = params["entropy_bound"]
        stability: int = params["stability_threshold"]
        confidence: float = params["confidence_threshold"]

        device = self._devices[0]
        batch = len(canvas_batch)
        n_tokens = batch * canvas_len
        rng = np.random.default_rng()

        # Canvas init: uniform random over the vocabulary (HF
        # EntropyBoundSampler.initialize_canvas).
        current = rng.integers(
            0, vocab, size=(batch, canvas_len), dtype=np.int64
        )
        views = [
            _CanvasTokenView(ctx, current[i])
            for i, ctx in enumerate(canvas_batch)
        ]
        for view in views:
            self._kv_manager.alloc(cast(TextContext, view))

        row_offsets = Buffer.from_numpy(
            np.arange(0, n_tokens + 1, canvas_len, dtype=np.uint32)
        ).to(device)
        sc_logits = self._device_zeros_bf16(n_tokens, vocab)
        sc_enabled = Buffer.from_numpy(np.zeros((1,), np.float32)).to(device)
        sc_enabled_on = Buffer.from_numpy(np.ones((1,), np.float32)).to(device)

        finished = np.zeros(batch, dtype=bool)
        final_argmax = np.zeros((batch, canvas_len), dtype=np.int64)
        argmax_history = np.full(
            (max(stability, 1), batch, canvas_len), -1, dtype=np.int64
        )

        for cur_step in range(max_steps, 0, -1):
            temperature = t_min + (t_max - t_min) * (cur_step / max_steps)
            sc_logits, argmax, sampled, entropy = self._sample_canvas(
                current,
                sc_logits,
                sc_enabled,
                temperature,
                row_offsets,
                views,
                rng,
            )
            sc_enabled = sc_enabled_on
            renoised = self._accept_and_renoise(
                current, sampled, entropy, entropy_bound, vocab, rng
            )

            # Stable-and-confident adaptive stopping (HF
            # StableAndConfidentStoppingCriteria), frozen for finished rows.
            stable = (
                (argmax_history == argmax[None, :, :]).all(axis=-1).all(axis=0)
            )
            argmax_history = np.roll(argmax_history, -1, axis=0)
            argmax_history[-1] = argmax
            confident = entropy.mean(axis=-1) < confidence
            newly_finished = (stable & confident) & ~finished
            final_argmax[newly_finished] = argmax[newly_finished]
            finished |= newly_finished

            current = np.where(finished[:, None], current, renoised)
            if finished.all():
                break
            for i, view in enumerate(views):
                if not finished[i]:
                    view.set_canvas(current[i])

        final_argmax[~finished] = argmax[~finished]
        return self._finalize(canvas_batch, final_argmax)
