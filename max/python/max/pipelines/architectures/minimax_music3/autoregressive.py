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
"""The frame loop: two models, eight steps a frame, two sequences throughout.

The two sequences are the classifier-free guidance pair. They are a *batch*, not
two runs: nothing in this architecture mixes across the batch, so guiding costs
one wider step rather than two steps.

A frame is eight steps -- one of the global model, seven of the depth decoder --
and they cannot be pipelined, because each needs what the other just produced.
What can be shared is graphs, and both stages compile exactly one apiece. For the
global model that is because attention is ragged, so prefill and decode differ
only in the row offsets, and compiling 36 layers twice would double a startup cost
measured in minutes. For the depth decoder it is because attention is causal, so
one graph at the full eight positions serves every step.

Sampling happens in the graph, in :mod:`.sampling`, and that is what makes the
depth decoder's seven steps one execution rather than seven: a code drawn on the
device feeds the next step without the loop ever learning what it was. What still
crosses per frame is the frame's conditioning, its finished codes, and the one
integer the loop cannot avoid reading -- the semantic code, which decides whether
the audio has ended.

The cache is the paged manager the text pipelines use, driven by hand. Its
bookkeeping is in tokens, and this model's tokens are frames, so the contexts here
hold placeholder ids: nothing reads them, and their only job is to tell the
manager how far each sequence has advanced.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn import CompiledModel
from max.experimental.tensor import Tensor, default_dtype
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import Weights
from max.nn.kv_cache import KVCacheInputs, MHAKVCacheParams
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache import PagedKVCacheManager

from . import sampling
from .components.depth_decoder import DepthDecoder
from .components.language_model import LanguageModel
from .model_config import (
    DepthDecoderConfig,
    LanguageModelConfig,
    SamplingConfig,
)
from .weight_adapters import (
    convert_depth_decoder_state,
    convert_language_model_state,
)

# The guidance pair.
BATCH = 2

# A forced code the graph should ignore and draw for itself. Any negative value
# would do; the codebooks are indexed from zero.
FREE = -1


def host(tensor: Tensor) -> npt.NDArray[np.float32]:
    """Copy a float32 device tensor to the host.

    Copied rather than viewed because the runtime is free to reuse the
    execution's output storage on the next call, and the frame loop keeps what
    it reads.
    """
    return tensor.to_numpy().copy()


class Step(NamedTuple):
    """What one step of the global model leaves for the rest of the frame.

    Four outputs of one tensor, in three dtypes and going four places, which is
    why they are cast in the graph rather than on the host.
    """

    hidden: Tensor
    """``(BATCH, hidden_size)``, the graph's dtype: feeds the depth sequence."""

    logits: Tensor
    """``(BATCH, head_vocab_size)`` float32. Nothing in the frame loop reads it:
    it is the gates' window onto what the draw below was made from."""

    code: Tensor
    """``(1,)`` int32, the frame's semantic code, drawn in the graph."""

    conditioning: Tensor
    """``(hidden_size,)`` float32, the conditional row: the diffusion stage's."""


class _Step(LanguageModel):
    """One step of the global model, as the frame loop consumes it.

    Takes the cache as the flat tensors an execution passes and returns what the
    loop reads, so that the shape of the graph lives with the model rather than
    in the driver. Subclassing rather than wrapping keeps the parameter paths --
    and so the checkpoint keys -- identical to :class:`LanguageModel`'s.
    """

    def __init__(
        self,
        config: LanguageModelConfig,
        kv_params: MHAKVCacheParams,
        device: Device,
        max_seq_len: int,
        recipe: SamplingConfig,
    ) -> None:
        super().__init__(config, kv_params, device, max_seq_len)
        self.recipe = recipe

    def input_types(self, batch: int) -> tuple[TensorType | BufferType, ...]:
        """The model's own inputs, plus the seed its draw rotates from."""
        embeds, offsets, *cache = super().input_types(batch)
        return (embeds, offsets, self.seed_type, *cache)

    @property
    def seed_type(self) -> TensorType:
        """A per-execution RNG seed, as :func:`max.graph.ops.random.SeedType`."""
        return TensorType(DType.uint64, [1], self.device)

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Tensor,
        input_row_offsets: Tensor,
        seed: Tensor,
        *cache: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = self.kv_params.unflatten_kv_inputs(
            iter(tensor._graph_value for tensor in cache)
        )
        assert isinstance(kv_inputs, KVCacheInputs)
        hidden, logits = super().forward(
            inputs_embeds, kv_inputs.inputs[0], input_row_offsets
        )
        sampling.seed(seed)
        scores = logits.cast(DType.float32)
        # The hidden state stays in the graph's dtype because its next stop is
        # another graph. The rest are cast because theirs is the host.
        return (
            hidden,
            scores,
            sampling.draw(
                scores,
                scale=self.recipe.cfg_scale,
                sampling_top_k=self.recipe.sampling_top_k,
                cfg_top_k=self.recipe.cfg_top_k,
            ),
            hidden[0].cast(DType.float32),
        )


class LanguageModelStage:
    """The global model, its cache, and the two guided sequences.

    One instance owns one request: :meth:`start` claims the cache, then
    :meth:`prefill` and :meth:`decode` advance it, and :meth:`release` returns
    the blocks.
    """

    def __init__(
        self,
        session: InferenceSession,
        device: Device,
        config: LanguageModelConfig,
        weights: Weights,
        *,
        dtype: DType,
        max_length: int,
        recipe: SamplingConfig,
    ) -> None:
        """Compiles the model and sizes its cache.

        Args:
            session: The session the *cache manager* allocates and copies
                through. The model brings its own: :meth:`Module.compile` owns
                that, and nothing here needs to reach into it.
            device: Where the weights and cache live.
            config: The model's shape and its token-id contract.
            weights: The opened ``language_model`` checkpoint.
            dtype: ``bfloat16`` in the pipeline. ``float32`` does not fit on a
                22 GiB device and is only ever used on the host, for gating.
            max_length: Prompt plus frames, which sizes the block pool.
            recipe: How the semantic code is drawn, which the graph bakes in.
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self.kv_params = MHAKVCacheParams(
            dtype=dtype,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            num_layers=config.num_hidden_layers,
            devices=[DeviceRef.from_device(device)],
        )
        pages_per_sequence = -(-max_length // self.kv_params.page_size)
        self.cache = PagedKVCacheManager(
            params=self.kv_params,
            session=session,
            total_num_pages=pages_per_sequence * BATCH,
            max_batch_size=BATCH,
        )

        # The parameters are declared but never materialized under `F.lazy()`;
        # `compile` binds the checkpoint's values to them.
        with F.lazy(), default_dtype(dtype):
            module = _Step(
                config,
                self.kv_params,
                device,
                max_seq_len=max_length,
                recipe=recipe,
            ).to(device)
        # `_Step` widens `LanguageModel`'s two outputs to four, which the
        # compiled type cannot express: it is derived from the base signature.
        self.model: CompiledModel[..., Any] = module.compile(
            *module.input_types(BATCH),
            weights=convert_language_model_state(weights, config, dtype),
        )
        self.contexts: list[TextContext] = []

    def start(self, prompt_length: int) -> None:
        """Claim cache blocks for a request whose prompt is *prompt_length* long."""
        self.release()
        self.contexts = [
            TextContext(
                max_length=self.max_length,
                # Placeholders: the manager counts tokens, and this model's are
                # frames that have no ids.
                tokens=TokenBuffer(np.zeros(prompt_length, dtype=np.int64)),
            )
            for _ in range(BATCH)
        ]
        for context in self.contexts:
            self.cache.claim(context)

    def release(self) -> None:
        """Return this request's blocks to the pool."""
        for context in self.contexts:
            self.cache.release(context)
        self.contexts = []

    def _step(self, embeds: Tensor, width: int, seed: Tensor) -> Step:
        """Run one step of *width* positions per sequence.

        Args:
            embeds: ``(BATCH * width, hidden_size)`` input embeddings, the two
                sequences laid end to end.
            width: Positions per sequence: the prompt length, or one frame.
            seed: This execution's RNG seed, which its draw rotates from.

        Returns:
            What the rest of the frame needs, all still on the device.
        """
        for context in self.contexts:
            self.cache.alloc(context)
        offsets = Tensor.from_dlpack(
            np.arange(BATCH + 1, dtype=np.uint32) * width
        ).to(self.device)
        cache_inputs = self.cache.runtime_inputs_for_leaf([self.contexts])
        outputs = self.model(
            embeds, offsets, seed, *cache_inputs.inputs[0].flatten()
        )
        for context in self.contexts:
            # Marks this step's positions processed and queues one more, which is
            # the next frame. The id is a placeholder; ``step`` then checks that
            # the blocks the forward wrote match what the context claims.
            context.update(0)
            self.cache.step(context)
        hidden, logits, code, conditioning = outputs
        return Step(
            hidden=hidden, logits=logits, code=code, conditioning=conditioning
        )

    def prefill(self, embeds: Tensor, prompt_length: int, seed: Tensor) -> Step:
        """Consume the prompt, returning what the first frame is chosen from."""
        return self._step(embeds, prompt_length, seed)

    def decode(self, embeds: Tensor, seed: Tensor) -> Step:
        """Consume one frame's feedback embedding per sequence.

        A frame costs one position, not eight: its codes are summed into a single
        vector before they reach the model, which is why a six-minute song fits in
        a 10240-position window.
        """
        return self._step(embeds, 1, seed)


class Frame(NamedTuple):
    """One finished frame: its codes, what it conditions, and what it feeds back."""

    codes: Tensor
    """``(num_codebooks,)`` int32, this frame's semantic code and its seven
    residuals."""

    conditioning: Tensor
    """``((num_codebooks - 1) * hidden_size,)`` float32, this stage's half of the
    diffusion conditioning: the conditional row's depth states, in depth order."""

    feedback: Tensor
    """``(BATCH, hidden_size)``, the global model's next input embedding."""

    logits: tuple[Tensor, ...]
    """One ``(BATCH, audio_vocab_size)`` float32 row per residual level, each as
    its own draw saw it. Nothing in the frame loop reads these either."""


class _Frame(DepthDecoder):
    """A whole frame's depth sequence, drawn level by level inside one graph.

    The seven residual levels cannot be parallelized -- each one's code is an
    input to the next one's embedding -- but they do not have to be seven
    executions. Unrolled here they are one, and the codes never leave the device,
    which is the entire point: on the host each level costs a full drain of the
    pipeline to move a single integer.

    Unrolling also stops the graph computing what it will not read. Every level
    runs the same four layers over the same eight positions, but only one head:
    level *j*'s. Executed seven times, the old shape evaluated all seven heads
    each time and discarded six.

    Subclassed for the same reason :class:`_Step` is: the graph's shape belongs
    with the model, and the parameter paths have to stay the decoder's own.
    """

    def __init__(
        self,
        config: DepthDecoderConfig,
        semantic_vocab_size: int,
        recipe: SamplingConfig,
    ) -> None:
        super().__init__(config, semantic_vocab_size)
        self.recipe = recipe

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.dtype, [BATCH, self.config.hidden_size], self.device
            ),
            TensorType(DType.int32, [1], self.device),
            TensorType(DType.uint64, [1], self.device),
            TensorType(
                DType.int32, [self.config.num_codebooks - 1], self.device
            ),
        )

    def _column(self, code: Tensor) -> Tensor:
        """One code, as a column both guided sequences carry."""
        return F.broadcast_to(code.reshape([1, 1]), [BATCH, 1])

    def _replace(self, codes: Tensor, level: int, code: Tensor) -> Tensor:
        """``codes`` with column *level* set to *code*."""
        columns = [codes[:, :level], self._column(code)]
        if level + 1 < self.config.num_codebooks:
            columns.append(codes[:, level + 1 :])
        return F.concat(columns, axis=1)

    def _states(self, hidden: Tensor, codes: Tensor) -> Tensor:
        """The depth sequence's states, given the codes chosen so far.

        The tail past the level being decoded is unread rather than special:
        causality means position *j* sees only the codes before it, so a zero
        there is a valid row that nothing looks at.
        """
        return super().forward(
            self.sequence(hidden, codes[:, : self.config.num_codebooks - 1])
        )

    def forward(  # type: ignore[override]
        self, hidden: Tensor, semantic: Tensor, seed: Tensor, forced: Tensor
    ) -> tuple[Tensor, ...]:
        sampling.seed(seed)
        codebooks = self.config.num_codebooks
        # Both rows carry the same codes: guidance splits the two sequences only
        # inside the model, and a frame has one set of codes either way.
        codes = F.concat(
            [
                self._column(semantic),
                # `F.full` rather than `F.zeros`, which fills with a float and
                # cannot make an integer tensor.
                F.full([BATCH, codebooks - 1], 0, dtype=DType.int32),
            ],
            axis=1,
        )
        logits: list[Tensor] = []
        for level in range(1, codebooks):
            # Head *j* is only ever read at position ``j + 1``, the one whose
            # output predicts the code it scores.
            head = self.audio_heads[level - 1]
            scores = head(self._states(hidden, codes)[:, level]).cast(
                DType.float32
            )
            drawn = sampling.draw(
                scores,
                scale=self.recipe.cfg_scale,
                sampling_top_k=self.recipe.sampling_top_k,
            )
            override = forced[level - 1 : level]
            codes = self._replace(
                codes, level, F.where(override < 0, drawn, override)
            )
            logits.append(scores)
        # One more pass, because the feedback embedding needs the code the loop
        # above just chose. Taking the conditioning from here as well is exact
        # rather than merely close: position *j* depends on the codes before it
        # and nothing after, so once every code is chosen every position's state
        # is what it was when its own level ran.
        states = self._states(hidden, codes)
        return (
            codes[0],
            # This stage's half of the diffusion conditioning: the conditional
            # row's states from position 1 on, laid end to end. Position 0 is
            # dropped because it carries the global state projected, which the
            # frame already contributes unprojected.
            F.reshape(states[0, 1:], [-1]).cast(DType.float32),
            # Squeezed because the global model reads its input ragged: the
            # frame is one position of each sequence, not a sequence axis.
            F.squeeze(self.feedback(codes), 1),
            *logits,
        )


class DepthDecoderStage:
    """The depth decoder, and the seven residual codes of one frame.

    One graph and one execution per frame. It is compiled at the full eight
    positions, with the seven levels and the frame's closing pass unrolled inside
    it, so a frame costs eight passes of a four-layer model with nothing crossing
    to the host in between.
    """

    def __init__(
        self,
        device: Device,
        config: DepthDecoderConfig,
        weights: Weights,
        language_model: Weights,
        language_model_config: LanguageModelConfig,
        *,
        dtype: DType,
        recipe: SamplingConfig,
    ) -> None:
        """Compiles the graph.

        Args:
            device: Where the weights live.
            config: The depth decoder's shape.
            weights: The opened ``rvq_depth_decoder`` checkpoint.
            language_model: The opened ``language_model`` checkpoint, read only
                for the semantic codes' embedding rows.
            language_model_config: Says which rows those are.
            dtype: Weight and activation dtype.
            recipe: How each residual code is drawn, which the graph bakes in.
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self._free = Tensor.from_dlpack(
            np.full(config.num_codebooks - 1, FREE, dtype=np.int32)
        ).to(device)

        with F.lazy(), default_dtype(dtype):
            module = _Frame(
                config, language_model_config.semantic_vocab_size, recipe
            ).to(device)
        self.decoder: CompiledModel[..., Any] = module.compile(
            *module.input_types(),
            weights=convert_depth_decoder_state(
                weights, language_model, language_model_config, dtype
            ),
        )

    def frame(
        self,
        hidden: Tensor,
        semantic: Tensor,
        seed: Tensor,
        forced: Tensor | None = None,
    ) -> Frame:
        """Draw one frame's residual codes, given its semantic code.

        Args:
            hidden: ``(BATCH, hidden_size)`` from the global model, which
                position 0 of the depth sequence carries.
            semantic: ``(1,)`` int32, the code the global model chose. Its row of
                the sliced head is also its row of the semantic embedding table,
                so it needs no translation.
            seed: This execution's RNG seed.
            forced: ``(num_codebooks - 1,)`` int32 overriding the draws, one per
                residual level, with a negative entry left to the graph. For
                teacher-forced gating; :obj:`None` draws every level.

        Returns:
            The finished frame, still on the device.
        """
        codes, conditioning, feedback, *logits = self.decoder(
            hidden, semantic, seed, self._free if forced is None else forced
        )
        return Frame(codes, conditioning, feedback, tuple(logits))


class Generation(NamedTuple):
    """One request's audio, as the codes chosen and what they condition."""

    codes: npt.NDArray[np.int32]
    """``(frames, num_codebooks)``, kept for debugging and for regression tests."""

    conditioning: npt.NDArray[np.float32]
    """``(frames, num_codebooks * hidden_size)``, the diffusion stage's input."""


Observer = Callable[[Step, Frame], None]
"""Called once per frame with what its two graphs produced, the logits included.

Sampling in the graph means the loop never sees what a draw was made from, and a
teacher-forced comparison against the reference needs exactly that. This is the
seam it reaches through; generation itself passes nothing.
"""


class Generator:
    """The two stages, and the frame loop that walks them together.

    A frame costs one step of the global model and seven of the depth decoder, and
    the two are interleaved rather than pipelined: the depth decoder needs the
    frame's global state, and the global model needs the frame's finished codes.
    """

    def __init__(
        self,
        language: LanguageModelStage,
        depth: DepthDecoderStage,
        config: LanguageModelConfig,
    ) -> None:
        self.language = language
        self.depth = depth
        self.config = config

    def _seed(self, rng: np.random.Generator) -> Tensor:
        """One execution's seed, drawn from the request's generator.

        Per execution rather than per request: a compiled graph's random ops
        rotate from whatever seed the execution hands them, so reusing one would
        make every frame draw the same codes.
        """
        return Tensor.from_dlpack(
            rng.integers(1 << 63, size=1, dtype=np.uint64)
        ).to(self.language.device)

    def _forced(self, codes: npt.NDArray[np.int32]) -> Tensor:
        """Stage a teacher-forced row of codes on the device."""
        return Tensor.from_dlpack(
            np.ascontiguousarray(codes, dtype=np.int32)
        ).to(self.language.device)

    def generate(
        self,
        prompt: Tensor,
        prompt_length: int,
        max_frames: int,
        *,
        rng: np.random.Generator,
        forced: npt.NDArray[np.int32] | None = None,
        observe: Observer | None = None,
    ) -> Generation:
        """Generate up to *max_frames* frames from an embedded prompt.

        Args:
            prompt: ``(BATCH * prompt_length, hidden_size)``, the two prompts
                laid end to end.
            prompt_length: Positions per prompt. Both are padded to one length.
            max_frames: Stop here even if the model would keep going.
            rng: Seeds each execution's in-graph draw. The seed is what makes a
                render reproducible; the draws themselves happen on the device.
            forced: ``(frames, num_codebooks)`` int32 overriding the draws, one
                row per frame, with negative entries left to the graph. For
                teacher-forced gating; :obj:`None` draws everything.
            observe: Sees each frame's logits, which nothing else does.

        Returns:
            The frames' codes and their conditioning.

        Raises:
            ValueError: If the model ends the audio before emitting a frame,
                which means the prompt is unusable rather than that the run
                failed.
        """
        self.language.start(prompt_length)
        step = self.language.prefill(prompt, prompt_length, self._seed(rng))
        codes: list[npt.NDArray[np.int32]] = []
        conditioning: list[npt.NDArray[np.float32]] = []

        # One extra pass, because the first is not a frame: it only advances the
        # state past `<|audio_start|>`. Its codes are still sampled and still fed
        # back -- they are what the second frame is conditioned on -- but nothing
        # is generated from them, so they never reach the diffusion stage.
        for index in range(max_frames + 1):
            row = None if forced is None else forced[index]
            chosen = step.code if row is None else self._forced(row[:1])
            # The one read the loop cannot defer: the semantic code decides
            # whether there is a frame to decode at all.
            if int(chosen.to_numpy()[0]) == self.config.end_of_audio_row:
                break
            frame = self.depth.frame(
                step.hidden,
                chosen,
                self._seed(rng),
                None if row is None else self._forced(row[1:]),
            )
            if observe is not None:
                observe(step, frame)
            if index:
                codes.append(frame.codes.to_numpy().copy())
                whole = (host(step.conditioning), host(frame.conditioning))
                conditioning.append(np.concatenate(whole))
                if len(conditioning) == max_frames:
                    break
            step = self.language.decode(frame.feedback, self._seed(rng))

        if not conditioning:
            raise ValueError(
                "the model ended the audio before emitting a frame, so this "
                "prompt generates nothing"
            )
        return Generation(
            np.stack(codes), np.stack(conditioning).astype(np.float32)
        )
