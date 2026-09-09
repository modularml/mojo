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
"""MAX pipeline for pixel generation using diffusion models."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np
from max.driver import load_devices
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.context import (
    GenerationStatus,
    PixelGenerationContextType,
)
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineOutputsDict,
    PixelGenerationInputs,
    RequestID,
)
from max.pipelines.request.open_responses import (
    ImageGenerationDetails,
    OutputImageContent,
    OutputVideoContent,
    Usage,
)
from max.pipelines.weights.weight_loading import auto_cast_weights_from_env

from .interface import DiffusionPipeline

if TYPE_CHECKING:
    from max.experimental.nn import CompiledModel
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.model_manifest import ModelManifest
    from max.pipelines.lib.pipeline_executor import PipelineExecutor

_logger = logging.getLogger("max.pipelines")


@runtime_checkable
class PixelGenerationModule(Protocol):
    """Interface a Module-driven architecture exposes to ``PixelGenerationPipeline``.

    A class satisfying this Protocol (typically also subclassing
    :class:`~max.experimental.nn.Module`) can be driven through the
    pipeline's ModuleV3 path: the pipeline wraps construction in
    :func:`~max.experimental.functional.lazy`, walks the Module tree
    via
    :func:`~max.pipelines.lib.weight_loader.adapt_module_loader`
    to bind weights, compiles the forward graph, and per request calls
    :meth:`prepare_inputs` to translate a :class:`PixelContext` batch
    into the compiled call's argument tuple and :meth:`from_outputs`
    to translate the resulting tensors back into a pipeline-facing
    output struct.
    """

    def __init__(self, manifest: ModelManifest) -> None: ...

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the compile-time input :class:`~max.graph.TensorType` tuple."""
        ...

    def prepare_inputs(self, contexts: list[Any]) -> Any:
        """Translates a context batch into the compiled call's argument tuple."""
        ...

    def from_outputs(self, tensors: list[Tensor]) -> Any:
        """Translates the compiled call's output tensors into a pipeline struct."""
        ...


class PixelGenerationPipeline(
    Pipeline[
        PixelGenerationInputs[PixelGenerationContextType], GenerationOutput
    ],
    Generic[PixelGenerationContextType],
):
    """Pixel generation pipeline for diffusion models.

    Args:
        pipeline_config: Configuration for the pipeline and runtime behavior.
        pipeline_model: The diffusion pipeline model class to instantiate.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[
            DiffusionPipeline
            | PipelineExecutor[Any, Any, Any]
            | Module[Any, Any]
        ],
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles
        from max.pipelines.lib.pipeline_executor import PipelineExecutor

        self._pipeline_config = pipeline_config
        # Use the first component's device_specs for session initialization.
        first_config = next(iter(pipeline_config.models.values()))
        self._devices = load_devices(first_config.device_specs)

        # Initialize Session.
        session = InferenceSession(devices=[*self._devices])

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        self._use_module = False
        self._use_executor = False
        self._module: PixelGenerationModule | None = None
        self._compiled: CompiledModel | None = None
        self._executor: PipelineExecutor[Any, Any, Any] | None = None
        self._pipeline_model: DiffusionPipeline | None = None

        if issubclass(pipeline_model, Module):
            # ModuleV3 path: construct the Module under F.lazy(), compose
            # the manifest's loader through the Module tree's
            # query-translating adapters, then compile.  Compilation,
            # session lifecycle, and the CompiledModel handle all live
            # here -- the Module itself carries only structure (forward +
            # input_types) and the request/response I/O contract
            # (prepare_inputs + from_outputs).
            self._use_module = True
            module_cls = cast(type[PixelGenerationModule], pipeline_model)
            with F.lazy():
                module_io = module_cls(manifest=pipeline_config.models)
            # The Module ABC carries ``.compile()`` and the walker's
            # ``.descendants`` traversal; the Protocol carries the I/O
            # methods.  Bridge with one cast since mypy can't express
            # the intersection (Module & PixelGenerationModule).
            module_base = cast(Module[Any, Any], module_io)
            # Compose the source loader through every Module's
            # ``adapt_loader``, then materialise just the parameters the
            # Module declares -- the loader stays cold for anything the
            # tree never asks for.
            # Imported here rather than at module scope to break a
            # circular import: ``max.pipelines.lib`` imports this module's
            # ``PixelGenerationPipeline``, so importing from
            # ``max.pipelines.lib.*`` at load time re-enters a
            # partially-initialized ``lib`` package.
            from max.pipelines.lib.weight_loader import adapt_module_loader

            loader = adapt_module_loader(
                module_base, pipeline_config.models.loader()
            )
            state_dict = {
                name: loader(name) for name, _ in module_base.parameters
            }
            self._compiled = module_base.compile(
                *module_io.input_types(),
                weights=state_dict,
                auto_cast=auto_cast_weights_from_env(),
            )
            self._module = module_io
        elif issubclass(pipeline_model, PipelineExecutor):
            self._use_executor = True
            self._executor = pipeline_model(
                manifest=pipeline_config.models,
                session=session,
                runtime_config=pipeline_config.runtime,
            )
        else:
            # Weight paths are resolved per-component inside
            # _load_sub_models.
            self._pipeline_model = pipeline_model(
                pipeline_config=self._pipeline_config,
                session=session,
                devices=self._devices,
                weight_paths=[],
                cache_config=pipeline_config.runtime.denoising_cache,
            )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    @property
    def max_batch_size(self) -> int:
        """Returns 1: pixel generation pipelines process one request at a time."""
        return 1

    def execute(
        self,
        inputs: PixelGenerationInputs[PixelGenerationContextType],
    ) -> PipelineOutputsDict[GenerationOutput]:
        """Runs the pixel generation pipeline for the given inputs."""
        model_inputs, flat_batch = self.prepare_batch(inputs.batch)
        if not flat_batch or model_inputs is None:
            return {}

        if self._use_module:
            assert self._compiled is not None
            assert self._module is not None
            # ``forward`` runs the text encoder, VAE image encoder, the
            # denoising loop, and the VAE decoder end-to-end.  Per-input
            # device placement is handled inside the Module's
            # ``prepare_inputs``.  Input order must match
            # ``FLUXModule.input_types()``.
            try:
                compiled_outputs = self._compiled(
                    model_inputs.tokens,
                    model_inputs.input_image,
                    model_inputs.latents,
                    model_inputs.num_inference_steps,
                    model_inputs.h_carrier,
                    model_inputs.w_carrier,
                    model_inputs.timesteps,
                    model_inputs.dts,
                    model_inputs.guidance,
                    model_inputs.text_ids,
                    model_inputs.latent_image_ids,
                )
            except Exception:
                _logger.error(
                    "Encountered an exception while executing pixel "
                    "batch (module path, denoise loop): batch_size=%d",
                    len(flat_batch),
                )
                raise
            module_outputs = self._module.from_outputs(list(compiled_outputs))
            images = np.from_dlpack(module_outputs.images)
            num_images_per_prompt = np.from_dlpack(
                model_inputs.num_images_per_prompt
            ).item()
            assert isinstance(num_images_per_prompt, int)
        elif self._use_executor:
            assert self._executor is not None
            try:
                executor_outputs = self._executor.execute(model_inputs)
            except Exception:
                _logger.error(
                    "Encountered an exception while executing pixel "
                    "batch (executor path): batch_size=%d",
                    len(flat_batch),
                )
                raise
            images = np.from_dlpack(executor_outputs.images)
            num_images_per_prompt = np.from_dlpack(
                model_inputs.num_images_per_prompt
            ).item()
            assert isinstance(num_images_per_prompt, int)
        else:
            assert self._pipeline_model is not None
            try:
                model_outputs = self._pipeline_model.execute(
                    model_inputs=model_inputs
                )
            except Exception:
                _logger.error(
                    "Encountered an exception while executing pixel "
                    "batch: batch_size=%d, num_images_per_prompt=%s, "
                    "height=%s, width=%s, num_inference_steps=%s",
                    len(flat_batch),
                    model_inputs.num_images_per_prompt,
                    model_inputs.height,
                    model_inputs.width,
                    model_inputs.num_inference_steps,
                )
                raise
            images = model_outputs.images
            num_images_per_prompt = model_inputs.num_images_per_prompt

        expected_images = len(flat_batch) * num_images_per_prompt

        # Video output: shape [B, C, T, H, W]
        if isinstance(images, np.ndarray) and images.ndim == 5:
            video_responses: dict[RequestID, GenerationOutput] = {}
            for index, (request_id, _context) in enumerate(flat_batch):
                # video_clip shape: [C, T, H, W]
                video_clip = images[index]
                # Video decoders are expected to return uint8 pixel values.
                # Reorder to [T, H, W, C] and keep raw frames until the serving
                # layer decides how to encode them for the final response.
                frames = np.transpose(video_clip, (1, 2, 3, 0))
                video_responses[request_id] = GenerationOutput(
                    request_id=request_id,
                    final_status=GenerationStatus.END_OF_SEQUENCE,
                    output=[OutputVideoContent.from_numpy_frames(frames)],
                )
            return video_responses

        if images.shape[0] != expected_images:
            raise ValueError(
                "Unexpected number of images returned from pipeline: "
                f"expected {expected_images}, got {images.shape[0]}."
            )

        responses: dict[RequestID, GenerationOutput] = {}
        for index, (request_id, _context) in enumerate(flat_batch):
            offset = index * num_images_per_prompt
            pixel_data = images[offset : offset + num_images_per_prompt]

            output_format = getattr(_context, "output_format", "jpeg")
            # Per the unified usage spec, image generation keeps token counts
            # at 0; billing-relevant metadata (dimensions, megapixels, steps,
            # image count) lives under image_generation_details, measured from
            # the actual output arrays rather than the requested dimensions.
            usage = Usage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                image_generation_details=ImageGenerationDetails.from_images(
                    pixel_data, steps=_context.num_inference_steps
                ),
            )
            responses[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=[
                    OutputImageContent.from_numpy(img, format=output_format)
                    for img in pixel_data
                ],
                usage=usage,
            )

        return responses

    def prepare_batch(
        self,
        batch: dict[RequestID, PixelGenerationContextType],
    ) -> tuple[
        Any,
        list[tuple[RequestID, PixelGenerationContextType]],
    ]:
        """Prepare model inputs for pixel generation execution.

        Delegates to the pipeline model for model-specific input preparation.

        Args:
            batch: Dictionary mapping request IDs to their PixelContext objects.

        Returns:
            A tuple of:
                - Model inputs ready for execution, or None if batch is empty.
                - list: Flattened batch as (request_id, context) tuples for
                  response mapping.

        Raises:
            ValueError: If batch size is larger than 1 (not yet supported).
        """
        if not batch:
            return None, []

        # Flatten batch to list of (request_id, context) tuples
        flat_batch = list(batch.items())

        if len(flat_batch) > 1:
            raise ValueError(
                "Batching of different requests is not supported yet."
            )

        if self._use_module:
            assert self._module is not None
            contexts = [ctx for _rid, ctx in flat_batch]
            model_inputs = self._module.prepare_inputs(contexts)
        elif self._use_executor:
            assert self._executor is not None
            contexts = [ctx for _rid, ctx in flat_batch]
            model_inputs = self._executor.prepare_inputs(contexts)
        else:
            assert self._pipeline_model is not None
            model_inputs = self._pipeline_model.prepare_inputs(flat_batch[0][1])
        return model_inputs, flat_batch

    def release(self, request_id: RequestID) -> None:
        """Release resources associated with a request.

        Args:
            request_id: The request ID to release resources for.
        """
