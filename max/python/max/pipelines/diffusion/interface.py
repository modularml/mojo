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

"""Pipeline utilities for MAX-optimized diffusion pipelines."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, overload

import numpy as np
import numpy.typing as npt
from max._core.driver import Device
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import Graph, TensorType
from max.graph.weights import load_weights
from max.pipelines.context import PixelContext
from max.pipelines.modeling.base.component_model import ComponentModel
from tqdm import tqdm

from .cache import DenoisingCacheState
from .config import DEFAULT_DENOISING_CACHE_CONFIG, DenoisingCacheConfig
from .first_block_cache import FirstBlockCache
from .taylorseer import TaylorSeer, run_denoising_step

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

_logger = logging.getLogger("max.pipelines")

_CompileTarget: TypeAlias = Callable[..., Any] | Module[..., Any]
_CompileDecorator: TypeAlias = Callable[[_CompileTarget], "CompileWrapper"]


@dataclass
class DiffusionPipelineOutput:
    """Output of a diffusion pipeline."""

    images: npt.NDArray[np.uint8]
    """NHWC uint8 NumPy array of shape (B, H, W, C) with values in [0, 255]."""


class DiffusionPipeline(ABC):
    """Base class for diffusion pipelines.

    Subclasses must define `components` mapping component names to ComponentModel types.
    """

    components: dict[str, type[ComponentModel]] | None = None

    unprefixed_weight_component: str | None = None
    """When set, weight files without a ``<component>/`` prefix are assigned to
    this component.  This supports multi-repo layouts where quantized weights
    for one component (e.g. the transformer) are shipped as flat files in a
    separate repo while the remaining components use the base model repo."""

    default_num_inference_steps: int = 50
    """Default number of denoising steps when the user does not specify one.

    Subclasses may override this to provide a model-appropriate default.
    """

    default_residual_threshold: float = 0.05
    """Model-specific default for the FBCache relative difference threshold.

    Subclasses may override this to provide a model-appropriate default.
    Used when the request does not specify a ``residual_threshold``.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        weight_paths: list[Path],
        cache_config: DenoisingCacheConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.cache_config: DenoisingCacheConfig = (
            cache_config
            if cache_config is not None
            else DEFAULT_DENOISING_CACHE_CONFIG
        )
        self.pipeline_config = pipeline_config
        self.session = session
        self.devices = devices

        for name, model in self._load_sub_models(weight_paths).items():
            setattr(self, name, model)

        self.init_remaining_components()

    @abstractmethod
    def init_remaining_components(self) -> None:
        """Initialize non-ComponentModel components (e.g., image processors)."""

    @abstractmethod
    def prepare_inputs(self, context: PixelContext) -> Any:
        """Prepare inputs for the pipeline."""
        raise NotImplementedError(
            f"prepare_inputs is not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def execute(
        self, model_inputs: Any, **kwargs: Any
    ) -> DiffusionPipelineOutput:
        """Execute the pipeline with the given model inputs.

        Args:
            model_inputs: Prepared model inputs from prepare_inputs.
            **kwargs: Additional pipeline-specific execution parameters.

        Returns:
            A DiffusionPipelineOutput containing NHWC uint8 images.
        """
        raise NotImplementedError(
            f"execute is not implemented for {self.__class__.__name__}"
        )

    def _load_sub_models(
        self, weight_paths: list[Path]
    ) -> dict[str, ComponentModel]:
        """Load all ComponentModel sub-components defined in `components`.

        Uses per-component ``MAXModelConfig`` instances from the
        ``ModelManifest`` to obtain each component's config, encoding,
        and weight paths.
        """
        if not self.components:
            raise ValueError(
                f"{self.__class__.__name__}.components is not set."
            )

        # Imported here rather than at module scope to break a circular
        # import: ``max.pipelines.lib`` (``registry.py``,
        # ``pipeline_variants/__init__.py``) imports ``PixelGenerationPipeline``
        # from ``diffusion/pipeline.py`` -- a real, necessary dependency, not
        # just a re-export -- so importing from ``max.pipelines.lib.*`` at
        # load time here would re-enter a partially-initialized ``lib``
        # package whenever ``diffusion`` (this package's Bazel target) is
        # what triggers ``lib`` to load first.
        from max.pipelines.lib.config.model_config import (
            _resolve_component_encoding_and_weights,
        )

        models = self.pipeline_config.models
        loaded_sub_models: dict[str, ComponentModel] = {}

        for name, component_cls in tqdm(
            self.components.items(), desc="Loading sub models"
        ):
            if not issubclass(component_cls, ComponentModel):
                continue

            component_config = models.get(name)
            if component_config is None:
                raise ValueError(
                    f"Missing model config for component '{name}' "
                    f"in manifest. Available: {list(models.keys())}"
                )

            config_dict = component_config.huggingface_config.to_dict()
            resolved_encoding, resolved_weight_path = (
                _resolve_component_encoding_and_weights(component_config)
            )
            encoding = resolved_encoding or "bfloat16"
            abs_paths = self._get_component_weight_paths(
                component_config, resolved_weight_path
            )

            init_params = inspect.signature(component_cls.__init__).parameters
            init_kwargs: dict[str, Any] = {
                "config": config_dict,
                "encoding": encoding,
                "devices": self.devices,
                "weights": load_weights(abs_paths),
            }
            if "session" in init_params:
                init_kwargs["session"] = self.session
            if "cache_config" in init_params:
                init_kwargs["cache_config"] = self.cache_config

            loaded_sub_models[name] = component_cls(**init_kwargs)

        return loaded_sub_models

    def _get_component_weight_paths(
        self, component_config: Any, weight_path: list[Path]
    ) -> list[Path]:
        """Resolve absolute weight paths for a single component.

        Args:
            component_config: The component's own ``MAXModelConfig``.
            weight_path: The component's resolved weight path (see
                :func:`_resolve_component_encoding_and_weights`).
        """
        return component_config.resolved_weight_paths(weight_path)

    # -----------------------------------------------------------------
    # Denoising cache support (FBCache + TaylorSeer)
    # -----------------------------------------------------------------

    _taylorseer: TaylorSeer | None = None
    _fbc: FirstBlockCache | None = None
    _cache_dtype: DType
    _cache_device: Device

    def _init_cache_state(self, dtype: DType, device: Device) -> None:
        """Initialize pipeline-level cache tensors and TaylorSeer graphs.

        Call once during ``init_remaining_components()``, after the
        transformer has been loaded and compiled.
        """
        self._taylorseer = None
        if self.cache_config.taylorseer:
            self._taylorseer = TaylorSeer(
                max_order=self.cache_config.taylorseer_max_order,
                dtype=dtype,
                device=device,
            )

        self._fbc = None
        if self.cache_config.first_block_caching:
            self._fbc = FirstBlockCache(dtype=dtype, device=device)

        self._cache_dtype = dtype
        self._cache_device = device

    def create_cache_state(
        self,
        batch_size: int,
        seq_len: int,
        transformer_config: Any,
        text_seq_len: int = 0,
    ) -> DenoisingCacheState:
        """Create per-request cache state with fresh tensors.

        Args:
            batch_size: Batch dimension (from prompt_embeds).
            seq_len: Sequence length (from latents).
            transformer_config: Transformer config carrying dimension info.
                Must have ``num_attention_heads``, ``attention_head_dim``,
                ``patch_size``, ``out_channels``, and ``in_channels`` attributes.
            text_seq_len: Text sequence length. Reserved for cache modes that
                require text-aware allocations.
        """
        for attr in (
            "num_attention_heads",
            "attention_head_dim",
            "patch_size",
            "out_channels",
            "in_channels",
        ):
            assert hasattr(transformer_config, attr), (
                f"transformer_config missing required attribute '{attr}'"
            )

        residual_dim = (
            transformer_config.num_attention_heads
            * transformer_config.attention_head_dim
        )
        output_dim = (
            transformer_config.patch_size
            * transformer_config.patch_size
            * (
                transformer_config.out_channels
                or transformer_config.in_channels
            )
        )

        state = DenoisingCacheState()

        if self.cache_config.first_block_caching:
            assert self._fbc is not None
            fbc_state = self._fbc.create_state(
                batch_size, seq_len, residual_dim, output_dim
            )
            state.prev_residual = fbc_state.prev_residual
            state.prev_output = fbc_state.prev_output

        if self.cache_config.taylorseer:
            assert self._taylorseer is not None
            ts_state = self._taylorseer.create_state(
                batch_size, seq_len, output_dim
            )
            state.taylor_factor_0 = ts_state.factor_0
            state.taylor_factor_1 = ts_state.factor_1
            state.taylor_factor_2 = ts_state.factor_2

        return state

    def run_transformer(
        self,
        cache_state: DenoisingCacheState,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        """Run the transformer for one denoising step.

        Subclasses must override this to call their transformer with the
        appropriate model-specific arguments.  The method should return
        ``(noise_pred,)`` when first_block_caching is disabled, or
        ``(new_residual, noise_pred)`` when first_block_caching is enabled.

        Args:
            cache_state: Per-request mutable cache state for this stream.
            **kwargs: Model-specific arguments forwarded from
                ``run_denoising_step``.
        """
        raise NotImplementedError

    def run_denoising_step(
        self,
        step: int,
        cache_state: DenoisingCacheState,
        device: Device,
        **kwargs: Any,
    ) -> Tensor:
        """Execute one denoising step with caching logic.

        Delegates the actual transformer call to ``self.run_transformer()``,
        which subclasses override with model-specific arguments.

        Args:
            step: Current step index.
            cache_state: Per-request mutable cache state for this stream.
            device: Target device.
            **kwargs: Model-specific arguments forwarded to
                ``run_transformer``.

        Returns:
            noise_pred tensor for this step.
        """
        return run_denoising_step(
            step=step,
            cache_state=cache_state,
            cache_config=self.cache_config,
            device=device,
            compute_fn=lambda: self.run_transformer(cache_state, **kwargs),
            taylorseer=self._taylorseer,
        )

    def _resolve_absolute_paths(
        self, weight_paths: list[Path], relative_paths: list[str]
    ) -> list[Path]:
        """Match relative component paths to absolute weight paths."""
        absolute_paths = [
            abs_path
            for abs_path in weight_paths
            for rel_path in relative_paths
            if rel_path in str(abs_path)
        ]

        if not absolute_paths:
            raise ValueError(f"Component weights not found: {relative_paths}")
        return absolute_paths


class CompileWrapper:
    """Wraps a compile target with optional input type annotations."""

    def __init__(
        self,
        compile_target: _CompileTarget,
        input_types: Iterable[TensorType] | None = None,
    ) -> None:
        """Initialize the CompileWrapper.

        Args:
            compile_target: The function or module to be compiled.
            input_types: A list of input types (TensorTypes) required for compilation.

        Raises:
            ValueError: If input_types is not provided.
        """
        target_name = getattr(
            compile_target, "__name__", type(compile_target).__name__
        )
        if input_types is None:
            raise ValueError(
                f"input_types must be provided for compilation of {target_name}."
            )

        input_types_tuple = tuple(input_types)
        self._compiled_model: Model | None = None
        self._compiled_module = None

        if isinstance(compile_target, Module):
            self._compiled_module = compile_target.compile(*input_types_tuple)
            return

        with Graph(
            compile_target.__name__, input_types=input_types_tuple
        ) as graph:
            output = compile_target(*graph.inputs)
            if isinstance(output, Iterable):
                graph.output(*output)
            else:
                graph.output(output)
            compiled_graph = graph

        device: CPU | Accelerator
        if any(input_type.device.is_gpu() for input_type in input_types_tuple):
            device = Accelerator()
        else:
            device = CPU()
        session = InferenceSession([device])
        self._compiled_model = session.load(compiled_graph)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled session with the given arguments.

        Args:
            *args: Positional arguments to pass to the session.
            **kwargs: Keyword arguments to pass to the session.

        Returns:
            The result of the session execution.
        """
        if self._compiled_module is not None:
            return self._compiled_module(*args, **kwargs)

        if self._compiled_model is None:
            raise RuntimeError("CompileWrapper has no compiled target.")

        normalized_args = tuple(self._unwrap_tensor(arg) for arg in args)
        normalized_kwargs = {
            key: self._unwrap_tensor(val) for key, val in kwargs.items()
        }
        buffers = self._compiled_model(*normalized_args, **normalized_kwargs)
        outputs = [Tensor.from_dlpack(buffer) for buffer in buffers]
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def _unwrap_tensor(value: Any) -> Any:
        try:
            if hasattr(value, "driver_tensor"):
                return value.driver_tensor
            return value
        except TypeError:
            return value


@overload
def max_compile(
    compile_target: _CompileTarget,
    input_types: Iterable[TensorType] | None = ...,
) -> CompileWrapper: ...


@overload
def max_compile(
    compile_target: None = ...,
    input_types: Iterable[TensorType] | None = ...,
) -> _CompileDecorator: ...


def max_compile(
    compile_target: _CompileTarget | None = None,
    input_types: Iterable[TensorType] | None = None,
) -> _CompileDecorator | CompileWrapper:
    """Decorator or function to compile a target with specified input types.

    Args:
        compile_target: The function or module to compile. If None, returns a decorator.
        input_types: The input types for the compilation.

    Returns:
        A CompileWrapper instance if compile_target is provided, otherwise a decorator.
    """
    if compile_target is None:

        def decorator(f: _CompileTarget) -> CompileWrapper:
            return CompileWrapper(f, input_types)

        return decorator

    return CompileWrapper(compile_target, input_types)
