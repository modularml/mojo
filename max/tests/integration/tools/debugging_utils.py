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

import copy
import inspect
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, cast

import torch
from create_pipelines import PIPELINE_ORACLES, GenericOracle
from max import driver, pipelines
from max._entrypoints.cli.entrypoint import configure_cli_logging
from max.driver.buffer import load_max_buffer
from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.experimental.nn import Module as ModuleV3
from max.nn.hooks import PrintHook
from max.nn.layer import Module
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from max.tests.integration.tools.hf_config_overrides import (
    apply_hf_config_override,
    apply_non_strict_load,
    set_config_overrides,
)
from run_models import (
    _detect_hf_flakes,
    get_max_default_encoding,
    get_torch_device,
    maybe_log_hf_downloads,
    run_max_model,
    run_torch_model,
    run_vllm_model,
)
from test_common import torch_print_hook
from test_common.github_utils import github_log_group
from test_common.test_data import MockTextGenerationRequest


@contextmanager
def apply_max_hooks(output_directory: Path | None) -> Iterator[PrintHook]:
    """Create and manage MAX print hooks."""
    hook = PrintHook()
    orig_infer_init: Any = None

    if output_directory is not None:
        orig_infer_init = InferenceSession.__init__

        def _patched_inference_init(
            session_self: InferenceSession, *args: Any, **kwargs: Any
        ) -> None:
            orig_infer_init(session_self, *args, **kwargs)
            session_self.set_debug_print_options(
                style=PrintStyle.BINARY_MAX_CHECKPOINT,
                output_directory=output_directory,
            )

        InferenceSession.__init__ = _patched_inference_init  # type: ignore[method-assign,assignment]

    try:
        yield hook
    finally:
        hook.remove()
        if orig_infer_init is not None:
            InferenceSession.__init__ = orig_infer_init  # type: ignore[method-assign]


@contextmanager
def apply_name_layers_after_state_load(hook: PrintHook) -> Iterator[None]:
    """Wrap Module.load_state_dict to name layers after loading."""
    orig_load = Module.load_state_dict

    def _name_layers_after_load(
        module_self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        result = orig_load(module_self, *args, **kwargs)
        hook.name_layers(module_self)
        return result

    cast(Any, Module).load_state_dict = _name_layers_after_load
    try:
        yield
    finally:
        cast(Any, Module).load_state_dict = orig_load


@contextmanager
def apply_v3_hooks(hook: PrintHook) -> Iterator[None]:
    """Patch max.experimental.nn.Module to fire the given hook during tracing.

    Two patches are applied:
    - ``__call__`` is wrapped so the hook sees every v3 module's inputs and
      outputs during graph tracing inside ``Module.compile()``, mirroring how
      ``_call_with_hooks`` works for legacy ``Layer`` subclasses.
    - ``compile`` is wrapped so layers are named (via ``hook.name_layers_v3``)
      before tracing begins, giving tensors the correct dot-separated
      attribute-path names rather than auto-generated class-name labels.
    """
    orig_call = ModuleV3.__call__
    orig_compile = ModuleV3.compile

    # TODO: MODELS-1208: Remove this once we have a proper v3 print hook.
    def _patched_call(module_self: Any, *args: Any, **kwargs: Any) -> Any:
        outputs = orig_call(module_self, *args, **kwargs)
        hook(module_self, args, kwargs, outputs)
        return outputs

    def _compile_with_naming(
        module_self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        hook.name_layers_v3(cast(ModuleV3[..., Any], module_self))
        return orig_compile(module_self, *args, **kwargs)

    cast(Any, ModuleV3).__call__ = _patched_call
    cast(Any, ModuleV3).compile = _compile_with_naming
    try:
        yield
    finally:
        cast(Any, ModuleV3).__call__ = orig_call
        cast(Any, ModuleV3).compile = orig_compile


@contextmanager
def debug_context(
    *,
    output_directory: Path | None,
    hf_config_overrides: dict[str, Any] | None,
) -> Iterator[None]:
    """Context manager to manage model execution when debugging.

    This context manages:
    1. HuggingFace config overrides to modify the model configuration
    2. Places print hooks for both MAX and Torch models to inspect intermediate tensors
    3. Legacy API (max.nn.layer.Layer): names layers after state dict loading
    4. v3 API (max.experimental.nn.Module): patches Module.__call__ to fire hooks
       during graph tracing, and names layers at the start of Module.compile()
    """
    with ExitStack() as stack:
        if hf_config_overrides is not None:
            stack.enter_context(apply_hf_config_override(hf_config_overrides))
            stack.enter_context(apply_non_strict_load())
        hook = stack.enter_context(apply_max_hooks(output_directory))
        # Legacy API (max.nn.layer.Layer) hooks.
        stack.enter_context(apply_name_layers_after_state_load(hook))
        # v3 API (max.experimental.nn.Module) hooks.
        stack.enter_context(apply_v3_hooks(hook))
        yield


@_detect_hf_flakes
def run_debug_model(
    device_specs: list[driver.DeviceSpec],
    framework_name: str,
    pipeline_name: str,
    output_path: Path,
    encoding_name: pipelines.SupportedEncoding | None = None,
    max_batch_size: int | None = None,
    log_hf_downloads: bool = False,
    num_steps: int = 1,
    prompt: str | None = None,
    images: tuple[str, ...] | None = None,
    hf_config_overrides: dict[str, Any] | None = None,
    prefer_module_v3: bool = False,
) -> None:
    """Run a model with print hooks enabled and write intermediate tensors.

    Intermediate tensors are written to the output directory if specified.
    Config overrides can be applied to both MAX and Torch models via hf_config_overrides.
    Set prefer_module_v3=True to select the ModuleV3 architecture variant when
    one is registered under the ``<arch>_ModuleV3`` key in the pipeline registry.
    """
    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)
    configure_cli_logging(level="INFO")

    if pipeline_name in PIPELINE_ORACLES:
        pipeline_oracle = PIPELINE_ORACLES[pipeline_name]
        if prefer_module_v3 and isinstance(pipeline_oracle, GenericOracle):
            # Shallow-copy to avoid mutating the shared PIPELINE_ORACLES entry,
            # then merge the flag so PipelineConfig picks up prefer_module_v3=True
            # and the registry appends _ModuleV3 to the arch name.
            pipeline_oracle = copy.copy(pipeline_oracle)
            pipeline_oracle.config_params = {
                **pipeline_oracle.config_params,
                "prefer_module_v3": True,
            }
    else:
        pipeline_oracle = GenericOracle(
            model_path=pipeline_name,
            config_params={"prefer_module_v3": True}
            if prefer_module_v3
            else {},
        )

    # Build input based on user-provided prompt and/or images
    if prompt is None and not images:
        inputs = pipeline_oracle.inputs[:1]
    elif images and len(images) > 0:
        inputs = [
            MockTextGenerationRequest.with_images(
                prompt=prompt
                if prompt is not None
                else pipeline_oracle.inputs[0].prompt,
                images=list(images),
            )
        ]
    else:
        inputs = [
            MockTextGenerationRequest.text_only(
                prompt
                if prompt is not None
                else pipeline_oracle.inputs[0].prompt
            )
        ]

    evaluation_batch_size: int | list[int]
    if max_batch_size is None:
        if pipeline_oracle.default_batch_size is None:
            evaluation_batch_size = 1
        else:
            evaluation_batch_size = pipeline_oracle.default_batch_size
    else:
        evaluation_batch_size = max_batch_size

    title = f"{pipeline_name} - {framework_name.upper()} - {encoding_name or 'Default Encoding'}"
    with ExitStack() as stack:
        stack.enter_context(github_log_group(title))
        if framework_name in {"max", "torch"}:
            stack.enter_context(
                debug_context(
                    output_directory=output_path,
                    hf_config_overrides=hf_config_overrides,
                )
            )
        if framework_name == "max":
            if encoding_name is None:
                max_encoding_name = get_max_default_encoding(
                    pipeline_oracle, pipeline_name, device_specs
                )
            else:
                max_encoding_name = encoding_name

            with maybe_log_hf_downloads(log_hf_downloads):
                max_pipeline_and_tokenizer = (
                    pipeline_oracle.create_max_pipeline(
                        encoding=max_encoding_name,
                        device_specs=device_specs,
                    )
                )

            # Print model source file path
            try:
                pipeline_model = (
                    max_pipeline_and_tokenizer.pipeline._pipeline_model
                )
                model_class = type(pipeline_model)
                model_file = inspect.getfile(model_class)
                print(f"\n{'=' * 80}")
                print(
                    f"Model class: {model_class.__module__}.{model_class.__name__}"
                )
                print(f"Model source file: {model_file}")
                print(f"{'=' * 80}\n")
            except Exception as e:
                print(f"Warning: Could not determine model source file: {e}")

            print(f"Running {pipeline_name} model on MAX")
            run_max_model(
                task=pipeline_oracle.task,
                max_pipeline_and_tokenizer=max_pipeline_and_tokenizer,
                inputs=inputs,
                num_steps=num_steps,
                evaluation_batch_size=evaluation_batch_size,
                reference=None,
            )
        elif framework_name == "torch":
            torch_device = get_torch_device(device_specs)
            # For multi-gpu, use auto to handle mapping automatically.
            device: Any = "auto" if len(device_specs) > 1 else torch_device

            with maybe_log_hf_downloads(log_hf_downloads):
                torch_pipeline_and_tokenizer = (
                    pipeline_oracle.create_torch_pipeline(
                        encoding=encoding_name,
                        device=device,
                    )
                )

            # Print model source file path
            try:
                model_class = type(torch_pipeline_and_tokenizer.model)
                model_file = inspect.getfile(model_class)
                print(f"\n{'=' * 80}")
                print(
                    f"Model class: {model_class.__module__}.{model_class.__name__}"
                )
                print(f"Model source file: {model_file}")
                print(f"{'=' * 80}\n")
            except Exception as e:
                print(f"Warning: Could not determine model source file: {e}")

            # Apply HuggingFace config overrides directly to the model config
            if hf_config_overrides:
                set_config_overrides(
                    torch_pipeline_and_tokenizer.model.config,
                    hf_config_overrides,
                    "config",
                )

            export_path = str(output_path) if output_path is not None else None
            hook = torch_print_hook.TorchPrintHook(export_path=export_path)
            hook.name_layers(torch_pipeline_and_tokenizer.model)

            print(f"Running {pipeline_name} model on Torch")
            run_torch_model(
                pipeline_oracle=pipeline_oracle,
                torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
                device=torch_device,
                inputs=inputs,
                num_steps=num_steps,
            )
        elif framework_name == "vllm":
            # vLLM does not support print hooks or intermediate tensors.
            if output_path is not None:
                print(
                    "Warning: vLLM does not export intermediate tensors; "
                    "ignoring --output."
                )
            print(f"Running {pipeline_name} model on vLLM")
            vllm_pipeline = pipeline_oracle.create_vllm_pipeline(
                encoding=encoding_name,
                device_specs=device_specs,
            )
            run_vllm_model(
                pipeline_oracle=pipeline_oracle,
                vllm_pipeline=vllm_pipeline,
                inputs=inputs,
                num_steps=num_steps,
                max_batch_size=max_batch_size,
            )
        else:
            raise NotImplementedError(
                f"Framework {framework_name!r} not implemented"
            )


def load_intermediate_tensors(
    model: str,
    framework: str,
    output_dir: Path = Path("/tmp/intermediate_tensors/torch"),
    device_type: str | None = None,
    encoding_name: pipelines.SupportedEncoding | None = None,
) -> dict[str, torch.Tensor]:
    """Run a Transformers model using Torch with print hooks enabled and return intermediate tensors as a dictionary mapping tensor name to torch.Tensor.

    Args:
        model: Hugging Face model id (e.g., "google/gemma-3-1b-it").
        framework: Framework to run the model on (e.g., "torch", "max").
        dir: Output directory where Torch print hooks will write .pt files; may contain subdirectories.
        device_type: Device selector passed through DevicesOptionType (e.g., "gpu", "cpu", "gpu:0,1,2").
        encoding_name: Optional explicit encoding/dtype (e.g., "bfloat16"). If None, the pipeline default is used.

    Returns:
        Dict keyed by emitted file name (str) with loaded torch.Tensor values.
    """

    run_debug_model(
        pipeline_name=model,
        framework_name=framework,
        output_path=output_dir,
        device_specs=(
            _default_device_specs()
            if device_type is None
            else device_specs_from_normalized_device_handle(
                normalize_device_specs_input(device_type)
            )
        ),
        encoding_name=encoding_name or None,
    )
    tensors_map: dict[str, torch.Tensor] = {}
    if framework == "torch":
        files = sorted(
            output_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime
        )
        for file in files:
            torch_tensor = torch.load(file)
            tensors_map[file.name] = torch_tensor
    elif framework == "max":
        files = sorted(
            output_dir.rglob("*.max"), key=lambda p: p.stat().st_mtime
        )
        for file in files:
            tensor = load_max_buffer(file)
            torch_tensor = torch.from_dlpack(tensor).cpu()
            tensors_map[file.name] = torch_tensor
    else:
        raise ValueError(f"Framework not supported: {framework}")
    return tensors_map


def get_torch_testdata(
    model: str,
    module_name: str,
    output_dir: Path = Path("/tmp/intermediate_tensors/torch"),
    device_type: str | None = None,
    encoding_name: pipelines.SupportedEncoding | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get input and output tensors for a specific module from a torch model run.

    This function runs the model with print hooks enabled and extracts the input
    and output tensors for the specified module. Module naming follows the pattern
    used by TorchPrintHook.name_layers(), where modules are named "model.{module_name}"
    (see torch_print_hook.py).

    Args:
        model: Hugging Face model id (e.g., "google/gemma-3-1b-it").
        module_name: Name of the module to get tensors for (e.g., "encoder.layer.0").
                    The "model." prefix is added automatically. Pass empty string for
                    the top-level model.
        output_dir: Output directory where Torch print hooks write .pt files.
        device_type: Device selector passed through DevicesOptionType (e.g., "gpu", "cpu").
        encoding_name: Optional explicit encoding/dtype (e.g., "bfloat16").

    Returns:
        Tuple of (input_tensor, output_tensor) for the specified module.
        Input tensor corresponds to the first argument passed to the module's forward method.
        Output tensor corresponds to the result of the module's forward method.

    Raises:
        KeyError: If the specified module tensors are not found in the output.
                 The error message will list all available tensor keys.

    Example:
        >>> input_tensor, output_tensor = get_torch_testdata(
        ...     model="gpt2",
        ...     module_name="transformer.h.0",
        ... )
    """
    # Load all intermediate tensors from the torch run
    tensors_map = load_intermediate_tensors(
        model=model,
        framework="torch",
        output_dir=output_dir,
        device_type=device_type,
        encoding_name=encoding_name,
    )

    # Construct the expected module name following TorchPrintHook.name_layers() pattern
    # See torch_print_hook.py line 45: name = f"model.{module_name}" if module_name else "model"
    full_module_name = f"model.{module_name}" if module_name else "model"

    # Look for input and output tensors
    # Forward hooks save tensors with different suffixes for inputs/outputs
    # Typical patterns: {module_name}.input.pt, {module_name}.output.pt
    # or {module_name}.args.0.pt for first input argument
    output_key = f"{full_module_name}.output.pt"
    input_key = f"{full_module_name}.input.pt"

    # Try alternative naming patterns if primary keys not found
    if output_key not in tensors_map:
        # Output might be saved without .output suffix
        output_key = f"{full_module_name}.pt"

    if input_key not in tensors_map:
        # Input might be saved as args.0 (first positional argument)
        input_key = f"{full_module_name}.args.0.pt"

    # Check if we found the tensors
    available_keys = sorted(tensors_map.keys())

    if output_key not in tensors_map:
        raise KeyError(
            f"Output tensor for module '{full_module_name}' not found.\n"
            f"Tried keys: '{full_module_name}.output.pt', '{full_module_name}.pt'\n"
            f"Available tensors: {available_keys}"
        )

    if input_key not in tensors_map:
        raise KeyError(
            f"Input tensor for module '{full_module_name}' not found.\n"
            f"Tried keys: '{full_module_name}.input.pt', '{full_module_name}.args.0.pt'\n"
            f"Available tensors: {available_keys}"
        )

    input_tensor = tensors_map[input_key]
    output_tensor = tensors_map[output_key]

    return input_tensor, output_tensor


__all__ = [
    "apply_max_hooks",
    "apply_name_layers_after_state_load",
    "apply_v3_hooks",
    "debug_context",
    "get_torch_testdata",
    "load_intermediate_tensors",
    "run_debug_model",
]
