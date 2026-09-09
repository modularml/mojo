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
"""Top-level pipeline dispatch helpers for cascade.

Selects and builds the correct :class:`GenAIPipeline` for a
:class:`PipelineArgs` by constructing a :class:`PipelineConfig` via
:meth:`PipelineConfig.from_args`, resolving the model's architecture against
the MAX :obj:`~max.pipelines.PIPELINE_REGISTRY`, and building the cascade
pipeline class the architecture declares
(:attr:`~max.pipelines.lib.registry.SupportedArchitecture.cascade_pipeline_factory`).
There is no task-based routing: the resulting pipeline's interfaces (see
``serve.all_routes``) determine which HTTP routes it serves.

The ``dummy_textgen`` and ``dummy_imgen`` model paths are in-process test
fixtures with no Hugging Face config, so they are selected by an exact
model-path match on the raw arguments and never construct a config or touch
the registry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from max.experimental.cascade.interfaces.pipeline import GenAIPipeline
from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
)
from max.experimental.cascade.pipelines.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)
from max.experimental.cascade.pipelines.echo_textgen import EchoTextGenPipeline
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineArgs
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.registry import SupportedArchitecture

# Dummy pipelines are in-process test fixtures selected by an exact model-path
# sentinel; they have no Hugging Face config and never hit the registry.
_DUMMY_BUILDERS: dict[str, Callable[[], Awaitable[GenAIPipeline]]] = {
    "dummy_textgen": build_dummy_textgen_pipeline,
    "dummy_imgen": build_dummy_imgen_pipeline,
}

# An ``echo:<repo-id>`` model path selects the echo text-gen pipeline: the real
# tokenizer for ``<repo-id>`` paired with a token-echoing worker in place of the
# model. Like the dummy sentinels, selection lives entirely in the model path,
# so no separate flag threads through the serve CLI and dispatcher.
_ECHO_PREFIX = "echo:"


def count_unique_device_specs(args: PipelineArgs) -> int:
    """Count the unique GPU ``device_specs`` sets across the pipeline's models.

    Each distinct set of GPU device IDs is treated as needing its own worker
    process; models sharing the same GPUs share a process. Used by the serve
    CLI to size the ``gpu`` worker pool from the raw model arguments.

    Args:
        args: The pipeline arguments whose model device specs to inspect.

    Returns:
        The number of unique GPU ``device_specs`` sets, at least 1.
    """
    per_model_specs = [args.device_specs]
    if args.draft_model is not None:
        per_model_specs.append(args.draft_model.device_specs)
    unique: set[frozenset[tuple[str, int]]] = set()
    for specs in per_model_specs:
        gpu_specs = frozenset(
            (spec.device_type, spec.id)
            for spec in specs
            if spec.device_type == "gpu"
        )
        if gpu_specs:
            unique.add(gpu_specs)
    return max(1, len(unique))


def _resolve_architecture(config: PipelineConfig) -> SupportedArchitecture:
    """Resolve the ``SupportedArchitecture`` for *config*'s main model.

    Registers the built-in architectures (idempotent), reads the architecture
    class name from the model's Hugging Face config, and looks it up in the
    MAX registry.

    Args:
        config: Fully-specified ``PipelineConfig`` for a real (non-dummy) model.

    Raises:
        ValueError: If the architecture is unknown to the MAX registry.
    """
    architecture_name = config.models.main_architecture_name
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        architecture_name=architecture_name,
        prefer_module_v3=config.runtime.prefer_module_v3,
    )
    if arch is None:
        raise ValueError(
            f"No MAX architecture found for {architecture_name!r}."
        )
    return arch


async def build_pipeline(args: PipelineArgs) -> GenAIPipeline:
    """Build the cascade pipeline described by *args*.

    Selection is driven entirely by the model path, read from the raw
    arguments. Dummy fixtures (``dummy_textgen`` / ``dummy_imgen``) are
    matched by exact model path and need no resolution. An ``echo:<repo-id>``
    path builds an :class:`EchoTextGenPipeline` for ``<repo-id>``'s real
    tokenizer with the model worker replaced by a token-echoing worker -- no
    config is ever constructed and no weights are downloaded, so it measures
    cascade framework overhead without a model forward pass. Every other model
    is constructed into a :class:`PipelineConfig` via
    :meth:`PipelineConfig.from_args` -- the same construction boundary
    ``max.serve``'s entrypoints use -- then resolved against the MAX
    architectures registry, and its
    :attr:`~max.pipelines.lib.registry.SupportedArchitecture.cascade_pipeline_factory`
    class is constructed from the config.

    Config construction and model-factory construction happen here, exactly
    where ``max.serve``'s API process does them:
    :meth:`PipelineConfig.from_args` builds and validates the config, and
    ``retrieve_factory`` (invoked while the pipeline constructs its workers)
    runs memory planning and returns the picklable factory the model worker
    invokes. That factory (and the tokenizer's eos set) is bound onto any
    :class:`MAXModelWorker` in the pipeline. Callers just pass the raw
    ``args`` and get back a deployable pipeline; the constructed config is
    never mutated again downstream, so workers never construct or resolve it
    themselves.

    Args:
        args: The flat pipeline arguments, typically constructed by cyclopts
            from the serve CLI flags. ``args.tokenizer_impl`` names the
            ``TokenizerWorker`` subclass the text-gen pipelines use, as
            ``"module.path:ClassName"``; ``None`` (the default) uses the
            HuggingFace tokenizer.

    Raises:
        ValueError: If no model is specified.
        NotImplementedError: If the architecture declares no cascade pipeline.
    """
    model_path = args.model_path
    if not model_path:
        raise ValueError(
            "No models specified. Pass a model path via --model-path <repo-id>."
        )

    if dummy_builder := _DUMMY_BUILDERS.get(model_path):
        return await dummy_builder()

    if model_path.startswith(_ECHO_PREFIX):
        return EchoTextGenPipeline(
            model_path.removeprefix(_ECHO_PREFIX), args.tokenizer_impl
        )

    # We need to call register models explicitly instead of having it run when
    # importing max.pipelines.architectures, because we have cascade as a library
    # to be able to be exported to multiple Bazel targets.
    register_all_models()

    config = PipelineConfig.from_args(args)

    arch = _resolve_architecture(config)
    factory = arch.cascade_pipeline_factory
    if factory is None:
        raise NotImplementedError(
            f"Architecture {arch.name!r} (model {model_path!r}) has no cascade "
            "pipeline. Set cascade_pipeline_factory on its "
            "SupportedArchitecture to enable cascade serving."
        )

    if factory is CommonTextGenPipeline:
        return CommonTextGenPipeline(config, args.tokenizer_impl)
    if args.tokenizer_impl is not None:
        raise ValueError(
            f"--tokenizer-impl {args.tokenizer_impl!r} is only supported for "
            f"the common text-generation pipeline, not {arch.name!r}."
        )

    pipeline = factory(config)
    assert isinstance(pipeline, GenAIPipeline)
    return pipeline
