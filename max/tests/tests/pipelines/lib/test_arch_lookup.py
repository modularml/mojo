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
"""Tests for the architecture lookup tables."""

from __future__ import annotations

import sys
import types

import pytest
from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib.arch_lookup import ArchLookup, SupportedArchitecture
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.interfaces import (
    ArchConfig,
    ModelInputs,
    ModelOutputs,
)
from max.pipelines.lib.interfaces.pipeline_model import PipelineModel
from max.pipelines.lib.tokenizer import TextTokenizer
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig


class StubPipelineModel(PipelineModel[TextContext]):
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


class StubArchConfig(ArchConfig):
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int = 1,
    ) -> StubArchConfig:
        return cls()

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return 1

    def get_max_seq_len(self) -> int:
        return 1


def make_arch(name: str, task: PipelineTask) -> SupportedArchitecture:
    return SupportedArchitecture(
        name=name,
        task=task,
        example_repo_ids=[],
        default_encoding="bfloat16",
        supported_encodings={"bfloat16"},
        pipeline_model=StubPipelineModel,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=StubArchConfig,
    )


def register_lazy_module(
    monkeypatch: pytest.MonkeyPatch,
    lookup: ArchLookup,
    arch: SupportedArchitecture,
    module_name: str,
) -> None:
    """Registers ``arch`` lazily behind a fake importable module."""
    module = types.ModuleType(module_name)
    monkeypatch.setattr(module, "ARCH", arch, raising=False)
    monkeypatch.setitem(sys.modules, module_name, module)
    lookup.register_lazy(arch.name, module=module_name, symbol="ARCH")


def test_materialize_imports_lazy_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = ArchLookup()
    builtin = make_arch("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)
    register_lazy_module(monkeypatch, lookup, builtin, "fake_builtin_llama")

    resolved = lookup.resolve("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)

    assert resolved is builtin


def test_materialize_preserves_explicit_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eagerly registered arch (e.g. --custom-architectures) wins over the
    lazy built-in with the same name and task."""
    lookup = ArchLookup()
    builtin = make_arch("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)
    register_lazy_module(monkeypatch, lookup, builtin, "fake_builtin_llama")

    custom = make_arch("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)
    lookup.register(custom, allow_override=True)

    resolved = lookup.resolve("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)

    assert resolved is custom


def test_materialize_registers_lazy_arch_with_different_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = ArchLookup()
    builtin = make_arch("LlamaForCausalLM", PipelineTask.EMBEDDINGS_GENERATION)
    register_lazy_module(monkeypatch, lookup, builtin, "fake_builtin_llama")

    custom = make_arch("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)
    lookup.register(custom, allow_override=True)

    assert (
        lookup.resolve("LlamaForCausalLM", PipelineTask.EMBEDDINGS_GENERATION)
        is builtin
    )
    assert (
        lookup.resolve("LlamaForCausalLM", PipelineTask.TEXT_GENERATION)
        is custom
    )
