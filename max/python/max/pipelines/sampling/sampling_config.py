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
"""MAX sampling configuration."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from max.config import ConfigFileModel
from max.dtype import DType
from max.pipelines.context import SamplingParamsGenerationConfigDefaults
from pydantic import BeforeValidator, ConfigDict, Field, PrivateAttr

_logger = logging.getLogger("max.pipelines")


def _coerce_dtype(value: Any) -> DType | Any:
    """Coerce string values to DType enum members.

    DType is a C++ enum with integer values, so Pydantic cannot natively
    coerce YAML/CLI strings like ``"float32"`` into :class:`DType` members.
    This validator handles case-insensitive name matching.
    """
    if isinstance(value, DType):
        return value
    if isinstance(value, str):
        value_cf = value.casefold()
        for member in DType:
            if member.name.casefold() == value_cf:
                return member
    return value


_CoercedDType = Annotated[DType, BeforeValidator(_coerce_dtype)]

# Global default structured-output backend, used when neither the user nor the
# resolved architecture specifies one. Single source of truth for the fallback
# in ``PipelineConfig._resolve_default_structured_output_backend`` and
# ``StructuredOutputHelper.from_tokenizer``.
DEFAULT_STRUCTURED_OUTPUT_BACKEND = "xgrammar"

# Global default for whitespace-tolerant structured-output grammars, used when
# neither the user nor the resolved architecture specifies one. False (compact
# JSON, no whitespace between tokens) is the Gemma-4 runaway mitigation from
# 0c57a6bd331; flipping it is a product decision, not a per-model tweak.
DEFAULT_STRUCTURED_OUTPUT_ANY_WHITESPACE = False


class SamplingConfig(ConfigFileModel):
    """Configuration for the sampling stage of token generation."""

    model_config = ConfigDict(frozen=True)

    in_dtype: _CoercedDType = Field(
        default=DType.float32,
        description="The data type of the input tokens.",
    )

    out_dtype: _CoercedDType = Field(
        default=DType.float32,
        description="The data type of the output logits.",
    )

    enable_structured_output: bool = Field(
        default=False,
        description=(
            "Enable structured generation/guided decoding for the server. This "
            "allows the user to pass a JSON schema in the ``response_format`` "
            "field, which the LLM will adhere to."
        ),
    )

    structured_output_backend: str | None = Field(
        default=None,
        description=(
            "Grammar backend for constrained decoding. One of ``xgrammar`` or "
            "``llguidance``. When unset (``None``), resolved at config "
            "construction to the architecture's default if it declares one, "
            "else the global default ``xgrammar``. An explicit value always "
            "wins."
        ),
    )

    structured_output_any_whitespace: bool | None = Field(
        default=None,
        description=(
            "Whether structured-output (``response_format``) grammars accept "
            "whitespace between JSON tokens. ``False`` (the resolved default) "
            "constrains generation to compact JSON -- no whitespace, "
            "``','``/``':'`` separators -- which mitigates runaway generation "
            "on some models but also masks the newline/indentation tokens "
            "models prefer at structural boundaries. ``True`` uses the "
            "grammar engine's whitespace-tolerant JSON. When unset "
            "(``None``), resolved at config construction to the "
            "architecture's default if it declares one, else ``False``. An "
            "explicit value always wins. Tool-call grammars are unaffected."
        ),
    )

    enable_tool_call_constrained_decode: bool = Field(
        default=True,
        description=(
            "Whether tool-call requests are constrained to a server-generated "
            "grammar during decoding. When enabled (the default), a configured "
            "``runtime.tool_parser`` both produces a decode-time grammar and "
            "parses the resulting output. Set to ``False`` to keep the parser "
            "(tool calls are still parsed out of generated text) while skipping "
            "the constrained-decode/bitmask path for tool calls -- useful when "
            "the grammar path is undesirable but tool-call parsing is still "
            "wanted. With this disabled, ``tool_choice=required`` or a named "
            "function can no longer force a tool call. Independent of "
            "``enable_structured_output``, which gates user-supplied "
            "``response_format`` JSON schemas."
        ),
    )

    enable_variable_logits: bool = Field(
        default=False,
        description=(
            "Enable the sampling graph to accept a ragged tensor of different "
            "sequences as inputs, along with their associated ``logit_offsets``. "
            "This is needed to produce additional logits for echo and "
            "speculative decoding purposes."
        ),
    )

    enable_penalties: bool = Field(
        default=False,
        description=(
            "Whether to apply frequency and presence penalties to the model's "
            "output."
        ),
    )

    enable_min_tokens: bool = Field(
        default=False,
        description=(
            "Whether to enable ``min_tokens``, which blocks the model from "
            "generating stopping tokens before the ``min_tokens`` count is reached."
        ),
    )

    sample_on_host: bool = Field(
        default=False,
        description=(
            "Run the token sampler on the host CPU instead of the model "
            "device. The last-token logits are copied device-to-host and "
            "sampling (top-k/argmax) runs on CPU. Default is to sample on "
            "the model device."
        ),
    )

    _config_file_section_name: str = PrivateAttr(default="sampling_config")
    """The section name to use when loading this config from a ConfigFileModel file.
    This is used to differentiate between different config sections in a single
    ConfigFileModel file."""

    @classmethod
    def from_generation_config_sampling_defaults(
        cls,
        sampling_params_defaults: SamplingParamsGenerationConfigDefaults,
        **kwargs,
    ) -> SamplingConfig:
        """Creates a SamplingConfig from generation config defaults and kwargs.

        Inspects the provided defaults to determine if penalty-related or
        min-tokens-related fields are set to non-default values; if so,
        enables the corresponding flags in the result unless already set in
        kwargs.

        Args:
            sampling_params_defaults: The generation config defaults
                containing explicit values for sampling parameters.
            **kwargs: Additional keyword arguments to override or supplement
                the config.

        Returns:
            A new SamplingConfig instance with the appropriate fields set.
        """
        config_kwargs = kwargs.copy()

        gen_config_explicit = sampling_params_defaults.values_to_update
        if config_kwargs.get("enable_penalties", False) is False:
            has_penalties = any(
                field in gen_config_explicit
                and gen_config_explicit[field] not in (None, 0, 1.0)
                for field in [
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                ]
            )
            if has_penalties:
                config_kwargs["enable_penalties"] = True

        if config_kwargs.get("enable_min_tokens", False) is False:
            has_min_tokens = any(
                field in gen_config_explicit
                and gen_config_explicit[field] not in (None, 0)
                for field in ["min_tokens", "min_new_tokens"]
            )
            if has_min_tokens:
                config_kwargs["enable_min_tokens"] = True

        return cls(**config_kwargs)
