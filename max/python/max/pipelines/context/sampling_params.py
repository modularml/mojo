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

"""Defines sampling parameters and generation configuration defaults for MAX pipeline requests."""

from __future__ import annotations

import logging
import math
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any

from .logit_processors_type import LogitsProcessor


def _validate_temperature(value: float, name: str) -> None:
    """Validate that a temperature-style parameter is in ``[0.0, 2.0]``.

    Raises ``ValueError`` rather than ``max.pipelines.context.exceptions.
    InputError``: this keeps the import simple. Aligns with the surrounding
    ``min_p`` / ``top_p`` / ``repetition_penalty`` validations which also
    raise ``ValueError``. The chat-completion route maps ``ValueError`` to
    ``HTTPException(400, detail="Value error.")`` so the descriptive message
    is lost to the client; switch every ``__post_init__`` validation to
    ``InputError`` to fix this.
    """
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise ValueError(f"{name} must be in [0.0, 2.0], was {value}.")


_logger = logging.getLogger("max.pipelines")


@dataclass
class SamplingParamsInput:
    """Input dataclass for creating SamplingParams instances.

    All fields are optional, allowing partial specification with ``None`` values
    indicating "use default". This enables static type checking while maintaining
    the flexibility to specify only the parameters you want to override.
    """

    top_k: int | None = None
    """The number of most probable tokens to keep when sampling. Defaults to ``None`` (use class default)."""

    top_p: float | None = None
    """The cumulative probability threshold for nucleus sampling. Defaults to ``None`` (use class default)."""

    min_p: float | None = None
    """The minimum probability threshold for a token relative to the most likely token. Defaults to ``None`` (use class default)."""

    temperature: float | None = None
    """The temperature for controlling output randomness. Defaults to ``None`` (use class default)."""

    thinking_temperature: float | None = None
    """Temperature override for tokens inside a ``<think>...</think>`` block.
    Requires a configured reasoning parser to resolve boundary token IDs."""

    frequency_penalty: float | None = None
    """The penalty applied proportionally to token frequency in the generated text. Defaults to ``None`` (use class default)."""

    presence_penalty: float | None = None
    """The flat penalty applied to tokens that have appeared at least once. Defaults to ``None`` (use class default)."""

    repetition_penalty: float | None = None
    """The factor by which logits of repeated tokens are divided. Defaults to ``None`` (use class default)."""

    max_new_tokens: int | None = None
    """The maximum number of tokens to generate. Defaults to ``None`` (use class default)."""

    min_new_tokens: int | None = None
    """The minimum number of tokens to generate before stopping. Defaults to ``None`` (use class default)."""

    ignore_eos: bool | None = None
    """Whether to continue generating past end-of-sequence tokens. Defaults to ``None`` (use class default)."""

    stop: list[str] | None = None
    """A list of strings that, when generated, will stop the generation. Defaults to ``None`` (use class default)."""

    stop_token_ids: list[int] | None = None
    """A list of token IDs that, when generated, will stop the generation. Defaults to ``None`` (use class default)."""

    detokenize: bool | None = None
    """Whether to convert output token IDs back to text. Defaults to ``None`` (use class default)."""

    seed: int | None = None
    """The random seed for reproducible sampling. Defaults to ``None`` (use class default)."""

    logits_processors: Sequence[LogitsProcessor] | None = None
    """Callables applied to model logits before sampling. Defaults to ``None`` (use class default)."""


@dataclass(frozen=True)
class SamplingParamsGenerationConfigDefaults:
    """Default sampling parameter values extracted from a model's GenerationConfig.

    This class encapsulates sampling parameter defaults that come from a HuggingFace
    model's GenerationConfig. These defaults have middle priority when creating
    SamplingParams instances:

    Priority order (highest to lowest):
    1. User-provided values (SamplingParamsInput)
    2. Model's GenerationConfig values (this class)
    3. SamplingParams class defaults

    All fields default to None, indicating that the model's GenerationConfig does not
    explicitly set that parameter. When None, SamplingParams will fall back to its
    own class defaults.

    .. code-block:: python

        from max.pipelines.context.sampling_params import (
            SamplingParams,
            SamplingParamsGenerationConfigDefaults,
            SamplingParamsInput,
        )

        defaults = SamplingParamsGenerationConfigDefaults(
            temperature=0.7,
            top_k=50,
            max_new_tokens=512,
        )
        params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(),
            sampling_params_defaults=defaults,
        )
    """

    temperature: float | None = None
    """Temperature value from the model's GenerationConfig, if explicitly set."""

    top_p: float | None = None
    """Top-p (nucleus sampling) value from the model's GenerationConfig, if explicitly set."""

    top_k: int | None = None
    """Top-k sampling value from the model's GenerationConfig, if explicitly set."""

    repetition_penalty: float | None = None
    """Repetition penalty value from the model's GenerationConfig, if explicitly set."""

    max_new_tokens: int | None = None
    """Maximum number of new tokens from the model's GenerationConfig, if explicitly set."""

    min_new_tokens: int | None = None
    """Minimum number of new tokens from the model's GenerationConfig, if explicitly set."""

    do_sample: bool | None = None
    """If ``False``, uses greedy sampling."""

    eos_token_id: int | list[int] | None = None
    """EOS token ID from the model's GenerationConfig, if explicitly set."""

    @cached_property
    def _values_to_update(self) -> dict[str, float | int | list[int]]:
        values: dict[str, float | int | list[int]] = {}
        for _field in fields(self):
            field_value = getattr(self, _field.name)
            if field_value is not None:
                if _field.name == "eos_token_id":
                    if isinstance(field_value, int):
                        values["stop_token_ids"] = [field_value]
                    elif isinstance(field_value, list):
                        values["stop_token_ids"] = field_value
                else:
                    values[_field.name] = field_value
        return values

    @property
    def values_to_update(self) -> dict[str, float | int | list[int]]:
        """Non-``None`` field values as a dictionary.

        Returns:
            A dictionary mapping field names to their values, excluding any fields
            that are ``None``. This dictionary can be used to update
            :class:`SamplingParams` default values.

        .. code-block:: python

            from max.pipelines.context.sampling_params import (
                SamplingParamsGenerationConfigDefaults,
            )

            defaults = SamplingParamsGenerationConfigDefaults(
                temperature=0.7,
                top_k=50,
            )
            defaults.values_to_update
            # {'temperature': 0.7, 'top_k': 50}
        """
        # Return a copy of the values, because the caller mutates the returned dict.
        return self._values_to_update.copy()


@dataclass(frozen=False)
class SamplingParams:
    """Request specific sampling parameters that are only known at run time."""

    top_k: int = -1
    """Limits the sampling to the K most probable tokens. This defaults to -1 (to sample all tokens), for greedy sampling set to 1."""

    top_p: float = 1
    """Only use the tokens whose cumulative probability is within the top_p threshold. This applies to the top_k tokens."""

    min_p: float = 0.0
    """Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this."""

    temperature: float = 1
    """Controls the randomness of the model's output; higher values produce more diverse responses.
    For greedy sampling, set to temperature to 0."""

    thinking_temperature: float | None = None
    """Temperature override for tokens inside a ``<think>...</think>`` block.
    ``None`` falls back to ``temperature``. Requires a configured reasoning
    parser; ignored otherwise."""

    frequency_penalty: float = 0.0
    """The frequency penalty to apply to the model's output. A positive value will penalize new tokens
    based on their frequency in the generated text: tokens will receive a penalty proportional to the
    count of appearances."""

    presence_penalty: float = 0.0
    """The presence penalty to apply to the model's output. A positive value will penalize new tokens
    that have already appeared in the generated text at least once by applying a constant penalty."""

    repetition_penalty: float = 1.0
    """The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens
    that have already appeared in the generated text at least once by dividing the logits by the
    repetition penalty."""

    max_new_tokens: int | None = None
    """The maximum number of new tokens to generate in the response.

    When set to an integer value, generation will stop after this many tokens.
    When ``None`` (default), the model may generate tokens until it reaches its
    internal limits or other stopping criteria are met.
    """

    min_new_tokens: int = 0
    """The minimum number of tokens to generate in the response."""

    ignore_eos: bool = False
    """If ``True``, the response will ignore the EOS token, and continue to
    generate until the max tokens or a stop string is hit."""

    stop: list[str] | None = None
    """A list of detokenized sequences that can be used as stop criteria when generating a new sequence."""

    stop_token_ids: list[int] | None = None
    """A list of token ids that are used as stopping criteria when generating a new sequence."""

    detokenize: bool = True
    """Whether to detokenize the output tokens into text."""

    seed: int = field(default_factory=lambda: secrets.randbits(32))
    """The seed to use for the random number generator. Defaults to a cryptographically secure random value."""

    logits_processors: Sequence[LogitsProcessor] | None = None
    """Callables to post-process the model logits.
    See :obj:`~max.pipelines.modeling.types.logit_processors_type.LogitsProcessor` for examples."""

    @classmethod
    def from_input_and_generation_config(
        cls,
        input_params: SamplingParamsInput,
        sampling_params_defaults: SamplingParamsGenerationConfigDefaults,
    ) -> SamplingParams:
        """Creates a :class:`SamplingParams` instance with defaults from a HuggingFace GenerationConfig.

        Combines three sources of values in priority order (highest to lowest):

        1. User-provided values in ``input_params`` (non-``None``)
        2. Model's GenerationConfig values (only if explicitly set in the model's config)
        3. :class:`SamplingParams` class defaults

        Args:
            input_params: Dataclass containing user-specified parameter values.
                Values of ``None`` will be replaced with model defaults or class defaults.
            sampling_params_defaults: :class:`SamplingParamsGenerationConfigDefaults`
                containing default sampling parameters extracted from the model's
                GenerationConfig.

        Returns:
            A new :class:`SamplingParams` instance with model-aware defaults.

        .. code-block:: python

            from max.pipelines.context.sampling_params import (
                SamplingParams,
                SamplingParamsGenerationConfigDefaults,
                SamplingParamsInput,
            )

            sampling_params_defaults = SamplingParamsGenerationConfigDefaults(
                top_k=50,
            )
            params = SamplingParams.from_input_and_generation_config(
                SamplingParamsInput(temperature=0.7),
                sampling_params_defaults=sampling_params_defaults,
            )
        """
        # Start with model's generation config values (only if explicitly set)
        defaults: dict[str, Any] = sampling_params_defaults.values_to_update

        # Handle special mappings from GenerationConfig
        if "do_sample" in defaults and defaults["do_sample"] is not None:
            # If do_sample is False, set greedy defaults (unless user overrides)
            if not defaults["do_sample"]:
                defaults["temperature"] = 0
                defaults["top_k"] = 1

            # This isn't included in SamplingParams, therefore we should remove it.
            del defaults["do_sample"]

        # Overlay user-provided values (highest priority)
        for _field in fields(input_params):
            value = getattr(input_params, _field.name)
            if value is not None:
                defaults[_field.name] = value

        return cls(**defaults)

    def log_sampling_info(self) -> None:
        """Logs comprehensive sampling parameters information.

        Displays all sampling parameters in a consistent visual format similar to
        pipeline configuration logging.
        """
        _logger.info("Sampling Config")
        _logger.info("=" * 60)

        # Core sampling parameters
        _logger.info(f"    top_k                  : {self.top_k}")
        _logger.info(f"    top_p                  : {self.top_p}")
        _logger.info(f"    min_p                  : {self.min_p}")
        _logger.info(f"    temperature            : {self.temperature}")

        # Penalty parameters
        _logger.info(f"    frequency_penalty      : {self.frequency_penalty}")
        _logger.info(f"    presence_penalty       : {self.presence_penalty}")
        _logger.info(f"    repetition_penalty     : {self.repetition_penalty}")

        # Generation control parameters
        _logger.info(f"    max_new_tokens         : {self.max_new_tokens}")
        _logger.info(f"    min_new_tokens         : {self.min_new_tokens}")
        _logger.info(f"    ignore_eos             : {self.ignore_eos}")
        _logger.info(f"    detokenize             : {self.detokenize}")

        # Stopping criteria
        if self.stop:
            stop_str = ", ".join(f'"{s}"' for s in self.stop)
            _logger.info(f"    stop_strings           : [{stop_str}]")
        else:
            _logger.info("    stop_strings           : None")

        if self.stop_token_ids:
            stop_ids_str = ", ".join(str(id) for id in self.stop_token_ids)
            _logger.info(f"    stop_token_ids         : [{stop_ids_str}]")
        else:
            _logger.info("    stop_token_ids         : None")
        _logger.info("")

    def __post_init__(self):
        # Normalize None values to field defaults. These can arrive when a
        # user explicitly sends top_p=null or top_k=null in a request.
        # Treating None as "use the default" keeps the fast gumbel-sampling
        # path reachable and avoids NaN propagation through numpy.
        _defaults = self.__dataclass_fields__
        if self.top_k is None:
            self.top_k = _defaults["top_k"].default
        if self.top_p is None:
            self.top_p = _defaults["top_p"].default

        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError("min_p must be in [0.0, 1.0]")

        # repetition_penalty divides the logits in the sampling kernel, so a
        # value <= 0 or non-finite yields Inf/NaN logits that wedge the
        # request like a bad temperature. ``NaN <= 0`` / ``inf <= 0`` are both
        # False, so the bound check alone lets non-finite values through --
        # gate on isfinite first.
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError(
                "repetition_penalty must be a finite value greater than 0, "
                f"was {self.repetition_penalty}."
            )

        if self.top_k == 0:
            self.top_k = -1
        if self.top_k < -1:
            raise ValueError(
                f"top_k must be -1 or greater than 0, was {self.top_k}."
            )

        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], was {self.top_p}.")

        # Temperature is divided into logits in the sampling kernel; a
        # negative or non-finite value produces NaN/Inf logits that
        # propagate through top-k/top-p and yield out-of-range token
        # ids. Under the overlap scheduler the resulting hung request
        # keeps the in-flight batch slot alive forever, wedging every
        # subsequent request. Extreme positive values (e.g. 100) make
        # the distribution near-uniform; the model almost never samples
        # EOS and the request runs out the token window before
        # returning -- the client sees a timeout. Pin to OpenAI's
        # documented range so the bad value never reaches the GPU.
        _validate_temperature(self.temperature, "temperature")
        if self.thinking_temperature is not None:
            _validate_temperature(
                self.thinking_temperature, "thinking_temperature"
            )

        # If temperature is 0, set top_k to 1.
        if self.temperature == 0:
            _logger.debug("Temperature is 0, overriding top_k to 1.")
            self.top_k = 1

        # Penalties are added to / subtracted from logits before sampling; a
        # non-finite value yields NaN/Inf logits that wedge the request the
        # same way a bad temperature does. Pin to OpenAI's documented range.
        for _penalty_name, _penalty in (
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ):
            if not math.isfinite(_penalty) or not -2.0 <= _penalty <= 2.0:
                raise ValueError(
                    f"{_penalty_name} must be in [-2.0, 2.0], was {_penalty}."
                )

        # ``bool`` is an ``int`` subclass in Python, so a JSON ``true`` for
        # max_tokens/min_tokens would silently coerce to 1 and truncate the
        # response; reject it explicitly. Negative counts underflow the step
        # budget. (``max_new_tokens=None`` means "use the model's limit".)
        if isinstance(self.max_new_tokens, bool) or (
            self.max_new_tokens is not None and self.max_new_tokens < 0
        ):
            raise ValueError(
                "max_new_tokens must be a non-negative integer, was "
                f"{self.max_new_tokens!r}."
            )
        if isinstance(self.min_new_tokens, bool) or self.min_new_tokens < 0:
            raise ValueError(
                "min_new_tokens must be a non-negative integer, was "
                f"{self.min_new_tokens!r}."
            )

    @property
    def needs_penalties(self) -> bool:
        """Whether penalties are needed for the set of sampling parameters."""
        return (
            self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
            or self.repetition_penalty != 1.0
        )
