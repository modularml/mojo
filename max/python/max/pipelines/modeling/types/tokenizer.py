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

"""Defines the :class:`PipelineTokenizer` protocol for language model tokenizers used in MAX pipelines."""

from __future__ import annotations

__all__ = ["PipelineTokenizer", "TokenizerEncoded", "UnboundContextType"]

from typing import Protocol, TypeVar, runtime_checkable

from max.pipelines.request import RequestType

# TODO: Bound this to TextContext, after we've audited the class.
UnboundContextType = TypeVar("UnboundContextType", covariant=True)
TokenizerEncoded = TypeVar("TokenizerEncoded")


@runtime_checkable
class PipelineTokenizer(
    Protocol[UnboundContextType, TokenizerEncoded, RequestType]
):
    """Interface for LLM tokenizers."""

    @property
    def eos_token_ids(self) -> set[int]:
        """The full set of token ids that end generation for this model.

        The tokenizer's declared EOS plus any additional terminators the
        model ends its turn with (for example, chat turn-end tokens from the
        model's generation config).
        """
        ...

    @property
    def expects_content_wrapping(self) -> bool:
        """If ``True``, this tokenizer expects messages to be wrapped as a dict.

        Text messages are formatted as:

        .. code-block:: json

            {
              "role": "user",
              "content": [{ "type": "text", "text": "text content" }]
            }

        instead of:

        .. code-block:: json

            { "role": "user", "content": "text_content" }

        NOTE: Multimodal messages omit the ``content`` property.
        Both :obj:`image_urls` and :obj:`image` content parts are converted to:

        .. code-block:: json

            { "type": "image" }

        Their content is provided as byte arrays through the top-level property
        on the request object, that is, :obj:`RequestType.images`.
        """
        ...

    async def new_context(self, request: RequestType) -> UnboundContextType:
        """Creates a new context from a request object.

        This is sent to the worker process once and then cached locally.

        Args:
            request: Incoming request.

        Returns:
            Initialized context.
        """
        ...

    async def encode(
        self, prompt: str, add_special_tokens: bool
    ) -> TokenizerEncoded:
        """Encodes text prompts as tokens.

        Args:
            prompt: Un-encoded prompt text.
            add_special_tokens: Whether to add special tokens (for example, BOS).

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.
        """
        ...

    async def decode(self, encoded: TokenizerEncoded, **kwargs) -> str:
        """Decodes response tokens to text.

        Args:
            encoded: Encoded response tokens.
            **kwargs: Additional decoder options (for example, ``skip_special_tokens``).

        Returns:
            Un-encoded response text.
        """
        ...
