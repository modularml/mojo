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
"""Tests for structured-output sampling across grammar backends.

Parametrized over the pluggable grammar backends (``llguidance`` and
``xgrammar``): each compiles a JSON schema, fills a packed int32 bitmask, and
feeds it through the GPU ``token_sampler``, then asserts the sampled tokens are
grammar-legal. This exercises the ``GrammarBackend`` abstraction symmetrically
and confirms each backend's bitmask layout is compatible with the GPU
``apply_packed_bitmask`` path.
"""

import json

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.context import SamplingParams
from max.pipelines.lib import SamplingConfig, token_sampler
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    make_grammar_backend,
)
from transformers import AutoConfig, AutoTokenizer

_PERSON_SCHEMA = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


@pytest.fixture(scope="module")
def structured_output_sampler(session: InferenceSession) -> Model:
    """Compile token_sampler with structured output enabled."""
    sampling_config = SamplingConfig(
        enable_structured_output=True,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph = token_sampler(sampling_config, device=DeviceRef.GPU())
    return session.load(graph)


@pytest.mark.parametrize("backend_name", ["llguidance", "xgrammar"])
def test_structured_output_sampling(
    session: InferenceSession,
    structured_output_sampler: Model,
    modular_ai_llama_3_1_local_path: str,
    backend_name: str,
) -> None:
    config = AutoConfig.from_pretrained(modular_ai_llama_3_1_local_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        modular_ai_llama_3_1_local_path
    )
    vocab_size = config.vocab_size

    backend = make_grammar_backend(
        backend_name,
        hf_tokenizer,
        vocab_size,
    )
    compiled = backend.compile_json_schema(json.dumps(_PERSON_SCHEMA))
    matcher = backend.create_matcher(compiled)

    device = session.devices[0]
    batch_size = 1
    n_trials = 1

    sampling_params = SamplingParams(top_k=5)

    generated_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    temperature = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)
    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p = Buffer.from_numpy(
        np.array([sampling_params.top_p] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(
        np.array(sampling_params.top_p, dtype=np.float32)
    )
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)
    seed = Buffer.from_numpy(
        np.array([sampling_params.seed] * batch_size, dtype=np.uint64)
    ).to(device)
    for _ in range(n_trials):
        token_bitmask = backend.allocate_token_bitmask(batch_size, vocab_size)
        backend.fill_next_token_bitmask(matcher, token_bitmask, 0)

        logits = np.random.default_rng().random(
            size=(batch_size, vocab_size), dtype=np.float32
        )

        bitmask = np.ascontiguousarray(token_bitmask)

        _, new_tokens = structured_output_sampler(
            Buffer.from_dlpack(logits).to(device),
            generated_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p,
            seed,
            Buffer.from_dlpack(bitmask).to(device),
        )[:2]
        assert isinstance(new_tokens, Buffer)
        for token in new_tokens.to_numpy():
            token_ids = [int(t) for t in token]
            assert matcher.try_consume_tokens(token_ids) == len(token_ids)
