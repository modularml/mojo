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
"""Normalizing a model's config twice must return the same values.

Some architectures rewrite the HuggingFace config in place (Step-3.5 collapses
its per-layer rope_theta list down to a scalar), and the config gets normalized
more than once per build. If that rewrite isn't idempotent, the second pass
sees the already-rewritten config and returns different values, so the model
compiles with the wrong config. Architectures opt in below by listing their
normalization function and a factory for a representative config.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from max.pipelines.architectures.step3p5.model_config import Step3p5Config


def _make_step3p5_config() -> SimpleNamespace:
    # A trust_remote_code config, where rope_theta is still the raw per-layer
    # list: 5000000.0 on full-attention layers, 10000.0 on sliding-window ones.
    return SimpleNamespace(
        rope_theta=[5000000.0, 10000.0, 10000.0, 10000.0],
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        rope_scaling=None,
        hidden_act="silu",
    )


# Opt-in list: (id, config-normalization function, config factory).
_OPTED_IN_ARCHITECTURES = [
    ("step3p5", Step3p5Config._ensure_hf_config_aliases, _make_step3p5_config),
]


@pytest.mark.parametrize(
    "normalize, make_config",
    [(fn, make) for _, fn, make in _OPTED_IN_ARCHITECTURES],
    ids=[arch_id for arch_id, _, _ in _OPTED_IN_ARCHITECTURES],
)
def test_config_normalization_is_idempotent(
    normalize: Callable[[object], object],
    make_config: Callable[[], object],
) -> None:
    config = make_config()

    first_pass = normalize(config)
    second_pass = normalize(config)

    assert first_pass == second_pass, (
        "normalizing the config twice returned different values, so the "
        "rewrite is not idempotent and the model would build with the wrong "
        "config"
    )
