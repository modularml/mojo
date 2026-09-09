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

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import pytest
from max.pipelines.architectures.inkling.model_config import (
    parse_inkling_mtp_config,
)
from max.pipelines.architectures.unified_mtp_inkling.weight_adapters import (
    convert_with_mtp_state_dict,
)


def test_parse_inkling_mtp_config_reads_namespace_and_caps_depths() -> None:
    hf = SimpleNamespace(
        mtp_config=SimpleNamespace(
            num_nextn_predict_layers=8,
            chain_hidden_post_norm=True,
            local_layer_ids=[0, 2, 4],
            mtp_hidden_states_first=False,
        )
    )
    mtp = parse_inkling_mtp_config(hf)
    assert mtp is not None
    assert mtp.num_nextn_predict_layers == 8
    assert mtp.chain_hidden_post_norm is True
    assert mtp.local_layer_ids == (0, 2, 4)
    assert mtp.hidden_states_first is False
    assert mtp.num_depths(2) == 2
    assert mtp.num_depths(16) == 8


def test_parse_inkling_mtp_config_reads_dict_and_defaults() -> None:
    hf = SimpleNamespace(
        mtp_config={
            "num_nextn_predict_layers": 4,
            "local_layer_ids": [1],
        }
    )
    mtp = parse_inkling_mtp_config(hf)
    assert mtp is not None
    assert mtp.num_depths(3) == 3
    assert mtp.chain_hidden_post_norm is False
    assert mtp.local_layer_ids == (1,)
    assert mtp.hidden_states_first is True


def test_parse_inkling_mtp_config_missing_or_empty_is_none() -> None:
    assert parse_inkling_mtp_config(SimpleNamespace()) is None
    assert (
        parse_inkling_mtp_config(
            SimpleNamespace(mtp_config={"num_nextn_predict_layers": 0})
        )
        is None
    )


def test_inkling_mtp_config_num_depths_rejects_empty_spec() -> None:
    hf = SimpleNamespace(mtp_config={"num_nextn_predict_layers": 2})
    mtp = parse_inkling_mtp_config(hf)
    assert mtp is not None
    with pytest.raises(ValueError, match="at least one speculative token"):
        mtp.num_depths(0)


def test_convert_with_mtp_state_dict_routes_target_and_draft() -> None:
    weight = NonCallableMock()
    state_dict = {
        "model.llm.embed.weight": weight,
        "model.llm.layers.0.attn.wq_du.weight": weight,
        "model.mtp.chain_norm.weight": weight,
        "model.mtp.layers.0.hidden_norm.weight": weight,
        "model.mtp.layers.0.embed_norm.weight": weight,
        "model.mtp.layers.0.input_proj.weight": weight,
        "model.mtp.layers.0.transformer_block.attn.wq_du.weight": weight,
        "model.mtp.layers.0.transformer_block.attn_norm.weight": weight,
        "model.audio.unused.weight": weight,
        "model.visual.unused.weight": weight,
    }

    new_state_dict = convert_with_mtp_state_dict(
        state_dict,
        None,
    )

    assert "target.embed.weight" in new_state_dict
    assert "target.layers.0.attn.wq_du.weight" in new_state_dict
    assert "draft.chain_norm.weight" in new_state_dict
    assert "draft.layers.0.hidden_norm.weight" in new_state_dict
    assert "draft.layers.0.embed_norm.weight" in new_state_dict
    assert "draft.layers.0.input_proj.weight" in new_state_dict
    assert "draft.layers.0.decoder_layer.attn.wq_du.weight" in new_state_dict
    assert "draft.layers.0.decoder_layer.attn_norm.weight" in new_state_dict
    assert "vision.unused.weight" in new_state_dict
    assert not any(key.startswith("model.audio") for key in new_state_dict)
    assert not any(key.startswith("model.visual") for key in new_state_dict)
    assert not any(key.startswith("model.mtp") for key in new_state_dict)


def _depth_weight_names(depth: int) -> list[str]:
    return [
        f"model.mtp.layers.{depth}.hidden_norm.weight",
        f"model.mtp.layers.{depth}.input_proj.weight",
        f"model.mtp.layers.{depth}.transformer_block.attn.wq_du.weight",
    ]


def test_convert_with_mtp_state_dict_drops_depths_past_the_cap() -> None:
    """A K=2 draft leaves the checkpoint's remaining depths unread."""
    hf = SimpleNamespace(mtp_config={"num_nextn_predict_layers": 8})
    pipeline_config = NonCallableMock(
        speculative=SimpleNamespace(num_speculative_tokens=2)
    )
    state_dict = {
        name: NonCallableMock()
        for depth in range(8)
        for name in _depth_weight_names(depth)
    }
    state_dict["model.mtp.chain_norm.weight"] = NonCallableMock()

    new_state_dict = convert_with_mtp_state_dict(
        state_dict, hf, pipeline_config
    )

    assert "draft.chain_norm.weight" in new_state_dict
    for depth in (0, 1):
        assert f"draft.layers.{depth}.hidden_norm.weight" in new_state_dict
    for depth in range(2, 8):
        assert not any(
            key.startswith(f"draft.layers.{depth}.") for key in new_state_dict
        )
        # Dropped depths must not be read off disk, not merely discarded.
        for name in _depth_weight_names(depth):
            state_dict[name].data.assert_not_called()


def test_convert_with_mtp_state_dict_keeps_all_depths_without_a_width() -> None:
    """With no pipeline config to resolve the width, nothing is dropped."""
    hf = SimpleNamespace(mtp_config={"num_nextn_predict_layers": 8})
    state_dict = {name: NonCallableMock() for name in _depth_weight_names(7)}

    new_state_dict = convert_with_mtp_state_dict(state_dict, hf, None)

    assert "draft.layers.7.hidden_norm.weight" in new_state_dict


def test_unified_mtp_inkling_inputs_dataclass_imports() -> None:
    from max.pipelines.architectures.unified_mtp_inkling.model import (
        UnifiedMTPInklingInputs,
    )

    assert (
        "host_input_row_offsets" in UnifiedMTPInklingInputs.__dataclass_fields__
    )
    assert "draft_conv_pools" in UnifiedMTPInklingInputs.__dataclass_fields__
    assert "draft_tokens" in UnifiedMTPInklingInputs.__dataclass_fields__
    assert "image_embeddings" in UnifiedMTPInklingInputs.__dataclass_fields__
    assert "image_indices" in UnifiedMTPInklingInputs.__dataclass_fields__
