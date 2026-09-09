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

from pathlib import Path

import pytest
from tuning_codegen import (
    generate_target,
    load_manifest,
    render_region,
    render_snippet,
    replace_generated_region,
)


def test_render_snippet_formats_mojo_values() -> None:
    assert (
        render_snippet(
            "Config(M=[@M], enabled=[@ENABLED])",
            {"M": 64, "ENABLED": True},
            indent="  ",
        )
        == "  Config(M=64, enabled=True)"
    )


def test_render_snippet_rejects_missing_parameters() -> None:
    with pytest.raises(ValueError, match="Missing snippet parameters: N"):
        render_snippet("Config(M=[@M], N=[@N])", {"M": 64})


def test_load_manifest_preserves_order_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "tuning.yaml"
    manifest.write_text(
        """
params:
- M: 64
  N: 128
- M: 32
  N: 256
"""
    )
    assert load_manifest(manifest) == [
        {"M": 64, "N": 128},
        {"M": 32, "N": 256},
    ]

    manifest.write_text(
        """
params:
- M: 64
- M: 64
"""
    )
    with pytest.raises(ValueError, match="duplicate params entry at index 1"):
        load_manifest(manifest)


def test_replace_generated_region_requires_unique_markers() -> None:
    source = "before\n# BEGIN\nstale\n# END\nafter\n"
    assert (
        replace_generated_region(
            source,
            begin_marker="# BEGIN",
            end_marker="# END",
            generated="fresh",
        )
        == "before\n# BEGIN\nfresh\n# END\nafter\n"
    )

    with pytest.raises(ValueError, match="exactly one begin marker"):
        replace_generated_region(
            source + "# BEGIN\n",
            begin_marker="# BEGIN",
            end_marker="# END",
            generated="fresh",
        )


def test_render_region_selects_entry_template() -> None:
    assert render_region(
        [
            {"M": 64},
            {"M": 128, "M_END": 160, "_template": "ranged"},
        ],
        "Config(M=[@M])",
        source_label="tuning.yaml",
        snippet_variants={"ranged": "Config(M=[@M], M_end=[@M_END])"},
        indent="",
    ) == (
        "# Automatically generated from [tuning.yaml]\n"
        "# index: [0]\n"
        "Config(M=64),\n"
        "# Automatically generated from [tuning.yaml]\n"
        "# index: [1]\n"
        "Config(M=128, M_end=160),"
    )

    with pytest.raises(ValueError, match="unknown template 'missing'"):
        render_region(
            [{"M": 64, "_template": "missing"}],
            "Config(M=[@M])",
            source_label="tuning.yaml",
        )


def test_generate_target_renders_manifest_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "tuning.yaml"
    snippet = tmp_path / "tuning.mojo.snippet"
    target = tmp_path / "tuning_configs.mojo"
    manifest.write_text("params:\n- M: 64\n- M: 128\n")
    snippet.write_text("Config(M=[@M])\n")
    target.write_text("# BEGIN\nstale\n# END\n")

    assert generate_target(
        manifest_path=manifest,
        snippet_path=snippet,
        target_path=target,
        begin_marker="# BEGIN",
        end_marker="# END",
        source_label="tuning.yaml",
    ) == (
        "# BEGIN\n"
        "        # Automatically generated from [tuning.yaml]\n"
        "        # index: [0]\n"
        "        Config(M=64),\n"
        "        # Automatically generated from [tuning.yaml]\n"
        "        # index: [1]\n"
        "        Config(M=128),\n"
        "# END\n"
    )
