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

"""Tests for the FLUX.2 int8 W8A8 default-enable logic.

The int8 W8A8 path defaults ON only for klein (guidance-distilled,
``guidance_embeds: false``) bf16 checkpoints when every device is an Apple
M5 GPU; ``APPLE_FLUX2_INT8_W8A8`` is a two-way override (the deprecated
``FLUX2_KLEIN_INT8_W8A8`` alias still works, with a warning). FLUX.2-dev and
non-M5 devices must stay on bf16 by default, and a force-on with any non-M5
device must warn and fall back to bf16 (the int8 op is M5-only), never raise.
"""

import logging
from unittest.mock import Mock, PropertyMock

import pytest
from max.driver import CPU as HostCPU
from max.driver import Device
from max.nn.quant_config import QuantFormat
from max.pipelines.architectures.flux2 import model_config
from max.pipelines.architectures.flux2.model_config import (
    Flux2Config,
    _int8_w8a8_enabled,
    _int8_w8a8_forced,
    _is_apple_m5,
)

# Klein transformer configs ship an explicit ``"guidance_embeds": false``;
# dev configs omit the key entirely (Flux2Config defaults it to True).
KLEIN_CONFIG = {"guidance_embeds": False}
DEV_CONFIG: dict[str, object] = {}

M5 = Mock(api="metal", architecture_name="5-metal4")
M5_NO_METAL4 = Mock(api="metal", architecture_name="5")
M4 = Mock(api="metal", architecture_name="4-metal4")
CUDA = Mock(api="cuda", architecture_name="sm_100a")
CPU = Mock(api="cpu")


def test_is_apple_m5() -> None:
    assert _is_apple_m5(M5)
    assert _is_apple_m5(M5_NO_METAL4)
    assert not _is_apple_m5(M4)
    assert not _is_apple_m5(CUDA)
    assert not _is_apple_m5(CPU)
    # Metal device whose architecture lookup raises (unknown device name).
    unknown = Mock(api="metal")
    type(unknown).architecture_name = PropertyMock(
        side_effect=RuntimeError("Unknown device name")
    )
    assert not _is_apple_m5(unknown)


def _delenv_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clears both the primary env override and its deprecated alias."""
    monkeypatch.delenv("APPLE_FLUX2_INT8_W8A8", raising=False)
    monkeypatch.delenv("FLUX2_KLEIN_INT8_W8A8", raising=False)


def test_default_on_for_klein_on_m5(monkeypatch: pytest.MonkeyPatch) -> None:
    _delenv_overrides(monkeypatch)
    assert _int8_w8a8_enabled(KLEIN_CONFIG, [M5])
    assert _int8_w8a8_enabled(KLEIN_CONFIG, [M5_NO_METAL4])


def test_default_off_outside_klein_m5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _delenv_overrides(monkeypatch)
    # dev (no guidance_embeds key) stays bf16 even on M5.
    assert not _int8_w8a8_enabled(DEV_CONFIG, [M5])
    assert not _int8_w8a8_enabled({"guidance_embeds": True}, [M5])
    # klein on anything that is not exclusively M5 stays bf16.
    assert not _int8_w8a8_enabled(KLEIN_CONFIG, [M4])
    assert not _int8_w8a8_enabled(KLEIN_CONFIG, [CUDA])
    assert not _int8_w8a8_enabled(KLEIN_CONFIG, [CPU])
    assert not _int8_w8a8_enabled(KLEIN_CONFIG, [M5, CUDA])
    assert not _int8_w8a8_enabled(KLEIN_CONFIG, [])


def test_env_override_forces_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # On M5 the force-on bypasses the klein gate (dev config enables too).
    for value in ("1", "true", "yes", "TRUE"):
        monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", value)
        assert _int8_w8a8_enabled(DEV_CONFIG, [M5])
        assert _int8_w8a8_enabled(KLEIN_CONFIG, [M5])


def test_env_override_force_on_non_m5_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Force-on with a non-M5 device warns and falls back to bf16, no raise.

    The int8 W8A8 matmul op is Apple-M5-only; honoring the force-on
    elsewhere would raise at graph build.
    """
    monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", "1")
    device_cases: list[list[Device]] = [[CUDA], [M4], [CPU], [M5, CUDA], []]
    for devices in device_cases:
        with caplog.at_level(logging.WARNING, logger="max.pipelines"):
            caplog.clear()
            assert not _int8_w8a8_enabled(KLEIN_CONFIG, devices)
        assert any(
            "only implemented for Apple M5" in r.message for r in caplog.records
        )


def test_env_override_forces_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "false", "off", ""):
        monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", value)
        assert not _int8_w8a8_enabled(KLEIN_CONFIG, [M5])


def test_int8_w8a8_forced_tri_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_int8_w8a8_forced``: None unset, False off, True on-M5 only."""
    _delenv_overrides(monkeypatch)
    assert _int8_w8a8_forced([M5]) is None
    monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", "0")
    assert _int8_w8a8_forced([M5]) is False
    monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", "1")
    assert _int8_w8a8_forced([M5]) is True
    assert _int8_w8a8_forced([CUDA]) is False


def test_deprecated_env_alias(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The deprecated ``FLUX2_KLEIN_INT8_W8A8`` name still works and warns.

    The warning fires once per process, and the primary
    ``APPLE_FLUX2_INT8_W8A8`` takes precedence when both are set.
    """
    _delenv_overrides(monkeypatch)
    monkeypatch.setenv("FLUX2_KLEIN_INT8_W8A8", "1")
    monkeypatch.setattr(model_config, "_warned_deprecated_env_var", False)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        assert _int8_w8a8_forced([M5]) is True
    assert any(
        "deprecated" in r.message and "APPLE_FLUX2_INT8_W8A8" in r.message
        for r in caplog.records
    )
    # One-time: a second read honors the alias without warning again.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        assert _int8_w8a8_forced([M5]) is True
    assert not any("deprecated" in r.message for r in caplog.records)
    # The old name forces off too (full value semantics, not just on).
    monkeypatch.setenv("FLUX2_KLEIN_INT8_W8A8", "0")
    assert _int8_w8a8_forced([M5]) is False
    # The primary name wins when both are set.
    monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", "0")
    monkeypatch.setenv("FLUX2_KLEIN_INT8_W8A8", "1")
    assert _int8_w8a8_forced([M5]) is False


def test_nvfp4_encoding_int8_requant_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVFP4 encoding: unset keeps W4A16 NVFP4; force-on on M5 requants int8.

    Force-on with a non-M5 device falls back to the NVFP4 W4A16 config
    (with a warning), never the Apple-only int8 op.
    """
    # Real driver device so DeviceRef.from_device works.
    devices: list[Device] = [HostCPU()]

    _delenv_overrides(monkeypatch)
    cfg = Flux2Config.initialize_from_config({}, "float4_e2m1fnx2", devices)
    assert cfg.quant_config is not None
    assert cfg.quant_config.format == QuantFormat.NVFP4

    monkeypatch.setenv("APPLE_FLUX2_INT8_W8A8", "1")
    monkeypatch.setattr(model_config, "_is_apple_m5", lambda d: True)
    cfg = Flux2Config.initialize_from_config({}, "float4_e2m1fnx2", devices)
    assert cfg.quant_config is not None
    assert cfg.quant_config.is_int8_w8a8

    monkeypatch.setattr(model_config, "_is_apple_m5", lambda d: False)
    cfg = Flux2Config.initialize_from_config({}, "float4_e2m1fnx2", devices)
    assert cfg.quant_config is not None
    assert cfg.quant_config.format == QuantFormat.NVFP4
