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
"""Tests for the eager cold-start policy."""

import json
from collections.abc import Callable, Generator, Mapping
from pathlib import Path

import pytest
from max import _core, _eager_policy, _interpreter_ops
from max._interpreter_ops import gc_compile
from max.driver import CPU, Device
from max.graph import Module

_ALLOW = _eager_policy.ALLOW_LAZY_COMPILE_ENV_VAR


@pytest.fixture(autouse=True)
def _isolated_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[None]:
    """Resets warned-once state and points the warm cache at an empty dir."""
    _eager_policy._reset_for_test()
    monkeypatch.delenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", raising=False)
    monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
    yield
    _eager_policy._reset_for_test()


class _StubSpec:
    """A family whose sweep must never really compile."""

    name = "stub"

    def build_module(self) -> Module:
        raise AssertionError("policy tests must not reach a real compile")

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        raise AssertionError("policy tests must not reach a real compile")

    def sweep_devices(self) -> list[Device]:
        return []


def _unreachable_build(module: Module) -> None:
    raise AssertionError("policy tests must not reach a real compile")


@pytest.fixture
def stubbed_family(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., gc_compile.GCOpFamily]:
    """A ``GCOpFamily`` factory whose sweep adopts *adopts* iff ``warmed``."""

    def make(
        *, warmed: bool, adopts: Mapping[str, object] | None = None
    ) -> gc_compile.GCOpFamily:
        family = gc_compile.GCOpFamily(spec=_StubSpec())
        monkeypatch.setattr(gc_compile, "read_manifest", lambda: None)
        monkeypatch.setattr(gc_compile, "warm_stamp_matches", lambda: warmed)

        def sweep() -> bool:
            family.cache.update(adopts or {})  # type: ignore[arg-type]
            return True

        monkeypatch.setattr(family, "compile_sweep", sweep)
        return family

    return make


class _SweepableSpec:
    """A family spec whose ``build_module`` either succeeds trivially or
    raises, so ``GCOpFamily.compile_sweep`` runs for real without compiling.
    """

    def __init__(self, name: str, *, explode: bool = False) -> None:
        self.name = name
        self.explode = explode

    def build_module(self) -> Module:
        if self.explode:
            raise RuntimeError(f"{self.name} sweep failed")
        return Module()

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        raise AssertionError("not used by compile_sweep")

    def sweep_devices(self) -> list[Device]:
        # Non-empty, or compile_sweep no-ops before reaching build_module.
        return [CPU()]


class _FakeSession:
    """Stands in for a real ``InferenceSession`` so a sweep never compiles."""

    def load_all(
        self, module: object, *, weights_registry: Mapping[str, object]
    ) -> dict[str, object]:
        return {}


def test_warmed_machine_serves_under_refusal(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_family: Callable[..., gc_compile.GCOpFamily],
) -> None:
    """Serves an eager op from the adopted warm under a refusing policy."""
    # A policy check ahead of ensure_swept() refused eager ops on
    # provisioned machines.
    monkeypatch.setenv(_ALLOW, "0")
    key = "stub_cpu_0_float32"
    sentinel = object()
    family = stubbed_family(warmed=True, adopts={key: sentinel})

    assert family.model_for(key, CPU(), _unreachable_build) is sentinel


def test_no_stamp_when_a_later_family_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep where one family raises leaves no warm stamp behind."""
    # Stamping per family let the first family's success claim the
    # whole machine.
    monkeypatch.setattr(gc_compile, "read_manifest", lambda: None)
    monkeypatch.setattr(
        gc_compile.engine, "InferenceSession", lambda **_: _FakeSession()
    )
    first = gc_compile.GCOpFamily(spec=_SweepableSpec("first"))
    second = gc_compile.GCOpFamily(spec=_SweepableSpec("second", explode=True))
    monkeypatch.setattr(_interpreter_ops, "GC_FAMILIES", (first, second))

    with pytest.raises(RuntimeError, match="second sweep failed"):
        _interpreter_ops.compile_all_families()

    assert first.swept, "the first family really did sweep"
    assert not gc_compile.provisioned(), (
        "a partial sweep must not claim the machine is warm"
    )


def test_other_version_stamp_does_not_match() -> None:
    """A stamp from another MAX version does not satisfy ``provisioned``."""
    # The stamp sits above the versioned MEF directory the engine prunes.
    cache_dir = gc_compile._cache_dir()
    assert cache_dir is not None
    stamp = cache_dir / gc_compile._WARM_STAMP_NAME
    assert gc_compile.write_warm_stamp()
    assert gc_compile.provisioned()

    payload = json.loads(stamp.read_text())
    current = f";version={_core.__version__}"
    assert current in payload["context"]
    payload["context"] = payload["context"].replace(
        current, ";version=0.0.0.dev0"
    )
    stamp.write_text(json.dumps(payload))

    assert not gc_compile.provisioned()


@pytest.mark.parametrize(
    ("precompile", "lazy", "expect_raises"),
    [
        (True, True, False),
        (True, False, True),
        (False, True, False),
        (False, False, True),
    ],
    ids=[
        "import-sweep,lazy=1",
        "import-sweep,lazy=0",
        "lazy-dispatch,lazy=1",
        "lazy-dispatch,lazy=0",
    ],
)
def test_policy_matrix_on_an_unwarmed_machine(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_family: Callable[..., gc_compile.GCOpFamily],
    precompile: bool,
    lazy: bool,
    expect_raises: bool,
) -> None:
    """Checks the documented outcome of each precompile x lazy-compile cell."""
    monkeypatch.setattr(gc_compile, "provisioned", lambda: False)
    monkeypatch.setattr(_eager_policy, "allow_lazy_compile", lambda: lazy)

    if precompile:
        monkeypatch.setattr(gc_compile, "should_precompile", lambda: True)
        calls: list[None] = []
        monkeypatch.setattr(
            _interpreter_ops,
            "compile_all_families",
            lambda: calls.append(None),
        )
        if expect_raises:
            with pytest.raises(_eager_policy.EagerLazyCompileDisallowed):
                _interpreter_ops._precompile_gc_models()
            assert calls == []
        else:
            _interpreter_ops._precompile_gc_models()
            assert calls == [None]
        return

    family = stubbed_family(warmed=False)
    sentinel = object()
    monkeypatch.setattr(
        gc_compile, "compile_single_target", lambda *args: sentinel
    )
    key = "stub_cpu_0_float32"
    if expect_raises:
        with pytest.raises(_eager_policy.EagerLazyCompileDisallowed):
            family.model_for(key, CPU(), _unreachable_build)
    else:
        assert family.model_for(key, CPU(), _unreachable_build) is sentinel
