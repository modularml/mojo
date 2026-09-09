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

import os
from collections.abc import Generator
from pathlib import Path

import max.driver as md
import pytest
from hypothesis import settings
from max.engine import InferenceSession

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multiple times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")


@pytest.fixture(autouse=True)
def clean_up_gpus() -> Generator[None, None, None]:
    """Call synchronize after each test on all accelerators.

    GPU failures for a particular device can spill over to later tests,
    incorrectly reporting the source of the error. This fixture synchronizes
    all accelerators after each test, which will propagate any pending errors
    up to the Python level.
    """

    yield

    for i in range(md.accelerator_count()):
        accelerator = md.Accelerator(i)
        accelerator.synchronize()


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    devices: list[md.Device] = []
    for i in range(md.accelerator_count()):
        devices.append(md.Accelerator(i))

    devices.append(md.CPU())

    return InferenceSession(devices=devices)


@pytest.fixture
def adv_fusion_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enables the new MAP-dialect fusion system for the graph this test compiles.

    ``MAX_GC_USE_ADV_FUSION`` is presence-only (the value is ignored); "1"
    matches the convention used by every RUN line in the MLIR test suite.
    """
    monkeypatch.setenv("MAX_GC_USE_ADV_FUSION", "1")


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture
def counter_mojoc() -> Path:
    path = os.getenv("MODULAR_COUNTER_OPS_PATH")
    assert path is not None, "Test couldn't find `MODULAR_COUNTER_OPS_PATH` env"
    return Path(path)


@pytest.fixture
def custom_ops_mojoc() -> Path:
    path = os.getenv("CUSTOM_OPS_PATH")
    assert path is not None, "Test couldn't find `CUSTOM_OPS_PATH` env"
    return Path(path)


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None
    return Path(modular_path)


@pytest.fixture
def mo_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return modular_path / "max/tests/integration/Inputs/mo-model.mlir"


@pytest.fixture
def no_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec without inputs."""
    return modular_path / "max/tests/integration/Inputs/no-inputs.mlir"


@pytest.fixture
def scalar_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with scalar inputs."""
    return modular_path / "max/tests/integration/Inputs/scalar-input.mlir"


@pytest.fixture
def aliasing_outputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with outputs that alias each other."""
    return modular_path / "max/tests/integration/Inputs/aliasing-outputs.mlir"


@pytest.fixture
def named_inputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec that adds a series of named tensors."""
    return modular_path / "max/tests/integration/Inputs/named-inputs.mlir"
