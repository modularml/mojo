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
"""The checkpoint's five components, as a manifest presents them.

Everything below the executor takes configs and opened weights, not paths, so
that a local checkout and a Hub repo are the same thing to it. This module is
where that translation happens, and it is the only place in the architecture
that knows what a :class:`ModelManifest` is.
"""

from __future__ import annotations

from pathlib import Path

from max.graph.weights import Weights, load_weights
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest

from .model_config import MiniMaxMusic3Config

# The roles this port executes, named as the checkpoint's component index names
# them. Its `scheduler` and `tokenizer` entries are not among them: the flow
# schedule is arithmetic (see `denoise.euler_schedule`) and the tokenizer is the
# pipeline's rather than the executor's.
ROLES = (
    "language_model",
    "rvq_depth_decoder",
    "condition_encoder",
    "transformer",
    "vocoder",
)


class Checkpoint:
    """One checkpoint's configs, and its weights on demand.

    Weights are opened per call rather than cached, because a stage's weights
    should become collectable the moment the stage is dropped -- which is what
    makes staged residency work.
    """

    def __init__(self, manifest: ModelManifest) -> None:
        """Reads every component's config.

        Args:
            manifest: The manifest the pipeline resolved, whose keys are the
                checkpoint's component roles.

        Raises:
            ValueError: If a component this port needs is absent, which means
                the repo is not this pipeline however it is named.
        """
        missing = [role for role in ROLES if role not in manifest]
        if missing:
            raise ValueError(
                "not a MiniMax Music 3 checkpoint: its component index is "
                f"missing {', '.join(missing)}"
            )
        self._manifest = manifest
        self.config = MiniMaxMusic3Config.from_dicts(
            {
                role: manifest[role].huggingface_config.to_dict()
                for role in ROLES
            }
        )

    def weights(self, role: str) -> Weights:
        """Opens one component's checkpoint, downloading it if it is remote."""
        return load_weights(self.weight_paths(role))

    def weight_paths(self, role: str) -> list[Path]:
        """Resolves one component's weight files to local paths.

        Resolved on demand rather than read from the config's ``weight_path``,
        which a manifest's components carry only once something has asked for
        them.
        """
        config = self._manifest[role]
        _, weight_path = _resolve_component_encoding_and_weights(config)
        return config.resolved_weight_paths(weight_path)

    def weight_bytes(self) -> int:
        """Total on-disk size of every component this port executes.

        The checkpoint is bfloat16 throughout, so this is also what the weights
        cost on the device -- which is what decides whether they can all be
        there at once.
        """
        return sum(
            path.stat().st_size
            for role in ROLES
            for path in self.weight_paths(role)
        )
