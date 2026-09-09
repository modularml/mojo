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
"""CLI: compose N per-family warm fragments into one consumer cache dir.

Thin wrapper over ``compose_lib.merge`` (the merge logic lives there so the
unit test can import it without also owning this file's srcs).
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from max.tests.integration.xarch_warm import compose_lib


@click.command()
@click.option(
    "--warm",
    "warm_dirs",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path),
    help="A warm fragment dir (per-family CPU or accelerator). Repeatable.",
)
def main(warm_dirs: tuple[Path, ...]) -> None:
    out = Path(os.environ["MODULAR_DERIVED_PATH"])
    envelope, entries = compose_lib.merge(list(warm_dirs), out)
    print(
        f"XARCH_COMPOSE: merged {len(warm_dirs)} fragments -> {len(entries)} "
        f"entries ({sorted(e['device_class'] for e in entries)}); "
        f"envelope keys={sorted(envelope)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
