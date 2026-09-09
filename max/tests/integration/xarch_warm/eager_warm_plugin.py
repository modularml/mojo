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

"""Pytest plugin that points the eager warm cache at the build-warmed dir."""

from __future__ import annotations

import os
import sys

from python.runfiles import runfiles

_warm_rloc = os.environ.get("XARCH_WARM_RLOCATION")
if _warm_rloc:
    _runfiles = runfiles.Create()
    _resolved = _runfiles.Rlocation(_warm_rloc) if _runfiles else None
    if _resolved:
        os.environ["MODULAR_DERIVED_PATH"] = _resolved
    else:
        # Surface a miss: otherwise the warm silently won't adopt and the
        # cold-compile just reads as a timeout.
        print(
            f"[eager-warm] XARCH_WARM_RLOCATION={_warm_rloc!r} did not resolve; "
            "warm cache not adopted -- GC sweep will cold-compile.",
            file=sys.stderr,
            flush=True,
        )
