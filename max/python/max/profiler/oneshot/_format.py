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

"""Shared formatting helpers for the one-shot profiler renderers."""

from __future__ import annotations


def _format_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns / 1e9:>7.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:>7.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:>7.2f} us"
    return f"{ns:>7.0f} ns"
