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
"""Shared device-count config for the eager-GC-sweep warm producer and its
consumer test.

The host-CPU target is no longer pinned here: the producer sources it from a
per-build-platform ``select()`` (see ``BUILD.bazel``, mirroring the mojo
toolchain) and records it in the warm manifest, and the consumer reads it back
from that manifest, so the two ends agree without a shared constant. This file
now only carries the device count, which both sides still pin identically.
"""

# The MAX accelerator count warmed, as one MEF per slot. Per-slot MEFs are
# device-count-independent, so this is a ceiling, not an exact match: a box with
# <= this many GPUs adopts by force-loading slots 0..k-1. Set to the largest GPU
# count CI serves (the has_4_gpus tier, there is no has_8_gpus), so any
# smaller box is covered by one warm; a larger box falls back to lazy compile.
WARM_DEVICE_COUNT = 4
