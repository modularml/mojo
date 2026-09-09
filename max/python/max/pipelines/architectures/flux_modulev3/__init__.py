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
"""Standalone reference implementation of the FLUX VAE ModuleV3 pattern.

Demonstrates the *single-Module-with-two-forward-methods-and-shared-
weights* pattern that
:doc:`/max/docs/internal/ExperimentalModuleBestPractices` calls out
(see MXF-460).  The :class:`Vae` Module owns both encoder and decoder
plus the shared per-channel batch-norm statistics, and exposes the
two halves as :meth:`Vae.encode` / :meth:`Vae.decode` rather than a
single ``forward``.

The whole package is self-contained -- the supporting building blocks
(``Encoder``, ``Decoder``, ``DownEncoderBlock2D``, etc.), the per-
layer modules (``Downsample2D``, ``ResnetBlock2D``, ...), and the
config class live alongside ``Vae`` so the demonstration does not
depend on the broader ``autoencoders_modulev3`` package.
"""

from .vae import Vae

__all__ = ["Vae"]
