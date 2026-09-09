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

"""Defines the :class:`LogProbabilities` data structure for token log probability output."""

import msgspec


class LogProbabilities(msgspec.Struct, tag=True, omit_defaults=True):
    """Log probabilities for an individual output token.

    This is a data-only class that serves as a serializable data structure for
    transferring log probability information. It does not provide any functionality
    for calculating or manipulating log probabilities - it is purely for data storage
    and serialization purposes.
    """

    token_log_probabilities: list[float]
    """Probabilities of each token."""
    top_log_probabilities: list[dict[int, float]]
    """Top tokens and their corresponding probabilities."""
