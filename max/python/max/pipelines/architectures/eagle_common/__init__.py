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
"""Shared Eagle3 draft components reused across target architectures."""

from .eagle_mha_draft import Eagle3MHADraft, Eagle3MHADraftConfig
from .eagle_mla_draft import Eagle3MLADraft

__all__ = ["Eagle3MHADraft", "Eagle3MHADraftConfig", "Eagle3MLADraft"]
