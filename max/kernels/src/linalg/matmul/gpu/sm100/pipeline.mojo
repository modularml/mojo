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
"""SM100 pipeline utilities - re-exports from sm100_structured.

This module re-exports pipeline types from sm100_structured for backward
compatibility. New code should import directly from sm100_structured.pipeline.
"""

from structured_kernels.pipeline import ProducerConsumerPipeline
from structured_kernels.pipeline_backend import MbarPtr
