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
"""Contains implementations for several types of key-value caches.

KV caches are used in transformer models to store
key-value tensors output from self-attention layers, allowing previously
computed results to be reused across decoding steps.

This package provides the low-level cache types and index-remapping kernels
used during paged and sparse attention. These APIs are consumed by the
higher-level functions in the [`nn`](../nn/) package and are not typically
called directly by model authors.
"""
