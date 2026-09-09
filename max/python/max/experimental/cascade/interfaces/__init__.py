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
"""Cascade pipeline interfaces and shared request types.

Holds the abstractions (the ``CascadePipeline`` / ``GenAIPipeline`` bases, the
modality-agnostic ``GenAIInterface``, and its request and chunk types) that both
``workers`` and ``serve`` depend on, keeping them decoupled from concrete
pipeline implementations.
"""
