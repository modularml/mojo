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
"""Vendor CCL bindings for multi-GPU collective operations.

Provides Mojo wrappers around NCCL (NVIDIA) and RCCL (AMD) collective
communication libraries. The library is loaded at runtime from standard system
paths; the `is_allreduce_available()`, `is_allgather_available()`, and
`is_broadcast_available()` probes can be used to check availability before
calling into a collective.
"""
