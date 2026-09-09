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
"""State space model kernels for Mamba and related architectures.

This package provides GPU and CPU kernel implementations for the selective
scan, causal conv1d, and SSD (state-space duality) operations used by Mamba,
Mamba-2, and Gated DeltaNet models, including variable-length sequence variants
for continuous-batching inference.

Both forward and update (decode-step) paths are included, and operation
registrations in the `*_ops` modules wire these kernels into the MAX graph
compiler.
"""
