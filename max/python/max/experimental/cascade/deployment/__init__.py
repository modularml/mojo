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
"""Cascade deployment composition: routing and config-driven runtime assembly.

Composes the transport runtimes (:mod:`~max.experimental.cascade.core`,
:mod:`~max.experimental.cascade.grpc_runtime`,
:mod:`~max.experimental.cascade.http_runtime`) into a device-routed, pooled
deployment. Import from the submodules directly (``deployment.routing``,
``deployment.context_config``) to keep import costs low.
"""
