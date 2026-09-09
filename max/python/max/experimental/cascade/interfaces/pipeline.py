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
"""Base classes shared by all cascade-served pipelines."""

import asyncio

from max.experimental.cascade.core import Runtime, Worker
from max.experimental.cascade.interfaces.gen_ai import GenAIInterface


class CascadePipeline:
    """Abstract base class for cascade-served pipelines."""

    def _get_workers(self) -> dict[str, Worker]:
        """Return worker-valued pipeline attributes keyed by attribute name."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if isinstance(value, Worker)
        }

    async def deploy(self, runtime: Runtime) -> None:
        """Deploy worker attributes in place and replace them with proxies."""
        workers = self._get_workers()
        deployed = await asyncio.gather(
            *(runtime.deploy(worker) for worker in workers.values())
        )
        for name, proxy in zip(workers, deployed, strict=True):
            setattr(self, name, proxy)


class GenAIPipeline(CascadePipeline, GenAIInterface):
    """A deployable cascade pipeline that answers generative-AI requests.

    What ``build_pipeline`` returns and what the serving layer routes on. Kept
    distinct from :class:`CascadePipeline`, which is only the worker-deployment
    machinery -- serving-layer wrappers reuse that without generating anything
    themselves.
    """
