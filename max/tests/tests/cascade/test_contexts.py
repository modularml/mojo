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
"""Smoke tests for the in-process LocalRuntime."""

from __future__ import annotations

import pytest
from max.experimental.cascade import LocalRuntime, Worker, worker_method
from max.experimental.cascade.core.pipeline_method import (
    _pipeline_method_scope,
)


class _CPUEchoWorker(Worker):
    def __init__(self, prefix: str) -> None:
        super().__init__(deploy_hints=["cpu"])
        self.prefix = prefix

    @worker_method()
    async def echo(self, value: str) -> str:
        return f"{self.prefix}:{value}"


@pytest.mark.asyncio
async def test_local_runtime_deploys_worker() -> None:
    async with LocalRuntime() as runtime, _pipeline_method_scope():
        proxy = await runtime.deploy(_CPUEchoWorker("local"))
        # Two awaits: the first gets the ``Result`` handle, the second
        # resolves it to the value. Pipelines decorated with
        # ``@pipeline_method`` do the second await automatically when
        # passing a ``Result`` as an argument to another worker method.
        echo_handle = await proxy.echo("ok")
        assert await echo_handle == "local:ok"
