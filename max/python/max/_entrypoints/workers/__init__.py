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
import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack
from types import FrameType
from typing import Optional

import uvloop
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.modeling.types import PipelineTask
from max.serve.config import Settings
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.telemetry.metrics import METRICS
from max.serve.worker_interface._zmq_queue import generate_zmq_ipc_path
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface

logger = logging.getLogger("max._entrypoints")

# Global shutdown event for coordinating graceful shutdown
_shutdown_event: asyncio.Event | None = None


def sigterm_handler(sig: int, frame: FrameType | None) -> None:
    """Handle SIGTERM by setting the shutdown event."""
    logger.info("SIGTERM received, initiating graceful shutdown")
    if _shutdown_event is not None and not _shutdown_event.is_set():
        # Schedule the shutdown event to be set in the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_shutdown_event.set)
        except RuntimeError:
            # No running event loop, exit immediately
            logger.warning(
                "No running event loop found during SIGTERM, exiting immediately"
            )
            sys.exit(0)
    else:
        # Fallback: exit immediately if no shutdown event is available
        logger.info("Graceful shutdown complete, exiting with success code")
        sys.exit(0)


def sigint_handler(sig: int, frame: FrameType | None) -> None:
    """Handle SIGINT by setting the shutdown event."""
    logger.info("SIGINT received, initiating graceful shutdown")
    if _shutdown_event is not None and not _shutdown_event.is_set():
        # Schedule the shutdown event to be set in the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_shutdown_event.set)
        except RuntimeError:
            # No running event loop, raise KeyboardInterrupt as fallback
            raise KeyboardInterrupt("SIGINT received") from None
    else:
        # Fallback: raise KeyboardInterrupt if no shutdown event is available
        raise KeyboardInterrupt("SIGINT received")


def start_workers(
    settings: Settings,
    pipeline_args: PipelineArgs,
) -> None:
    global _shutdown_event  # noqa: PLW0602 (FIXME)

    async def run_workers() -> None:
        global _shutdown_event

        # Create shutdown event for coordinating graceful shutdown
        _shutdown_event = asyncio.Event()

        logger.info("Starting MAX Workers...")

        # Load the Tokenizer and Pipeline Factory
        pipeline_config = PipelineConfig.from_args(pipeline_args)
        retrieved = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=pipeline_args.task,
        )
        pipeline_factory = retrieved.factory

        try:
            async with AsyncExitStack() as exit_stack:
                # Start telemetry worker and Configure Metrics to use it
                metric_client = await exit_stack.enter_async_context(
                    start_telemetry_consumer(settings)
                )

                METRICS.configure(client=metric_client)

                # Start Model Worker
                zmq_endpoint_base = generate_zmq_ipc_path()
                _ = await exit_stack.enter_async_context(
                    start_model_worker(
                        pipeline_factory,
                        pipeline_config,
                        settings,
                        metric_client,
                        model_worker_interface=ZmqModelWorkerInterface(
                            pipeline_config.task,
                            context_type=PIPELINE_REGISTRY.retrieve_context_type(
                                pipeline_config
                            ),
                        ),
                        zmq_endpoint_base=zmq_endpoint_base,
                        memory_plan=retrieved.memory_plan,
                    )
                )

                METRICS.pipeline_load(pipeline_config.model.model_path)

                logger.info(
                    f"\n\n{'*' * 80}\n\n"
                    f"{'🚀 Headless server ready (Press CTRL+C to quit)'.center(80)}\n\n"
                    f"{'*' * 80}\n"
                )

                # Wait for shutdown signal instead of infinite loop
                await _shutdown_event.wait()
                logger.info("Shutdown signal received, cleaning up...")

        except KeyboardInterrupt:
            logger.info("MAX Workers shutting down gracefully.")
        except Exception:
            logger.exception("Error occurred starting MAX Workers")
        finally:
            _shutdown_event = None

        logger.info("MAX Workers Started!")

    # Set up signal handlers
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        uvloop.run(run_workers())
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt caught within MAX, exiting gracefully.")


__all__ = ["start_workers"]
