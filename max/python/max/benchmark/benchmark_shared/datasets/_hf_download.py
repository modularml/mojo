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

"""Dataset download helper that survives a racy Hugging Face cache."""

from __future__ import annotations

import logging
from typing import Any

import huggingface_hub
from huggingface_hub import errors as hf_hub_errors

_logger = logging.getLogger(__name__)


# Mirrors the weight-download retry in ``max.pipelines.weights.hf_utils``;
# duplicated to keep ``benchmark_shared`` decoupled from the pipelines package.
def hf_hub_download_with_retry(**kwargs: Any) -> str:
    """Calls ``hf_hub_download``, retrying past a racy ``.incomplete`` entry.

    A concurrent download or an evicted cache can delete the ``.incomplete``
    temp before ``hf_hub_download`` renames it, raising ``FileNotFoundError``;
    a clean re-fetch repairs it. An offline/uncached miss is not retryable.
    """
    try:
        return huggingface_hub.hf_hub_download(**kwargs)
    except hf_hub_errors.LocalEntryNotFoundError:
        raise  # offline/uncached miss, not retryable
    except FileNotFoundError:
        _logger.warning(
            "Retrying download of %s with force_download.",
            kwargs.get("filename", kwargs.get("repo_id")),
        )
        return huggingface_hub.hf_hub_download(
            **{**kwargs, "force_download": True}
        )
