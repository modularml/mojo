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
"""Utilities for loading and caching test data.

Example command to upload an image:

```bash
aws s3 cp qwen2_5vl_instruct_image_a.jpg "s3://modular-bazel-artifacts-public/artifacts/model_testdata/qwen2_5vl_instruct_image_a.jpg"
```

Example usage:


```python
from test_common.storage import load_image

image = load_image("s3://modular-bazel-artifacts-public/artifacts/model_testdata/qwen2_5vl_instruct_image_a.jpg")
```

"""

from __future__ import annotations

import os
import tarfile
import tempfile

import boto3
from botocore.exceptions import (
    NoCredentialsError,
    SSOTokenLoadError,
    UnauthorizedSSOTokenError,
)
from PIL import Image

_DEFAULT_CACHE_DIR = "~/.cache/modular/testdata"
_S3_BUCKET = "modular-bazel-artifacts-public"
_S3_PREFIX = f"s3://{_S3_BUCKET}/"


def load_from_s3(s3_path: str, cache_dir: str | None = None) -> str:
    # This unnecessarily checks that the bucket == _S3_BUCKET, we can
    # make this more general when needed.
    if not s3_path.startswith(_S3_PREFIX):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    s3_path = s3_path[len(_S3_PREFIX) :]
    cache_dir = cache_dir or os.path.expanduser(_DEFAULT_CACHE_DIR)
    local_path = os.path.join(cache_dir, s3_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        try:
            s3 = boto3.client("s3")
            s3.download_file(
                _S3_BUCKET,
                s3_path,
                local_path,
            )
        except UnauthorizedSSOTokenError as e:
            # A rejected token means the caller meant to authenticate, so
            # silently downgrading them to anonymous would hide the problem.
            raise RuntimeError(
                "AWS authentication failed. Please run 'aws sso login' to "
                "refresh your credentials."
            ) from e
        except (SSOTokenLoadError, NoCredentialsError):
            # The bucket is public, so a stale SSO session or a machine with no
            # AWS config at all can still read it unsigned.
            print("No usable AWS credentials, retrying anonymously")
            s3 = boto3.client(
                "s3", aws_access_key_id="", aws_secret_access_key=""
            )
            s3._request_signer.sign = lambda *args, **kwargs: None
            s3.download_file(
                _S3_BUCKET,
                s3_path,
                local_path,
            )
    return local_path


def load_bytes(s3_path: str, cache_dir: str | None = None) -> bytes:
    with open(load_from_s3(s3_path, cache_dir), "rb") as f:
        return f.read()


def load_image(image_path: str, cache_dir: str | None = None) -> Image.Image:
    return _convert_image_mode(
        Image.open(load_from_s3(image_path, cache_dir)), "RGB"
    )


def _rgba_to_rgb(
    image: Image.Image,
    background_color=(255, 255, 255),  # noqa: ANN001
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def _convert_image_mode(image: Image.Image, to_mode: str):  # noqa: ANN202
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return _rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


def load_from_tar(tar_path: str) -> str:
    """Loads and untars a file from S3, or disk"""
    if tar_path.startswith("s3://"):
        on_disk_tar_path = load_from_s3(tar_path)
    else:
        on_disk_tar_path = tar_path
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(on_disk_tar_path, "r:gz") as tar:
        tar.extractall(path=temp_dir)
    return temp_dir
