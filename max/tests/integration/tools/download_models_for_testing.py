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

from __future__ import annotations

import os
from dataclasses import dataclass

import click
import huggingface_hub


@dataclass(frozen=True)
class RepoDownloadRequest:
    repo_id: str
    revision: str | None
    revision_source: str


def parse_repo_spec(repo_spec: str) -> tuple[str, str | None]:
    repo_id, separator, revision = repo_spec.rpartition("@")
    if not separator:
        return repo_spec, None
    if not repo_id or not revision:
        raise click.ClickException(
            f"Invalid repo spec {repo_spec!r}. "
            "Use 'org/model' or 'org/model@revision'."
        )
    return repo_id, revision


def resolve_download_request(repo_spec: str) -> RepoDownloadRequest:
    repo_id, explicit_revision = parse_repo_spec(repo_spec)
    if explicit_revision is not None:
        return RepoDownloadRequest(
            repo_id=repo_id,
            revision=explicit_revision,
            revision_source="explicit revision",
        )

    return RepoDownloadRequest(
        repo_id=repo_id,
        revision=None,
        revision_source="default branch",
    )


def download_repo(
    request: RepoDownloadRequest,
    *,
    force_download: bool,
) -> str:
    return huggingface_hub.snapshot_download(
        repo_id=request.repo_id,
        force_download=force_download,
        revision=request.revision,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("repo_specs", nargs=-1, required=True)
@click.option(
    "--force-download",
    is_flag=True,
    help="Redownload snapshots even if the requested revision is already cached.",
)
def main(repo_specs: tuple[str, ...], force_download: bool) -> None:
    """Download one or more Hugging Face model repos into the local cache."""
    if (
        not os.getenv("HF_TOKEN")
        and not huggingface_hub.constants.HF_HUB_OFFLINE
    ):
        click.echo(
            "HF_TOKEN is not set. Public repos may still work, but gated repos "
            "will fail.",
            err=True,
        )

    failures: list[str] = []
    for repo_spec in repo_specs:
        request = resolve_download_request(repo_spec)
        revision_display = request.revision or "default branch"
        click.echo(
            f"Downloading {request.repo_id} "
            f"(revision: {revision_display}, "
            f"source: {request.revision_source})..."
        )
        try:
            snapshot_path = download_repo(
                request,
                force_download=force_download,
            )
        except Exception as error:
            failures.append(
                f"{request.repo_id} (revision: {revision_display}): {error}"
            )
            continue

        click.echo(f"Cached at {snapshot_path}")

    if failures:
        raise click.ClickException("\n".join(failures))


if __name__ == "__main__":
    main()
