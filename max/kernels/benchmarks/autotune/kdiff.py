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
import os
import subprocess
import sys
from time import sleep

import click
import yaml


def shell(arg_str: str, check: bool = False, verbose=True):  # noqa: ANN001, ANN201
    if not arg_str:
        return None
    print(f"$ [{arg_str}]")
    p = subprocess.run(
        arg_str,
        shell=True,
        check=check,
        capture_output=True,
        encoding="utf-8",
    )
    if verbose and p.stderr:
        print("> {p.stderr}")
    return p.stdout.split("\n")


def export_env(key, val) -> None:  # noqa: ANN001
    os.environ[key] = val


def export_arg_env() -> None:
    """
    export ARGO_SERVER='argo-workflows-staging-release-cea88dd5-server.argo.svc.platform-staging-karpenter-eks-cluster-3b48ea0.local:80'
    export ARGO_HTTP1=true
    export ARGO_SECURE=false
    export ARGO_BASE_HREF=
    export ARGO_TOKEN=''
    export ARGO_NAMESPACE=argo ; # or whatever your namespace is
    export KUBECONFIG=/dev/null ; # recommended
    """

    export_list = [
        [
            "ARGO_SERVER",
            "argo-workflows-staging-release-cea88dd5-server.argo.svc.platform-staging-karpenter-eks-cluster-3b48ea0.local:80",
        ],
        ["ARGO_HTTP1", "true"],
        ["ARGO_SECURE", "false"],
        ["ARGO_BASE_HREF", ""],
        ["ARGO_TOKEN", ""],
        ["ARGO_NAMESPACE", "argo"],
        ["KUBECONFIG", "/dev/null"],
    ]

    for k, v in export_list:
        export_env(k, v)


def check_argo_workflow_exists(git_sha):  # noqa: ANN001, ANN201
    # check if workflow exists and return the output of argo list as yaml else None
    result = shell(
        f'argo list --prefix "kernels-{git_sha}" --completed -o yaml'
    )
    if result[0] != "[]":
        result_yaml = yaml.safe_load("\n".join(result))
        return result_yaml
    return None


def search_workflows(branch_sha, last_n_commits=100, timeout_secs=60):  # noqa: ANN001, ANN201
    print(
        f"Checking {last_n_commits} of origin/main for existing CI workflows"
        " (baseline)"
    )
    main_git_sha_list = shell(
        f"git log --pretty=format:'%h' -{last_n_commits} origin/main"
    )
    main_sha = None
    for git_sha in main_git_sha_list:
        ref_main_yaml = check_argo_workflow_exists(git_sha=git_sha)
        if ref_main_yaml:
            main_sha = git_sha
            print(f"[Found /origin/main argo workflow for sha {main_sha}]")
            break

    while True:
        ref_branch_yaml = check_argo_workflow_exists(branch_sha)
        if not ref_branch_yaml:
            print(f"couldn't find, check back in {timeout_secs} seconds")
            sleep(timeout_secs)
        else:
            break

    return [main_sha, ref_main_yaml, ref_branch_yaml]


def download_artifacts(
    target_name,  # noqa: ANN001
    main_sha,  # noqa: ANN001
    branch_sha,  # noqa: ANN001
    ref_main_yaml,  # noqa: ANN001
    ref_branch_yaml,  # noqa: ANN001
    output_dir,  # noqa: ANN001
    extension,  # noqa: ANN001
) -> None:
    # print("found workflow for ", current_sha)
    # TODO: convert to PATH

    shell(f"mkdir -p {output_dir}/main")
    shell(f"mkdir -p {output_dir}/branch")

    artifact_name = "result-dir"
    shell(
        f"argo cp {ref_main_yaml[0]['metadata']['name']} --artifact-name"
        f" {artifact_name} {output_dir}/main"
    )
    shell(
        f"argo cp {ref_branch_yaml[0]['metadata']['name']} --artifact-name"
        f" {artifact_name} {output_dir}/branch"
    )

    result_main = shell(f"find {output_dir}/main/argo/|grep result-dir.tgz")
    result_branch = shell(f"find {output_dir}/branch/argo/|grep result-dir.tgz")

    shell(f"tar -xf {result_main[0]} -C {output_dir}")
    shell(f"tar -xf {result_branch[0]} -C {output_dir}")

    main_sha8 = main_sha[:8]
    branch_sha8 = branch_sha[:8]

    target_path_main = shell(
        f"find {output_dir}/benchmark-results/_{main_sha8}/|grep"
        f" {target_name}|grep csv"
    )
    target_path_branch = shell(
        f"find {output_dir}/benchmark-results/_{branch_sha8}/|grep"
        f" {target_name}|grep csv"
    )
    kp = shell(
        "./bazelw run -- //max/kernels/benchmarks/autotune:kplot"
        f" {target_path_main[0]} {target_path_branch[0]} "
        f"--label=main/{target_name} --label=branch/{target_name} "
        f" -o {output_dir}/main_vs_branch_{target_name} -x {extension}"
    )
    print("\n".join(kp))


def compare_to_main(target_name, branch_sha, output_dir, extension) -> None:  # noqa: ANN001
    export_arg_env()
    main_sha, ref_main_yaml, ref_branch_yaml = search_workflows(
        branch_sha=branch_sha
    )
    download_artifacts(
        target_name=target_name,
        main_sha=main_sha,
        branch_sha=branch_sha,
        ref_main_yaml=ref_main_yaml,
        ref_branch_yaml=ref_branch_yaml,
        output_dir=output_dir,
        extension=extension,
    )


help_str = """
kdiff: compare performance with origin/main

- Setup gh

    gh auth login

    ? What account do you want to log into? GitHub.com

    ? What is your preferred protocol for Git operations? HTTPS

    ? Authenticate Git with your GitHub credentials? Yes

    ? How would you like to authenticate GitHub CLI? Login with a web browser

    ! First copy your one-time code: ABCD-1234

    - Press Enter to open github.com in your browser...
"""


@click.command(
    help=help_str,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "-help", "--help"]},
)
@click.option(
    "--run", "-r", "run_branch", help="Run the workflow on CI.", multiple=False
)
@click.option(
    "--output-dir",
    "-o",
    "output_path",
    default="./",
    help="Path to output directory (tmp).",
)
@click.option(
    "--extension",
    "-x",
    "extension",
    default="png",
    type=click.STRING,
    help="output extension",  # TODO: complete docstring
)
@click.option(
    "--target",
    "targets",
    default=(),
    help="Set extra params from CLI.",
    multiple=True,
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
@click.argument("branch_sha", nargs=-1, type=click.UNPROCESSED)
def cli(
    branch_sha,  # noqa: ANN001
    run_branch,  # noqa: ANN001
    output_path,  # noqa: ANN001
    extension,  # noqa: ANN001
    targets,  # noqa: ANN001
    verbose,  # noqa: ANN001
) -> None:
    assert len(branch_sha) == 1
    assert output_path
    assert len(targets) > 0

    branch_sha = shell(f"git rev-parse --short=8 {branch_sha[0]}")[0]
    branch_name_list = shell(f"git branch -a --contains {branch_sha}")

    found = False
    if run_branch:
        for b in branch_name_list:
            if run_branch in b:
                found = True
                break
        assert found

        print("Kickoff")
        shell(
            f'gh workflow run "Scheduled Kernels Benchmarks" --ref {run_branch}'
        )
        print(
            "Check here"
            " [https://github.com/modularml/modular/actions/workflows/scheduled-kernels-benchmarks.yaml]"
        )

    if not verbose:
        sys.tracebacklimit = 1

    for tn in targets:
        compare_to_main(
            target_name=tn,
            branch_sha=branch_sha,
            output_dir=output_path,
            extension=extension,
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    main()
