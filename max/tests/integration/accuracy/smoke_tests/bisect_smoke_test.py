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
"""
Git bisect helper for smoke_test.py regressions.

Examples:
    # Find commit that broke the smoke test
    python3 bisect_smoke_test.py --model modularai/Llama-3.1-8B-Instruct-GGUF \
        --good-commit abc123 --bad-commit def456

    # Find commit where text accuracy dropped below 0.7
    python3 bisect_smoke_test.py --model modularai/Llama-3.1-8B-Instruct-GGUF \
        --good-commit abc123 --bad-commit def456 --text-threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BISECT_GOOD = 0
BISECT_BAD = 1
BISECT_SKIP = 125


def find_repo_root() -> Path:
    """Locate repo root so we can run bazel from the correct directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def clear_caches() -> None:
    """Clear build caches to ensure clean state after switching commits."""
    derived = find_repo_root() / ".derived"
    if not derived.exists():
        return
    for d in sorted(derived.iterdir()):
        if d.is_dir() and "cache" in d.name:
            print(f"Clearing cache: {d}")
            shutil.rmtree(d)


def run_smoke_test(model: str, output_path: Path) -> bool | None:
    """Returns True on success, False on failure, None on timeout."""

    repo_root = find_repo_root()
    cmd = [
        str(repo_root / "bazelw"),
        "run",
        "smoke-test",
        "--",
        model,
        "--framework",
        "max-ci",
        "--output-path",
        str(output_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=repo_root)
    try:
        returncode = proc.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        print("Smoke test timed out after 30 minutes, killing process")
        proc.kill()
        proc.wait()
        return None

    return returncode == 0


def parse_results(output_path: Path, model: str) -> dict[str, float] | None:
    """Extract accuracy metrics from smoke test output for threshold comparison."""
    model_dir = output_path / model.strip().replace("/", "__")
    metrics_file = model_dir / "eval_metrics.json"

    if not metrics_file.exists():
        return None

    metrics = json.loads(metrics_file.read_text())
    results = {}
    for entry in metrics:
        task = entry["eval_task"]
        accuracy = entry["accuracy"]
        if accuracy is not None:
            if "chartqa" in task:
                results["vision"] = accuracy
            elif "gsm8k" in task:
                results["text"] = accuracy
    return results


def check_thresholds(
    results: dict[str, float],
    text_threshold: float | None,
    vision_threshold: float | None,
) -> bool:
    """Determine if accuracy regression occurred by comparing against thresholds."""
    if text_threshold is not None:
        if results["text"] < text_threshold:
            print(f"Text accuracy {results['text']} < {text_threshold}")
            return False
    if vision_threshold is not None:
        if results["vision"] < vision_threshold:
            print(f"Vision accuracy {results['vision']} < {vision_threshold}")
            return False
    return True


def test_commit(
    model: str,
    text_threshold: float | None,
    vision_threshold: float | None,
) -> int:
    """Called by git bisect to test each commit and return appropriate exit code."""
    clear_caches()
    has_thresholds = text_threshold is not None or vision_threshold is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        success = run_smoke_test(model, output_path)

        if success is None:
            return BISECT_SKIP
        if not success:
            return BISECT_SKIP if has_thresholds else BISECT_BAD

        if not has_thresholds:
            return BISECT_GOOD

        results = parse_results(output_path, model)
        if results is None:
            return BISECT_SKIP

        return (
            BISECT_GOOD
            if check_thresholds(results, text_threshold, vision_threshold)
            else BISECT_BAD
        )


def start_bisect(
    script_path: Path,
    model: str,
    good_commit: str,
    bad_commit: str,
    text_threshold: float | None,
    vision_threshold: float | None,
) -> int:
    """Initialize git bisect and run it with this script as the test command."""
    repo_root = find_repo_root()

    test_cmd = [
        sys.executable,
        str(script_path),
        "--test",
        "--model",
        model,
    ]
    if text_threshold is not None:
        test_cmd.extend(["--text-threshold", str(text_threshold)])
    if vision_threshold is not None:
        test_cmd.extend(["--vision-threshold", str(vision_threshold)])

    subprocess.run(["git", "bisect", "start"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "bisect", "bad", bad_commit], cwd=repo_root, check=True
    )
    subprocess.run(
        ["git", "bisect", "good", good_commit], cwd=repo_root, check=True
    )

    print(f"Running bisect: {' '.join(test_cmd)}")
    result = subprocess.run(["git", "bisect", "run"] + test_cmd, cwd=repo_root)
    subprocess.run(["git", "bisect", "log"], cwd=repo_root)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Git bisect helper for smoke test regressions"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--good-commit", help="Known good commit")
    parser.add_argument("--bad-commit", help="Known bad commit")
    parser.add_argument(
        "--text-threshold", type=float, help="Min text accuracy"
    )
    parser.add_argument(
        "--vision-threshold", type=float, help="Min vision accuracy"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test current commit (used by git bisect)",
    )
    args = parser.parse_args()

    if args.test:
        try:
            result = test_commit(
                args.model, args.text_threshold, args.vision_threshold
            )
        except Exception as e:
            print(f"Error: {e}")
            result = BISECT_SKIP
        sys.exit(result)

    if not args.good_commit or not args.bad_commit:
        parser.error("--good-commit and --bad-commit are required")

    # Copy script outside repo so git checkout doesn't overwrite it
    with tempfile.TemporaryDirectory(prefix="bisect_script_") as tmpdir:
        script_path = Path(tmpdir) / "bisect_smoke_test.py"
        shutil.copy2(Path(__file__).resolve(), script_path)
        sys.exit(
            start_bisect(
                script_path,
                args.model,
                args.good_commit,
                args.bad_commit,
                args.text_threshold,
                args.vision_threshold,
            )
        )


if __name__ == "__main__":
    main()
