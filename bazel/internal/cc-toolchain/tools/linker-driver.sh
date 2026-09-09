#!/bin/bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##

set -euo pipefail

if [[ $OSTYPE == darwin* ]]; then
  platform=macos
else
  platform="linux-$(uname -m)"
fi

clang_root="$PWD/external/+http_archive+clang-$platform"
# File paths in tests differ
if [[ ! -d "$clang_root" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$script_dir/../../../.."
  clang_root="$repo_root/../+http_archive+clang-$platform"
fi

readonly clang="$clang_root/bin/clang++"
readonly dsymutil="$clang_root/bin/dsymutil"

ifs_input=""
ifs_output=""
dsym_path=""
binary_path=""
linker_args=()
for arg in "$@"; do
  case "$arg" in
    --modular-ifs-input=*) ifs_input="${arg#*=}" ;;
    --modular-ifs-output=*) ifs_output="${arg#*=}" ;;
    --modular-dsym-path=*) dsym_path="${arg#*=}" ;;
    --modular-binary-path=*) binary_path="${arg#*=}" ;;
    *) linker_args+=("$arg") ;;
  esac
done

"$clang" "${linker_args[@]}"

if [[ -n "$dsym_path" ]]; then
  "$dsymutil" -o "$dsym_path" "$binary_path"
fi

if [[ "${BUILD_IFS:-}" == "yes" ]]; then
  if [[ -z "$ifs_input" || -z "$ifs_output" ]]; then
    echo "error: interface library input and output paths are required" >&2
    exit 1
  fi

  if [[ $OSTYPE == darwin* ]]; then
    ifs_platform=mac
  elif [[ $(uname -m) == "x86_64" ]]; then
    ifs_platform=intel
  else
    ifs_platform=graviton
  fi

  ifs_root="$PWD/external/+http_archive+llvm-ifs/tools/$ifs_platform"

  if [[ "${MACOS:-}" == "true" ]]; then
    "$ifs_root/llvm-readtapi.stripped" -arch arm64 -extract "$ifs_input" -o "$ifs_output"
  else
    "$ifs_root/llvm-ifs.stripped" "$ifs_input" --output-elf="$ifs_output"
  fi
fi
