#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2024, Olivier Benz
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

set -e

# MAX SDK: Evaluate and set version
if [ "${MOJO_VERSION}" = "nightly" ]; then
  extDataDir=$HOME/.vscode-server/data/User/globalStorage/modular-mojotools.vscode-mojo-nightly
  while :
    do
    extDirs=( "$HOME"/.vscode-server/extensions/modular-mojotools.vscode-mojo-nightly* )
    [ "${#extDirs[@]}" -ge 2 ] && exit 1
    if [ -d "${extDirs[0]}" ]; then
      sdkVersion=$(jq -r '.sdkVersion' "${extDirs[0]}/package.json")
      break
    else
      sleep 1
    fi
  done
else
  extDataDir=$HOME/.vscode-server/data/User/globalStorage/modular-mojotools.vscode-mojo
  sdkVersion=$MOJO_VERSION
fi

# MAX SDK: Create symlink to /opt/modular
mkdir -p "$extDataDir/magic-data-home/envs"
ln -snf /opt/modular "$extDataDir/magic-data-home/envs/max"
mkdir -p "$extDataDir/versionDone/$sdkVersion"
