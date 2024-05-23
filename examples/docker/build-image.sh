#!/bin/sh
##===----------------------------------------------------------------------===##
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

# Usage
# ==========
# ./build-image.sh --mojo-version
#

# CLI option handling code
mojo_ver=${mojo_ver:=0.3}
container_engine=${container_engine:=docker}
extra_cap=${extra_cap:=}
while [ $# -gt 0 ]; do
        case "$1" in
                --use-podman)
                        container_engine=podman
                        extra_cap="--cap-add SYS_PTRACE"
                        ;;
                --mojo-version)
                        mojo_ver="$2"
                        shift
                        ;;
                --*)
                        echo "Unrecognized option $1"
                        ;;
        esac
        shift $(( $# > 0 ? 1 : 0 ))
done

build_image() {
        echo "# Building image with ${container_engine}..."
        ${container_engine} build --no-cache ${extra_cap} \
           --pull -t modular/mojo-v${mojo_ver}-`date '+%Y%d%m-%H%M'` \
           --file Dockerfile.mojosdk .
}

# Wrap the build in a function
build_image
