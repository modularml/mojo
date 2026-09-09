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
#
# The verbs UCX plugin links with `-Wl,-z,undefs`, so nothing about its
# rdma-core symbols can fail the link. Unversioned ones bind to the
# IBVERBS_1.0 compat definitions, whose struct ibv_device ABI differs: device
# names read as garbage and UCX enumerates zero RDMA devices, with no error
# anywhere. Only a repin of UCX or rdma-core can reintroduce that, which is
# exactly when nobody is looking for it.

set -euo pipefail

readonly readelf="$1"
readonly plugin="$2"

unversioned=$("$readelf" --dyn-symbols "$plugin" |
  awk 'NF > 1 && $(NF-1) == "UND" && $NF ~ /^(ibv_|mlx5dv_)/ && $NF !~ /@/ { print $NF }')

if [[ -n "$unversioned" ]]; then
  echo "error: undefined rdma-core symbols carry no version tag, so they will" \
       "bind to the wrong ABI and UCX will enumerate zero devices:" >&2
  echo "$unversioned" >&2
  exit 1
fi
