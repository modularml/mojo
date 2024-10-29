#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2023, Olivier Benz
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

# Change ownership of the workspaces folder
sudo chown "$(id -u)":"$(id -g)" /workspaces

# Change ownership of the symlinked .magic folder
sudo chown "$(id -u)":"$(id -g)" /var/tmp/.magic
