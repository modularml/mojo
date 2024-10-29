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

# Symlink .magic folder to /var/tmp/.magic
# Conda does not work on bind mounts...
mkdir -p /var/tmp/.magic
ln -snf /var/tmp/.magic .magic

# Install all dependencies
if [ -f /var/tmp/magicenv ]; then
  . /var/tmp/magicenv
fi
"${MAGIC_BIN_DIR}${MAGIC_BIN_DIR:+/}magic" install
