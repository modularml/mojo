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

# Create user's private bin
mkdir -p "$HOME/.local/bin"

# If existent, prepend the user's private bin to PATH
if ! grep -q "user's private bin" "$HOME/.bashrc"; then
  cat "/var/tmp/snippets/rc.sh" >> "$HOME/.bashrc"
fi
if ! grep -q "user's private bin" "$HOME/.zshrc"; then
  cat "/var/tmp/snippets/rc.sh" >> "$HOME/.zshrc"
fi

# Enable Oh My Zsh plugins
sed -i "s/plugins=(git)/plugins=(docker docker-compose git git-lfs pip screen tmux vscode)/g" \
  "$HOME/.zshrc"

# Remove old .zcompdump files
rm -f "$HOME"/.zcompdump*
