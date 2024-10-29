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

# Install magic
curl -ssL https://magic.modular.com | MAGIC_NO_PATH_UPDATE=1 bash

# Export MAGIC_BIN_DIR for postAttachCommand
. <(curl -ssL https://magic.modular.com | grep '^MODULAR_HOME\|^BIN_DIR')
echo "MAGIC_BIN_DIR=${BIN_DIR}" | sudo tee -a /var/tmp/magicenv

# Append the magic bin to PATH
echo -e "\n# Append the user's modular bin dir to PATH\nif [[ \"\$PATH\" != *\"${BIN_DIR}\"* ]] ; then\n    PATH=\"\$PATH:${BIN_DIR}\"\nfi" >> "$HOME/.bashrc"
echo -e "\n# Append the user's modular bin dir to PATH\nif [[ \"\$PATH\" != *\"${BIN_DIR}\"* ]] ; then\n    PATH=\"\$PATH:${BIN_DIR}\"\nfi" >> "$HOME/.zshrc"

# If existent, prepend the user's private bin to PATH
cat "/var/tmp/snippets/rc.sh" >> "$HOME/.bashrc"
cat "/var/tmp/snippets/rc.sh" >> "$HOME/.zshrc"

# Enable auto-completion for magic
cat "/var/tmp/snippets/rc2.sh" >> "$HOME/.bashrc"
cat "/var/tmp/snippets/rc2.sh" >> "$HOME/.zshrc"

# Enable Oh My Zsh plugins
sed -i "s/plugins=(git)/plugins=(docker docker-compose git pip vscode)/g" \
  "$HOME/.zshrc"

# Remove old .zcompdump files
rm -f "$HOME"/.zcompdump*
