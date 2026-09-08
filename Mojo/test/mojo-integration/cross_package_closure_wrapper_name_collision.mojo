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
# RUN: mkdir -p %t.contains-dupe-dir
# RUN: mojo precompile %S/inputs/containsDupe -o %t.contains-dupe-dir/containsDupe.mojoc
# RUN: mojo -I %t.contains-dupe-dir %s 2>&1

# Ensure the identical wrapper defined in containsDupe
# is not pulled into this file module scope,
# which would cause name conflicts because of the duplicate
# closure defined here. TODO: dedupe struct wrappers like traits are today?


from containsDupe import *


def main() raises:
    def identical() {var} -> String:
        return "hello"

    # CHECK: hello
    consume(identical)
