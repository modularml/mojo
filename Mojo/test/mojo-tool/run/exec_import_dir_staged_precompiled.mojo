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

# Staged incremental precompilation: packages precompiled one at a time into
# a plain (non-package) staging directory would resolve their dependencies
# from the `.mojoc`s already staged there — but importing a precompiled
# package through a directory is a nested mount, which is rejected: its
# recorded symbol roots resolve only for a top-level binding.
# TODO(MOCO-4487): once loading re-anchors recorded roots to the mount
# point, this flow resolves instead (restore the passing form of this test).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/stage/tmp
# RUN: mojo precompile %S/inputs/dir_precompiled/helper -o %t.dir/stage/tmp/helper.mojoc
# RUN: not mojo precompile -I %t.dir/stage %S/inputs/dir_precompiled/consumer -o %t.dir/stage/tmp/consumer.mojoc 2>&1 \
# RUN:   | FileCheck %s

# CHECK: error: precompiled package '{{.*}}helper.mojoc' must be imported directly from an import root, not as 'tmp.helper'

from tmp.consumer import foo


def main():
    # This program is unused while staged resolution is rejected; it stays
    # for the restored form of the test.
    foo()
