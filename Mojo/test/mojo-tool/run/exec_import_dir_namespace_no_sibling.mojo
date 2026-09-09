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

# A module inside a namespace portion searches from its import root, not its
# own directory: a bare import cannot name a sibling (Python removed such
# implicit relative imports in PEP 328); the dotted path resolves it, across
# portions included.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo %t.dir/two/foo
# RUN: echo 'import test2' > %t.dir/one/foo/test.mojo
# RUN: cp %S/inputs/namespace/test2.mojo %t.dir/one/foo/test2.mojo
# RUN: not mojo run -I %t.dir/one -I %t.dir/two %s 2>&1 | FileCheck %s --check-prefix=BARE

# BARE: error: unable to locate module 'test2'

# A module in the *other* portion reaches it with the dotted path.
# RUN: rm %t.dir/one/foo/test.mojo
# RUN: printf 'import foo.test2\n\n\ndef hello():\n    foo.test2.hello2()\n' > %t.dir/two/foo/test.mojo
# RUN: mojo run -I %t.dir/one -I %t.dir/two %s | FileCheck %s --check-prefix=DOTTED

# DOTTED: namespace test2 two

import foo.test


def main():
    foo.test.hello()
