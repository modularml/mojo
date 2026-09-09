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

# A source package anywhere on the import path beats plain-directory
# namespace portions, even in a later root (Python's rule: namespace
# packages are the fallback when no regular package exists). The symbol
# only exists in the package's copy, so resolving it proves the pick.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo %t.dir/two/foo
# RUN: echo "def portion_hello():" > %t.dir/one/foo/test.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/test.mojo
# RUN: touch %t.dir/two/foo/__init__.mojo
# RUN: echo "def package_hello():" > %t.dir/two/foo/test.mojo
# RUN: echo "    pass" >> %t.dir/two/foo/test.mojo
# RUN: %parse-mojo-isolated -I=%t.dir/one -I=%t.dir/two %s

from foo.test import package_hello
