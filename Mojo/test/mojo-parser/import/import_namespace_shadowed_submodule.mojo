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

# A module shadows a same-named plain directory across portions completely:
# the directory's contents are unreachable through the name, in either
# root order (Python's rule; with util.py present, 'foo.util' is
# not a package).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo/util %t.dir/two/foo
# RUN: echo "def inner_fn():" > %t.dir/one/foo/util/inner.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/util/inner.mojo
# RUN: echo "def util_fn():" > %t.dir/two/foo/util.mojo
# RUN: echo "    pass" >> %t.dir/two/foo/util.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/one -I=%t.dir/two %s
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/two -I=%t.dir/one %s

# expected-error @+1 {{'util' is a module, not a package; it has no nested module or package 'inner'}}
import foo.util.inner
