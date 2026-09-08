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

# Two roots providing the same module under one namespace directory is an
# error (stricter than Python, which silently takes path order).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo %t.dir/two/foo
# RUN: echo "def hello():" > %t.dir/one/foo/test.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/test.mojo
# RUN: cp %t.dir/one/foo/test.mojo %t.dir/two/foo/test.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/one -I=%t.dir/two %s

# expected-error @+1 {{ambiguous import 'test': found}}
import foo.test
