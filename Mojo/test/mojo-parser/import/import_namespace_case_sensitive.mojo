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

# A directory whose name differs only by case is not a namespace portion:
# portion matching stays case-exact even on case-insensitive filesystems.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo %t.dir/two/FOO
# RUN: echo "def hello():" > %t.dir/one/foo/test.mojo
# RUN: echo "    pass" >> %t.dir/one/foo/test.mojo
# RUN: echo "def hello2():" > %t.dir/two/FOO/test2.mojo
# RUN: echo "    pass" >> %t.dir/two/FOO/test2.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t.dir/one -I=%t.dir/two %s

import foo.test

# expected-error @+1 {{unable to locate module 'test2'}}
import foo.test2
