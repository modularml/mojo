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

# Within a namespace, a module in any portion beats a plain directory in an
# earlier portion: plain directories carry no marker, so a stray non-Mojo
# directory must not shadow a real module.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/one/foo/util %t.dir/two/foo
# RUN: echo "def util_fn():" > %t.dir/two/foo/util.mojo
# RUN: echo "    pass" >> %t.dir/two/foo/util.mojo
# RUN: %parse-mojo-isolated -I=%t.dir/one -I=%t.dir/two %s

from foo.util import util_fn
