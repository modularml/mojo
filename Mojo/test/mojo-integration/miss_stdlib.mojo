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

# RUN: not kgen-translate -import-mojo -mojo-search-paths=nope %s 2>&1 | FileCheck %s
#


# CHECK: unable to locate module 'std'
# CHECK: def baz(ignore: Bool):
# CHECK: ^
# CHECK: 'std' is required for all normal mojo compiles.
# CHECK: If you see this either:
def baz(ignore: Bool):
    pass
