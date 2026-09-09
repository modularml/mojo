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

# Test that Layout is not implicitly copyable.
# This is a common error that users were hitting,
# so it's good to make sure we don't regress.

# RUN: not kgen %s -elaborate 2>&1 | FileCheck %s

from layout import Layout, LayoutTensor


def kernel(tensor: LayoutTensor):
    # CHECK: cannot materialize comptime value of type 'Layout' to runtime because it is not 'ImplicitlyCopyable'
    var size = tensor.layout.size()
