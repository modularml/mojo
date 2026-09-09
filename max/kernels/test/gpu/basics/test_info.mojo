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

from max.gpu.host.info import *
from std.testing import *


def main() raises:
    assert_true(materialize[H100.compute > A100.compute]())
    assert_false(materialize[H100.compute < A100.compute]())
    assert_true(materialize[A100.compute < H100.compute]())
    assert_false(materialize[A100.compute > MI300X.compute]())
