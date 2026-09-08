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

# Test that @stable(recursive=True) suppresses warnings for methods defined in
# extensions of the imported type, not just methods defined in the struct body.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

@stable(recursive=True)
from test_std_mock import UnstableStruct


def test_extension_method_suppressed():
    # extension_method is defined in a __extension block in the mock package,
    # not in UnstableStruct's body.  The suppression should still apply,
    # the user does not care _where_ the methods are defined.
    var x = UnstableStruct()
    _ = x.extension_method()


# CHECK-NOT: warn_import_extension.mojo:{{.*}}warning: use of unstable API
