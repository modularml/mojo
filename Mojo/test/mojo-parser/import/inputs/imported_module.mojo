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

# This file is imported by 'import.mojo' as part of testing import functionality,
# and does not include any useful testing by itself.


# expected-note @+1 {{function declared here}}
def imported_fn():
    return


def _ignored_wildcard_fn():
    return


# Intentionally the same name as the package.
def imported_module():
    pass
