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

# Helper module imported by ../signature_only_imports.mojo. Not a test itself
# (excluded via lit.cfg.py). The distinctive integer literals in the body let
# the importing test assert whether this body was resolved.


def compute_secret_value() -> Int:
    var a = 7
    var b = 6
    return a * b
