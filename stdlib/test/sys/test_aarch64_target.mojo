# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# REQUIRES: aarch64
# COM: TODO (17471): Not all aarch64 have neon, so we need to guard against that,
# for now just require apple-silicon.
# REQUIRES: apple-silicon
# RUN: %mojo -debug-level %s

from sys.info import alignof, has_avx512f, has_neon, simdbitwidth
from testing import assert_false, assert_true, assert_equal


# CHECK-LABEL: test_arch_query
fn test_arch_query():
    print("== test_arch_query")

    # CHECK: True
    assert_true(has_neon())

    # CHECK: 128
    assert_equal(simdbitwidth(), 128)

    # CHECK: False
    assert_false(has_avx512f())


fn main():
    test_arch_query()
