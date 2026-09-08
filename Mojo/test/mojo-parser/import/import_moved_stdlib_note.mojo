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

# A failed import under a stdlib subpackage relocated to `max` carries a
# migration note. The mock `std` in test-packages has none of those
# subpackages, but does have `collections`.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics %s
# RUN: not %parse-mojo-isolated -split-input-file %s 2>&1 \
# RUN:   | FileCheck --check-prefix=NONOTE %s

# expected-error @+2 {{unable to locate module 'runtime'}}
# expected-note @+1 {{many stdlib items recently moved to the `max` package}}
import std.runtime

# // -----

# expected-error @+2 {{unable to locate module 'gpu'}}
# expected-note @+1 {{many stdlib items recently moved to the `max` package}}
from std.gpu import DeviceContext

# // -----

# expected-error @+2 {{unable to locate module 'algorithm'}}
# expected-note @+1 {{many stdlib items recently moved to the `max` package}}
from std.algorithm import vectorize

# // -----

# A subpackage that did not move gets the bare error. The NONOTE checks run
# last, so the trailing NONOTE-NOT covers everything after this error.

# NONOTE: error: unable to locate module 'not_a_subpackage'
# NONOTE-NOT: max` package
# expected-error @+1 {{unable to locate module 'not_a_subpackage'}}
import std.not_a_subpackage
