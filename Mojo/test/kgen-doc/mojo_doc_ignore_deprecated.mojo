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

# Test that `--ignore-deprecated=<name>` suppresses only the named
# declaration's deprecation warning during doc generation, while other
# `@deprecated` declarations still warn. Covers the `mojo doc`/`kgen-doc`
# path used by the `mojo_doc` Bazel rule (a separate CLI/config surface from
# the main parser's `--ignore-deprecated`, see StabilityMarkers.cpp).

# RUN: kgen-doc --ignore-deprecated=ignoredFn %s -o /dev/null 2>&1 | FileCheck %s


@deprecated("ignoredFn is deprecated")
def ignoredFn():
    pass


@deprecated("notIgnoredFn is deprecated")
def notIgnoredFn():
    pass


def caller():
    # CHECK-NOT: warning: ignoredFn is deprecated
    ignoredFn()
    # CHECK: warning: notIgnoredFn is deprecated
    notIgnoredFn()
