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

# We can run this file with various targets.
# RUN: not mojo -target-triple not-a-valid-target %s 2>&1 | FileCheck %s --check-prefix=INVALID_TARGET
# INVALID_TARGET: mojo: error: failed to create target info: unknown target triple 'not-a-valid-target'

# RUN: %mojo %s 2>&1 | FileCheck %s


def main():
    # CHECK: hello world
    print("hello world")
