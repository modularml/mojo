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

# RUN: not %mojo-build %s -o %t/this_dir_cant_exist/output 2>&1 | FileCheck %s --check-prefix=INVALID_OUTPUT_DIR
# INVALID_OUTPUT_DIR: error: unable to write file. The path '{{.*}}' does not exist.


def main():
    return
