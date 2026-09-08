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
#
# Test that, when no output file path argument is provided, `mojo build` creates
# files in the current workfing directory, using nice names that are based on
# the input file name.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm default_output_name || true
# RUN: %mojo-build %s
# RUN: test -x default_output_name

# RUN: rm default_output_name_2 || true
# RUN: %mojo-build %S/inputs/default_output_name_2.mojo
# RUN: test -x default_output_name_2


def main():
    pass
