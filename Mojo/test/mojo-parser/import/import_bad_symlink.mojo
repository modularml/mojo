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

# RUN: mkdir -p %t
# RUN: ln -s does-not-exist %t/non_existent_package.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t %s

# expected-error-re @+1 {{unable to locate module 'non_existent_package'}}
import non_existent_package

def main():
    pass
