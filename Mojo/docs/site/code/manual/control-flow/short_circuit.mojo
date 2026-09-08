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


def main():
    # Short-circuit "or" evaluation
    def true_func() -> Bool:
        print("Executing true_func")
        return True

    def false_func() -> Bool:
        print("Executing false_func")
        return False

    print('Short-circuit "or" evaluation')
    if true_func() or false_func():
        print("True result")

    # Short-circuit "and" evaluation
    print('Short-circuit "and" evaluation')
    if false_func() and true_func():
        print("True result")
