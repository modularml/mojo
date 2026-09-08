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


def __wrap_and_execute_main[
    main_func: def() thin -> None
](argc: Int, argv: Int) -> Int:
    return 0


def __wrap_and_execute_raising_main[
    main_func: def() thin raises -> None
](argc: Int, argv: Int) -> Int:
    return 0


def __mojo_main_prototype(argc: Int, argv: Int) -> Int:
    return 0
