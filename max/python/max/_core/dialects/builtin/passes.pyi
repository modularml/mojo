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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum

import max._core

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

class GreedySimplifyRegionLevel(enum.Enum):
    normal = 1

    disabled = 0

    aggressive = 2

class ConvergenceFailureAction(enum.Enum):
    warn = 0

    error = 1

    silent = 2

def RemoveDeadValuesPass(canonicalize: bool = True) -> max._core.Pass:
    """
    The goal of this pass is optimization (reducing runtime) by removing
    unnecessary instructions. Unlike other passes that rely on local information
    gathered from patterns to accomplish optimization, this pass uses a full
    analysis of the IR, specifically, liveness analysis, and is thus more
    powerful.

    Currently, this pass performs the following optimizations:
    (A) Removes function arguments that are not live,
    (B) Removes function return values that are not live across all callers of
    the function,
    (C) Removes unneccesary operands, results, region arguments, and region
    terminator operands of region branch ops, and,
    (D) Removes simple and region branch ops that have all non-live results and
    don't affect memory in any way.

    Here, a "simple op" refers to an op that isn't a symbol op, symbol-user op,
    region branch op, branch op, region branch terminator op, or return-like.

    It is noteworthy that we do not refer to non-live values as "dead" in this
    file to avoid confusing it with dead code analysis's "dead", which refers to
    unreachable code (code that never executes on hardware) while "non-live"
    refers to code that executes on hardware but is unnecessary. Thus, while the
    removal of dead code helps little in reducing runtime, removing non-live
    values should theoretically have significant impact (depending on the amount
    removed).

    It is also important to note that unlike other passes (like `canonicalize`)
    that apply op-specific optimizations through patterns, this pass uses
    different interfaces to handle various types of ops and tries to cover all
    existing ops through these interfaces.

    It is because of its reliance on (a) liveness analysis and (b) interfaces
    that makes it so powerful that it can optimize ops that don't have a
    canonicalizer and even when an op does have a canonicalizer, it can perform
    more aggressive optimizations, as observed in the test files associated with
    this pass.

    Example of optimization (A):-

    ```
    int add_2_to_y(int x, int y) {
      return 2 + y
    }

    print(add_2_to_y(3, 4))
    print(add_2_to_y(5, 6))
    ```

    becomes

    ```
    int add_2_to_y(int y) {
      return 2 + y
    }

    print(add_2_to_y(4))
    print(add_2_to_y(6))
    ```

    Example of optimization (B):-

    ```
    int, int get_incremented_values(int y) {
      store y somewhere in memory
      return y + 1, y + 2
    }

    y1, y2 = get_incremented_values(4)
    y3, y4 = get_incremented_values(6)
    print(y2)
    ```

    becomes

    ```
    int get_incremented_values(int y) {
      store y somewhere in memory
      return y + 2
    }

    y2 = get_incremented_values(4)
    y4 = get_incremented_values(6)
    print(y2)
    ```

    Example of optimization (C):-

    Assume only `%result1` is live here. Then,

    ```
    %result1, %result2, %result3 = scf.while (%arg1 = %operand1, %arg2 = %operand2) {
      %terminator_operand2 = add %arg2, %arg2
      %terminator_operand3 = mul %arg2, %arg2
      %terminator_operand4 = add %arg1, %arg1
      scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3, %terminator_operand4
    } do {
    ^bb0(%arg3, %arg4, %arg5):
      %terminator_operand6 = add %arg4, %arg4
      %terminator_operand5 = add %arg5, %arg5
      scf.yield %terminator_operand5, %terminator_operand6
    }
    ```

    becomes

    ```
    %result1, %result2 = scf.while (%arg2 = %operand2) {
      %terminator_operand2 = add %arg2, %arg2
      %terminator_operand3 = mul %arg2, %arg2
      scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3
    } do {
    ^bb0(%arg3, %arg4):
      %terminator_operand6 = add %arg4, %arg4
      scf.yield %terminator_operand6
    }
    ```

    It is interesting to see that `%result2` won't be removed even though it is
    not live because `%terminator_operand3` forwards to it and cannot be
    removed. And, that is because it also forwards to `%arg4`, which is live.

    Example of optimization (D):-

    ```
    int square_and_double_of_y(int y) {
      square = y ^ 2
      double = y * 2
      return square, double
    }

    sq, do = square_and_double_of_y(5)
    print(do)
    ```

    becomes

    ```
    int square_and_double_of_y(int y) {
      double = y * 2
      return double
    }

    do = square_and_double_of_y(5)
    print(do)
    ```

    Note: If `canonicalize` is set to "false", this pass does not remove any
    block arguments / op results from ops that implement the
    RegionBranchOpInterface. Instead, it just sets dead operands to
    "ub.poison".
    """
