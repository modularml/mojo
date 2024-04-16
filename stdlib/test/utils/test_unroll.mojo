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
# RUN: %mojo %s | FileCheck %s

from utils import StaticIntTuple, unroll


# CHECK-LABEL: test_unroll
fn test_unroll():
    print("test_unroll")

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    @parameter
    fn func[idx: Int]():
        print(idx)

    unroll[func, 4]()


# CHECK-LABEL: test_unroll
fn test_unroll2():
    print("test_unroll")

    # CHECK: (0, 0)
    # CHECK: (0, 1)
    # CHECK: (1, 0)
    # CHECK: (1, 1)
    @parameter
    fn func[idx0: Int, idx1: Int]():
        print(StaticIntTuple[2](idx0, idx1))

    unroll[func, 2, 2]()


# CHECK-LABEL: test_unroll
fn test_unroll3():
    print("test_unroll")

    # CHECK: (0, 0, 0)
    # CHECK: (0, 0, 1)
    # CHECK: (0, 0, 2)
    # CHECK: (0, 1, 0)
    # CHECK: (0, 1, 1)
    # CHECK: (0, 1, 2)
    # CHECK: (1, 0, 0)
    # CHECK: (1, 0, 1)
    # CHECK: (1, 0, 2)
    # CHECK: (1, 1, 0)
    # CHECK: (1, 1, 1)
    # CHECK: (1, 1, 2)
    # CHECK: (2, 0, 0)
    # CHECK: (2, 0, 1)
    # CHECK: (2, 0, 2)
    # CHECK: (2, 1, 0)
    # CHECK: (2, 1, 1)
    # CHECK: (2, 1, 2)
    # CHECK: (3, 0, 0)
    # CHECK: (3, 0, 1)
    # CHECK: (3, 0, 2)
    # CHECK: (3, 1, 0)
    # CHECK: (3, 1, 1)
    # CHECK: (3, 1, 2)
    @parameter
    fn func[idx0: Int, idx1: Int, idx2: Int]():
        print(StaticIntTuple[3](idx0, idx1, idx2))

    unroll[func, 4, 2, 3]()


# CHECK-LABEL: test_unroll_raises
fn test_unroll_raises() raises:
    print("test_unroll_raises")

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    @parameter
    fn func[idx: Int]() raises:
        print(idx)

    unroll[func, 4]()

    # CHECK: 0
    @parameter
    fn func2[idx: Int]() raises:
        print(idx)
        raise "Exception"

    try:
        unroll[func2, 4]()
    except e:
        # CHECK: raised Exception
        print("raised " + str(e))


fn main() raises:
    test_unroll()
    test_unroll2()
    test_unroll3()
    test_unroll_raises()
