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
# RUN: %mojo --debug-level full %s

from testing import assert_equal, assert_false, assert_true, assert_almost_equal

from complex import Quaternion, DualQuaternion


fn test_quaternion_ops() raises:
    var q1 = Quaternion(2, 3, 4, 5)
    var q2 = Quaternion(2, 3, 4, 5)
    var q3 = Quaternion(5, 4, 3, 2)
    assert_equal(7.348, q1.__abs__())
    assert_equal(Quaternion(4, 6, 8, 10), q1 + q2)
    assert_equal(Quaternion(0, 0, 0, 0), q1 - q2)
    assert_equal(DualQuaternion(-46, 12, 16, 20), q1 * q2)
    assert_equal(q1 * q2, q2 * q1)
    assert_equal(Quaternion(-24, 16, 40, 22), q1 * q3)
    assert_equal(Quaternion(-24, 30, 12, 36), q3 * q1)
    assert_equal(Quaternion(0.815, 0.259, 0, 0.519), q1 / q3)
    assert_equal(Quaternion(0.815, 0, 0.519, 0.259), q3 / q1)
    # assert_equal(..., q1**3)
    # assert_equal(..., q1.exp())
    # assert_equal(..., q1.ln())
    # assert_equal(..., q1.sqrt())
    # assert_equal(..., q1.phi())


fn test_quaternion_matrix() raises:
    pass


fn test_dualquaternion_ops() raises:
    var q1 = DualQuaternion(2, 3, 4, 5, 6, 7, 8, 9)
    var q2 = DualQuaternion(2, 3, 4, 5, 6, 7, 8, 9)
    var q3 = DualQuaternion(9, 8, 7, 6, 5, 4, 3, 2)
    assert_equal(DualQuaternion(4, 6, 8, 10, 12, 14, 16, 18), q1 + q2)
    assert_equal(DualQuaternion(0, 0, 0, 0, 0, 0, 0, 0), q1 - q2)
    assert_equal(DualQuaternion(-46, 12, 16, 20, -172, 64, 80, 96), q1 * q2)
    assert_equal(q1 * q2, q2 * q1)
    assert_equal(DualQuaternion(-64, 32, 72, 46, -136, 112, 184, 124), q1 * q3)
    assert_equal(DualQuaternion(-64, 54, 28, 68, -136, 156, 96, 168), q3 * q1)
    # assert_equal(..., q1**3)


fn test_dualquaternion_matrix() raises:
    pass


fn test_dualquaternion_screw() raises:
    pass


fn main() raises:
    test_quaternion_ops()
    test_quaternion_matrix()
    test_dualquaternion_ops()
    test_dualquaternion_matrix()
    test_dualquaternion_screw()
