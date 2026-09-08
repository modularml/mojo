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
# RUN: %mojo-build-asan -O0 %s -o %t
# RUN: not %t 1 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %t 2 2>&1 | FileCheck --check-prefix CHECK_2 %s
# RUN: not %t 3 2>&1 | FileCheck --check-prefix CHECK_3 %s
# RUN: not %t 4 2>&1 | FileCheck --check-prefix CHECK_4 %s
# RUN: not %t 5 2>&1 | FileCheck --check-prefix CHECK_5 %s
# RUN: not %t 6 2>&1 | FileCheck --check-prefix CHECK_6 %s
# RUN: not %t 7 2>&1 | FileCheck --check-prefix CHECK_7 %s
# RUN: not %t 8 2>&1 | FileCheck --check-prefix CHECK_8 %s
# RUN: not %t 9 2>&1 | FileCheck --check-prefix CHECK_9 %s
# RUN: not %t 10 2>&1 | FileCheck --check-prefix CHECK_10 %s
# RUN: not %t 11 2>&1 | FileCheck --check-prefix CHECK_11 %s
# RUN: not %t 12 2>&1 | FileCheck --check-prefix CHECK_12 %s

from std.sys.arg import argv
from std.testing import assert_equal


def main() raises:
    if len(argv()) <= 1:
        return
    var test = argv()[1]
    if test == "1":
        var l = List[Int](capacity=10)
        # CHECK_1: AddressSanitizer: container-overflow on address
        # CHECK_1: READ of size 8 at
        # CHECK_1: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[])
    elif test == "2":
        var l = List[Int]([1, 2, 3, 4, 5])
        l.reserve(30)
        assert_equal(l[0], 1)
        assert_equal(l[4], 5)
        # CHECK_2: AddressSanitizer: container-overflow on address
        # CHECK_2: READ of size 8 at
        # CHECK_2: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=29])
    elif test == "3":
        var l = List[Int]([1, 2, 3, 4, 5])
        # CHECK_3: AddressSanitizer: heap-buffer-overflow on address
        # CHECK_3: READ of size 8 at
        # CHECK_3: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=30])
    elif test == "4":
        var l = List[Int](unsafe_uninit_length=10)
        l.unsafe_ptr()[unsafe_offset=0] = 1
        l.unsafe_ptr()[unsafe_offset=9] = 1
        # note: above store and below store might be combined into something like stp, and then the
        # asan instramentation go around _that_ 16-byte write. This means we don't know what line
        # the error should point to. It may be the above or the below
        # CHECK_4: AddressSanitizer: heap-buffer-overflow on address
        # CHECK_4: WRITE of size {{.*}} at
        # CHECK_4: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}
        l.unsafe_ptr()[unsafe_offset=10] = 1
    elif test == "5":
        var l = List[Int](capacity=10)
        l.extend([1, 2, 3, 4])
        print(l.unsafe_ptr()[unsafe_offset=0])
        print(l.unsafe_ptr()[unsafe_offset=3])
        # CHECK_5: AddressSanitizer: container-overflow on address
        # CHECK_5: READ of size 8 at
        # CHECK_5: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=4])
    elif test == "6":
        var l = List[Int](length=1, fill=1)
        l.append(1)
        print(l.unsafe_ptr()[unsafe_offset=1])
        # CHECK_6: AddressSanitizer: heap-buffer-overflow on address
        # CHECK_6: READ of size 8 at
        # CHECK_6: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=2])
    elif test == "7":
        var l = List[Int]()
        l.extend([1, 2, 3, 4])
        print(l.unsafe_ptr()[unsafe_offset=0])
        print(l.unsafe_ptr()[unsafe_offset=3])
        # CHECK_7: AddressSanitizer: heap-buffer-overflow on address
        # CHECK_7: READ of size 8 at
        # CHECK_7: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=4])
    elif test == "8":
        var l = List[Int]([1, 2, 3, 4])
        _ = l.pop()
        print(l.unsafe_ptr()[unsafe_offset=2])
        # CHECK_8: AddressSanitizer: container-overflow on address
        # CHECK_8: READ of size 8 at
        # CHECK_8: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=3])
    elif test == "9":
        var l = List[Int64](capacity=10)
        l.extend(SIMD[.int64, 2](1, 2))
        print(l.unsafe_ptr()[unsafe_offset=0])
        print(l.unsafe_ptr()[unsafe_offset=1])
        # CHECK_9: AddressSanitizer: container-overflow on address
        # CHECK_9: READ of size 8 at
        # CHECK_9: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=2])
    elif test == "10":
        var l = List[Int]([1, 2, 3, 4])
        l.resize(2, 0)
        print(l.unsafe_ptr()[unsafe_offset=1])
        # CHECK_10: AddressSanitizer: container-overflow on address
        # CHECK_10: READ of size 8 at
        # CHECK_10: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=2])
    elif test == "11":
        var l = List[Int]([1, 2, 3, 4])
        l.resize(unsafe_uninit_length=10)
        print(l.unsafe_ptr()[unsafe_offset=1])
        l.unsafe_ptr()[unsafe_offset=9] = 0
        # CHECK_11: AddressSanitizer: heap-buffer-overflow on address
        # CHECK_11: READ of size 8 at
        # CHECK_11: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=10])
    elif test == "12":
        var l = List[Int]([1, 2, 3, 4])
        l.shrink(2)
        print(l.unsafe_ptr()[unsafe_offset=0])
        l.unsafe_ptr()[unsafe_offset=1] = 0
        # CHECK_12: AddressSanitizer: container-overflow on address
        # CHECK_12: READ of size 8 at
        # CHECK_12: #0 {{.*}} in test_asan_annotations_list::main() {{.*}}:[[#@LINE+1]]
        print(l.unsafe_ptr()[unsafe_offset=3])
