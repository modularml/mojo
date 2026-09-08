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
from std.sys.compile import SanitizeAddress
from std.ffi import external_call


@always_inline
def __sanitizer_annotate_contiguous_container(
    beg: OptionalPointer[NoneType, MutUntrackedOrigin],
    end: OptionalPointer[NoneType, MutUntrackedOrigin],
    old_mid: OptionalPointer[NoneType, MutUntrackedOrigin],
    new_mid: OptionalPointer[NoneType, MutUntrackedOrigin],
):
    # follows __annotate_contiguous_container from __debug_utils
    # https://github.com/llvm/llvm-project/blob/main/libcxx/include/__debug_utils/sanitizers.h
    comptime if SanitizeAddress:
        if not __is_run_in_comptime_interpreter and beg:
            external_call[
                "__sanitizer_annotate_contiguous_container", NoneType
            ](beg, end, old_mid, new_mid)
