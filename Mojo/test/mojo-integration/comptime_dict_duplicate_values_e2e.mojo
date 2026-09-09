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
# see MSTDL-1901
# XFAIL: asan
# RUN: %mojo %s | FileCheck %s

from std.collections import Dict
from std.hashlib import default_comp_time_hasher

# COM: testing MemoryBlob refCount in the interpreter
# COM: where to create the alias of the Dict, we have
# COM: two exact same attributes to represent the two ["001"]s
# COM: as the Dict values, but the interpreter only creates
# COM: one MemoryBlob for this memory type allocation due to
# COM: mlir::Attribute uniquing. Use refCount to make sure we free
# COM: these blobs correctly if needed in the interpreter.
comptime COUNTRY_CODE_TO_REGION_CODE: Dict[
    Int, List[String], default_comp_time_hasher
] = {
    800: ["001"],
    808: ["001"],
}


def _get_country_codes() -> List[String]:
    var result = List[String]()
    var vals = materialize[COUNTRY_CODE_TO_REGION_CODE.values()]()
    for value in vals:
        result += value.copy()
    return result^


def main() raises:
    var COUNTRY_CODES = _get_country_codes()
    for v in COUNTRY_CODES:
        print(v)


# CHECK: 001
# CHECK: 001
