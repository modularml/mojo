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

# This file is imported by 'import-debuginfo.mojo' and does not include any
# tests itself.


# Don't move things around in this file, or else location info will break.
def imported_fn():
    return


@fieldwise_init
struct VeryUniqueStruct(TrivialRegisterPassable):
    var very_unique_field: __mlir_type.index

    # C-3PO is a short and very unique argument name. We use it to make
    # FileCheck matching easier.
    @staticmethod
    def very_unique_func(`C-3PO`: __mlir_type.index) -> VeryUniqueStruct:
        return Self(`C-3PO`)
