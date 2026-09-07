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


# start-unsafe-from-address
def write_to_address(mmio_address: Int, value: Int32):
    var ptr = Pointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=mmio_address
    )

    # Writing to a raw memory address may require a volatile load/store as the
    # operation may have side effects not visible to the compiler.
    # You can specify this using the `volatile` parameter.
    ptr.unsafe_store[volatile=True](value)


# end-unsafe-from-address


def main():
    # Test opaque pointer creation via bitcast.
    # start-opaque-pointer
    var str = "Hello, world!"
    var str_ptr = Pointer(to=str)
    var opaque_ptr = str_ptr.unsafe_bitcast[NoneType]()
    # ... call some foreign function that takes a void pointer
    # end-opaque-pointer

    _ = opaque_ptr
