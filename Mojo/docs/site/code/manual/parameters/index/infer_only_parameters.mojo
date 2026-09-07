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


# start-dependent-type-basic
def dependent_type_basic[dtype: DType, value: Scalar[dtype]]():
    print("Value: ", value)
    print("Value is floating-point: ", dtype.is_floating_point())


# end-dependent-type-basic


# start-dependent-type-infer-only
def dependent_type[dtype: DType, //, value: Scalar[dtype]]():
    print("Value: ", value)
    print("Value is floating-point: ", dtype.is_floating_point())


# end-dependent-type-infer-only


def mutate_span(span: Span[mut=True, Byte, _]) raises:
    for i in range(0, len(span), 2):
        if i + 1 < len(span):
            span.swap_elements(i, i + 1)


def main() raises:
    # start-dependent-type-basic-call
    dependent_type_basic[DType.float64, Float64(2.2)]()
    # end-dependent-type-basic-call
    # start-dependent-type-infer-only-call
    dependent_type[Float64(2.2)]()
    # end-dependent-type-infer-only-call
    var s = String("Robinson Crusoe surfed the interwebs.")
    mutate_span(s.unsafe_as_bytes_mut())
    print(s)
