//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines common utilities for tablegen files defining and working with
// custom tensor types.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_TENSORUTILS_TD
#define SUPPORT_COMPILER_TENSORUTILS_TD

include "mlir/IR/AttrTypeBase.td"

def DimensionListParameter : ArrayRefParameter<"int64_t", "shape"> {
  let printer = [{
    for (int64_t dim : $_self) {
      if (ShapedType::isDynamic(dim))
        $_printer << '?';
      else
        $_printer << dim;
      $_printer << 'x';
    }
  }];
  let parser = [{
    [&]() -> FailureOr<llvm::SmallVector<int64_t>> {
        llvm::SmallVector<int64_t> dims;
        if ($_parser.parseDimensionList(dims, /*allowDynamic=*/true))
          return failure();

        return dims;
      }
    ()
  }];
}

#endif // SUPPORT_COMPILER_TENSORUTILS_TD
