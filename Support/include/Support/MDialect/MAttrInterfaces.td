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

#ifndef SUPPORT_MDIALECT_MATTRINTERFACES_TD
#define SUPPORT_MDIALECT_MATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// HasAlignedBytesInterface
//===----------------------------------------------------------------------===//

def HasAlignedBytesInterface : AttrInterface<"HasAlignedBytesInterface"> {
  let description = [{
    This interface allows an attribute to describe the size and
    alignment of its underlying ArrayRef<uint8_t> data as an !M.aligned_bytes.
  }];

  let cppNamespace = "::M";

  let methods = [
    InterfaceMethod<
      [{
        Returns the M::AlignedBytesType describing the size and alignment of an
        attribute.
      }],
      "AlignedBytesType",
      "getAlignedBytesType",
      (ins )>,
  ];
}

#endif  // SUPPORT_MDIALECT_MATTRINTERFACES_TD
