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

#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

using namespace M;
using namespace AsyncRT;

EncodedLocation MLIRLocationDecoder::getEncodedLocation(mlir::Location loc) {
  return {(intptr_t)loc.getAsOpaquePointer(),
          RCRef<MLIRLocationDecoder>::create()};
}

DecodedLocation MLIRLocationDecoder::decode(const EncodedLocation &loc) const {
  auto mlirLoc = mlir::Location::getFromOpaquePointer((void *)loc.getData());
  if (auto fileLineColLoc = dyn_cast<mlir::FileLineColLoc>(mlirLoc))
    return DecodedLocation{fileLineColLoc.getFilename().str(),
                           (int)fileLineColLoc.getLine(),
                           (int)fileLineColLoc.getColumn()};

  std::string locStr;
  llvm::raw_string_ostream stream(locStr);
  stream << mlirLoc;
  return DecodedLocation{stream.str()};
}

/// Implement the LocationDecoder hook for addRef.
void MLIRLocationDecoder::addRef() const {
  RCRef<ReferenceCounted<MLIRLocationDecoder>>::lowLevelAddRef(
      const_cast<MLIRLocationDecoder *>(this));
}

/// Implement the LocationDecoder hook for dropRef.
void MLIRLocationDecoder::dropRef() const {
  RCRef<ReferenceCounted<MLIRLocationDecoder>>::lowLevelDropRef(
      const_cast<MLIRLocationDecoder *>(this));
}

EncodedDiagnostic AsyncRT::getMLIRDiagnostic(Error e, mlir::Location loc) {
  return {std::move(e), MLIRLocationDecoder::getEncodedLocation(loc)};
}
