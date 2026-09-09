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

#ifndef ASYNCRT_SUPPORT_MLIRLOCATIONDECODER_H
#define ASYNCRT_SUPPORT_MLIRLOCATIONDECODER_H

#include "AsyncRT/Support/Diagnostic.h"
#include "AsyncRT/Support/Location.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ReferenceCounted.h"
#include <string>

namespace M::AsyncRT {

/// This class implements LocationDecoder and reports the MLIR location as
/// file/line/column when possible, otherwise just reports the printed location.
class MLIRLocationDecoder final : public ReferenceCounted<MLIRLocationDecoder>,
                                  public LocationDecoder {
public:
  MLIRLocationDecoder() = default;

  static EncodedLocation getEncodedLocation(mlir::Location loc);

  /// Implement the LocationDecoder hooks - the EncodedLocation contains a
  /// pointer that can be decoded with the context into a full mlir::Location.
  DecodedLocation decode(const EncodedLocation &loc) const override;
  void addRef() const override;
  void dropRef() const override;
};

/// Given an Error and an mlir::Location, we can create an EncodedDiagnostic.
EncodedDiagnostic getMLIRDiagnostic(Error e, mlir::Location loc);

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_MLIRLOCATIONDECODER_H
