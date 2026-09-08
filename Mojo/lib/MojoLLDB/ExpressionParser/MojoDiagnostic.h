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

#ifndef KGEN_LIB_MOJOLLDB_MOJODIAGNOSTIC_H
#define KGEN_LIB_MOJOLLDB_MOJODIAGNOSTIC_H

#include "Support/LLVMForwardDecls.h"
#include "lldb/Expression/DiagnosticManager.h"
#include <cstdint>

namespace M::KGEN::Mojo {
/// This class defines a custom diagnostic type used for diagnostics originating
/// from Mojo.
class MojoDiagnostic : public lldb_private::Diagnostic {
public:
  MojoDiagnostic(StringRef message, lldb::Severity severity,
                 bool hadFixitsAttached)
      : Diagnostic(lldb_private::eDiagnosticOriginLLVM, kMojoCompilerID,
                   lldb_private::DiagnosticDetail{
                       {}, severity, message.str(), message.str()}),
        hadFixitsAttached(hadFixitsAttached) {}

  static bool classof(const Diagnostic *diag) {
    return diag->getKind() == lldb_private::eDiagnosticOriginLLVM &&
           diag->GetCompilerID() == kMojoCompilerID;
  }

  /// Returns true if this diagnostic had fixits attached.
  bool hadFixits() const { return hadFixitsAttached; }

private:
  /// This is a random, likely unused number that we can use to identify a Mojo
  /// diagnostic.
  static constexpr uint32_t kMojoCompilerID = UINT32_MAX - 12;

  /// Flag indicating if this diagnostic had fixits attached.
  bool hadFixitsAttached = false;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_MOJODIAGNOSTIC_H
