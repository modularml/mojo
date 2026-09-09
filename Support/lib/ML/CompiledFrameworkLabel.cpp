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

#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

const char *CompiledFrameworkLabel::getAsOpNameOrNull() const {
  switch (value) {
  case kUnknown:
    return nullptr;
  case kModularModel:
    return "mgp.model";
  }
  llvm::report_fatal_error("missing case");
}

const char *CompiledFrameworkLabel::getAsFrameworkNameOrNull() const {
  return asLabelString(value);
}

bool CompiledFrameworkLabel::isValidOpName(StringRef opName) {
  return opName == "mgp.model";
}

bool CompiledFrameworkLabel::isValidFrameworkName(StringRef frameworkName) {
  return llvm::is_contained(
      {"tf",
       "mgp", // TODO(#6190): "mgp" isn't really a framework, replace with faux.
       "mof"},
      frameworkName);
}

CompiledFrameworkLabel
CompiledFrameworkLabel::getLabelForOpName(StringRef opName,
                                          StringRef frameworkName) {
  if (opName == "mef.model") {
    if (frameworkName == "mof")
      return CompiledFrameworkLabel{kModularModel};
  }
  llvm::errs() << opName << " & " << frameworkName << "\n";
  return CompiledFrameworkLabel{kUnknown};
}

const char *CompiledFrameworkLabel::getAsString() const {
  switch (value) {
  case kUnknown:
    return "unknown";
  case kModularModel:
    return "compiled Modular model";
  }
  llvm::report_fatal_error("missing case");
}

const char *
CompiledFrameworkLabel::asLabelString(CompiledFrameworkLabel::Cases label) {
  switch (label) {
  case kUnknown:
    return nullptr;
  case kModularModel:
    return "mof";
  }
  llvm::report_fatal_error("missing case");
}
