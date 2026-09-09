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
// Utilities for interrogating build-time settings.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BUILDINFO_H
#define SUPPORT_BUILDINFO_H

#include "Support/LLVMForwardDecls.h"

#include <string>

namespace llvm::json {
class OStream;
}

namespace M {

enum class BuildProperty {
  ModularVersion,
  GitRevision,
  BuildType,
  AsyncRTMaxProfilingLevel,
  PreferredMemoryAlignment,
};

struct BuildInfo {
  std::string modularVersion;
  std::string gitRevision;
  std::string buildType;
  int asyncrtMaxProfilingLevel;
  size_t preferredMemoryAlignment;

  void print(llvm::raw_ostream &os) const;
  void print(llvm::json::OStream &json) const;
  void print(BuildProperty property, llvm::raw_ostream &os) const;
};

/// Get information about this build of Modular.
BuildInfo getBuildInfo();

} // namespace M

#endif // SUPPORT_BUILDINFO_H
