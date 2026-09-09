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

#include "Support/BuildInfo.h"
#include "Config/Version.h"
#include "Support/AlignedAlloc.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

void BuildInfo::print(llvm::raw_ostream &os) const {
  os << "modular-version: " << modularVersion;
  os << "\ngit-revision: " << gitRevision;
  os << "\nbuild-type: " << buildType;
  os << "\naysncrt-max-profiling-level: "
     << llvm::format("0%04o", asyncrtMaxProfilingLevel);
  os << "\npreferred-mem-alignment: " << preferredMemoryAlignment;
  os << "\n";
}

void BuildInfo::print(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("modular-version", modularVersion);
  json.attribute("git-revision", gitRevision);
  json.attribute("build-type", buildType);
  json.attribute("asyncrt-max-profiling-level", asyncrtMaxProfilingLevel);
  json.attribute("preferred-mem-alignment", preferredMemoryAlignment);
  json.objectEnd();
}

void BuildInfo::print(BuildProperty property, llvm::raw_ostream &os) const {
  switch (property) {
  case BuildProperty::ModularVersion:
    os << modularVersion;
    break;
  case BuildProperty::GitRevision:
    os << gitRevision;
    break;
  case BuildProperty::BuildType:
    os << buildType;
    break;
  case BuildProperty::AsyncRTMaxProfilingLevel:
    os << asyncrtMaxProfilingLevel;
    break;
  case BuildProperty::PreferredMemoryAlignment:
    os << preferredMemoryAlignment;
    break;
  }
  os << "\n";
}

BuildInfo M::getBuildInfo() {
  BuildInfo buildInfo;

  ModularVersion modularVersion = getModularVersion();
  buildInfo.modularVersion = getModularVersionString();
  buildInfo.gitRevision = modularVersion.revision;
  buildInfo.buildType = modularVersion.buildType;
  buildInfo.asyncrtMaxProfilingLevel = MODULAR_ASYNCRT_MAX_PROFILING_LEVEL;
  buildInfo.preferredMemoryAlignment = kPreferredMemoryAlignment;

  return buildInfo;
}
