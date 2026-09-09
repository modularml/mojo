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

#include "Config/Version.h"
#include "GeneratedVersion.h"

using namespace M;

ProjectVersion M::getMAXVersion() {
  return ProjectVersion{
      .major = MAX_VERSION_MAJOR,
      .minor = MAX_VERSION_MINOR,
      .patch = MAX_VERSION_PATCH,
      .label = MAX_VERSION_LABEL,
      .revision = MODULAR_VERSION_REVISION,
      .buildType = MODULAR_BUILD_TYPE_LOWER,
  };
}

const char *M::getMAXVersionString() { return MAX_VERSION_STRING; }

ProjectVersion M::getMojoVersion() {
  return ProjectVersion{
      .major = MOJO_VERSION_MAJOR,
      .minor = MOJO_VERSION_MINOR,
      .patch = MOJO_VERSION_PATCH,
      .label = MOJO_VERSION_LABEL,
      .revision = MODULAR_VERSION_REVISION,
      .buildType = MODULAR_BUILD_TYPE_LOWER,
  };
}

const char *M::getMojoVersionString() { return MOJO_VERSION_STRING; }

// TODO: Remove

M::ModularVersion M::getModularVersion() { return M::getMAXVersion(); }

const char *M::getModularVersionString() { return MODULAR_VERSION_STRING; }
