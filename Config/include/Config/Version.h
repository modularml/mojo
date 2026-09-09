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
// Information about the Modular build version.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_VERSION_H
#define CONFIG_VERSION_H

namespace M {

struct ProjectVersion final {
  int major;
  int minor;
  int patch;
  const char *label;    // version label like "-rc.1"
  const char *revision; // Truncated Git SHA
  const char *buildType;
};

ProjectVersion getMAXVersion();
ProjectVersion getMojoVersion();
const char *getMAXVersionString();
const char *getMojoVersionString();

// TODO: Remove
using ModularVersion = ProjectVersion;
ModularVersion getModularVersion();
const char *getModularVersionString();

} // namespace M

#endif // CONFIG_VERSION_H
