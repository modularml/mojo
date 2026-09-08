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

#include "Target/Host/HostTraits.h"

#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

const HostTraits &HostTraits::get() {
  static const HostTraits instance;
  return instance;
}

llvm::StringRef HostTraits::defaultCPU(const llvm::Triple &triple) const {
  // 32-bit ARM (arm/armeb) has no host-CPU default; pin it to the arch's
  // baseline CPU.
  if (triple.isARM())
    return llvm::ARM::getDefaultCPU(triple.getArchName());
  return {};
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<HostTraits> registerHostTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
