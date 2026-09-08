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
// MLIR-lowering policy for host (CPU) targets.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostTraits.h"
#include "Target/TargetLowering.h"

namespace M::KGEN {
namespace {

class HostLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &HostTraits::get(); }

  // bast target that is always registered
  bool isBaseTarget() const override { return true; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<HostLowering> registerHostLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
