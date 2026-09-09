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

#ifndef KGEN_TARGET_HOST_HOSTTRAITS_H
#define KGEN_TARGET_HOST_HOSTTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct HostTraits final : TargetTraits {
  llvm::StringRef name() const override { return "host"; }
  bool matches(const llvm::Triple &triple) const override {
    // Covers the CPU targets the shipped build carries an LLVM backend for
    // (see BACKENDS in bazel/public-patches/llvm_project.bzl)
    return triple.isX86() || triple.isAArch64() || triple.isARM() ||
           triple.isRISCV();
  }
  llvm::StringRef getAsmExtension() const override { return ".s"; }
  llvm::StringRef getLLVMExtension() const override { return ".ll"; }
  llvm::StringRef getObjectExtension() const override { return ".o"; }
  llvm::StringRef defaultCPU(const llvm::Triple &triple) const override;

  /// Shared stateless instance for the backend `traits()`.
  static const HostTraits &get();

protected:
  // bast target that is always registered
  bool isBaseTarget() const override { return true; }
};

} // namespace M::KGEN

#endif // KGEN_TARGET_HOST_HOSTTRAITS_H
