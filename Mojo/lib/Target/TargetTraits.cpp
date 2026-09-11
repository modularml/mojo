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

#include "Target/TargetTraits.h"

#include "Support/Configuration.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

ErrorOrSuccess requireMaxForAccelerator(bool isMaxOnly) {
  if (!isMaxOnly)
    return success();
  // MAX presence is a process-level fact; cache it to avoid a config/filesystem
  // probe on every registry lookup.
  static const bool maxInstalled = isMaxInstalled();
  if (!maxInstalled)
    return Error("please install MAX for accelerator support");
  return success();
}

void requireMaxForAcceleratorRequest(llvm::StringRef targetAccelerator) {
  if (targetAccelerator.empty())
    return;
  static const bool maxInstalled = isMaxInstalled();
  if (!maxInstalled) {
    llvm::report_fatal_error("please install MAX for accelerator support",
                             /*gen_crash_diag=*/false);
  }
}

llvm::ArrayRef<TargetTraits::EmissionKind> TargetTraits::commonEmissionKinds() {
  static const EmissionKind kinds[] = {
      {"exe", EmitAs::OBJECT, "an executable binary file (default)"},
      {"shared-lib", EmitAs::OBJECT, "a shared (dynamic) library"},
      {"object", EmitAs::OBJECT, "a single object file (EXPERIMENTAL)"},
      {"llvm", EmitAs::LLVM,
       "unoptimized LLVM IR; GPU targets also emit one IR sidecar "
       "file per kernel"},
      {"llvm-bitcode", EmitAs::LLVM_BITCODE,
       "bitcode of unoptimized LLVM IR (.bc)"},
      {"asm", EmitAs::ASM,
       "target assembly; GPU targets also emit one sidecar file per "
       "kernel"},
  };
  return kinds;
}

void printSupportedEmissionKinds(llvm::raw_ostream &os, unsigned indent) {
  // Deduplicate across targets: one row per (kind, description), listing the
  // targets that share it.
  llvm::SmallVector<const TargetTraits *> targets;
  for (const std::unique_ptr<TargetTraits> &traits :
       TargetTraitsRegistry::get().targets())
    if (!traits->supportedEmissionKinds().empty())
      targets.push_back(traits.get());
  llvm::sort(targets, [](const TargetTraits *lhs, const TargetTraits *rhs) {
    return lhs->name() < rhs->name();
  });

  struct Row {
    llvm::StringRef kind;
    llvm::StringRef description;
    // Alphabetical, from the name-sorted target iteration.
    llvm::SmallVector<llvm::StringRef, 4> targets;
  };
  llvm::SmallVector<Row> rows;
  for (const TargetTraits *traits : targets) {
    for (const TargetTraits::EmissionKind &kind :
         traits->supportedEmissionKinds()) {
      auto *row = llvm::find_if(rows, [&](const Row &existing) {
        return existing.kind == kind.kind &&
               existing.description == kind.description;
      });
      if (row == rows.end())
        rows.push_back({kind.kind, kind.description, {traits->name()}});
      else
        row->targets.push_back(traits->name());
    }
  }
  llvm::stable_sort(
      rows, [](const Row &lhs, const Row &rhs) { return lhs.kind < rhs.kind; });

  size_t width = 0;
  auto label = [](const Row &row) {
    return (row.kind + " (" + llvm::join(row.targets, ", ") + ")").str();
  };
  for (const Row &row : rows)
    width = std::max(width, label(row).size() + 2);
  for (const Row &row : rows)
    os << std::string(indent, ' ') << llvm::left_justify(label(row), width)
       << "- " << row.description << "\n";
}

bool TargetTraits::supportsEmissionKind(llvm::StringRef kind) const {
  return llvm::any_of(supportedEmissionKinds(),
                      [&](const EmissionKind &emissionKind) {
                        return emissionKind.kind == kind;
                      });
}

ErrorOrSuccess TargetTraits::validateEmissionKind(llvm::StringRef kind) const {
  if (supportsEmissionKind(kind))
    return success();
  llvm::SmallVector<llvm::StringRef> kinds;
  for (const EmissionKind &emissionKind : supportedEmissionKinds())
    kinds.push_back(emissionKind.kind);
  return Error("target '" + name() + "' does not support emission kind '" +
               kind + "'; supported kinds: " + llvm::join(kinds, ", "));
}

ErrorOr<EmitAs> TargetTraits::emitAsForKind(llvm::StringRef kind) const {
  for (const EmissionKind &emissionKind : supportedEmissionKinds())
    if (emissionKind.kind == kind)
      return emissionKind.emitAs;
  return Error(validateEmissionKind(kind).getError());
}

bool isKnownEmissionKind(llvm::StringRef kind) {
  for (const std::unique_ptr<TargetTraits> &traits :
       TargetTraitsRegistry::get().targets())
    for (const TargetTraits::EmissionKind &emissionKind :
         traits->supportedEmissionKinds())
      if (emissionKind.kind == kind)
        return true;
  return false;
}

const TargetTraits *traitsForAcceleratorArch(llvm::StringRef arch) {
  if (arch.empty())
    return nullptr;
  for (const std::unique_ptr<TargetTraits> &traits :
       TargetTraitsRegistry::get().targets())
    for (const TargetTraits::AcceleratorArch &accel :
         traits->supportedAcceleratorArchs())
      if (accel.arch == arch)
        return traits.get();
  return nullptr;
}

static llvm::ManagedStatic<TargetTraitsRegistry> theTraitsRegistry;

TargetTraitsRegistry &TargetTraitsRegistry::get() { return *theTraitsRegistry; }

void TargetTraitsRegistry::add(std::unique_ptr<TargetTraits> traits) {
  Targets.push_back(std::move(traits));
}

ErrorOr<const TargetTraits *>
TargetTraitsRegistry::lookup(const llvm::Triple &triple) const {
  // A traits object that resolves `triple` to one it owns takes precedence over
  // a direct self-match.
  const TargetTraits *result = [&]() -> const TargetTraits * {
    const TargetTraits *directMatch = nullptr;
    for (const std::unique_ptr<TargetTraits> &traits : Targets) {
      const TargetTraits *resolved = traits->resolve(triple);
      if (!resolved)
        continue;
      if (resolved != traits.get())
        return resolved;
      if (!directMatch)
        directMatch = resolved;
    }
    return directMatch;
  }();
  if (!result) {
    return Error("target '" + triple.str() +
                 "' is not supported by this build");
  }
  if (ErrorOrSuccess e = requireMaxForAccelerator(!result->isBaseTarget()))
    return Error(e.getError());
  return result;
}

} // namespace M::KGEN
