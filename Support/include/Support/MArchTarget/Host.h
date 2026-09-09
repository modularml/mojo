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
// Utilities for interrogating and interacting with the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_HOST_H
#define SUPPORT_MARCHTARGET_HOST_H

#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/JSON.h"

#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
// Forward declare.
class MemoryBuffer;
} // namespace llvm

namespace llvm::json {
// Forward declare.
class OStream;
} // namespace llvm::json

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#define HOST_IS_APPLE_SILICON_PROCESSOR
#endif

namespace M {
//===----------------------------------------------------------------------===//
// CPU Model Info
//===----------------------------------------------------------------------===//

/// Get the actual model name of the host's CPU, e.g. "Intel(R) Xeon(R)
/// Platinum 8275CL CPU @ 3.00GHz".
ErrorOr<std::string> getHostCPUModelName();

//===----------------------------------------------------------------------===//
// HostMachineInfo
//===----------------------------------------------------------------------===//

enum class HostProperty {
  TargetTriple,
  OS,
  Arch,
  CPUModel,
  Features,
  SIMDBitWidth,
  CoreCount,
  L1CacheSize,
  L2CacheSize,
  L3CacheSize,
  L4CacheSize,
  Affinities
};

/// Information of host machine.
struct HostMachineInfo {
  std::string triple;
  std::string osName;
  std::string cpuArch;
  std::string cpuModelName;
  std::string osVersion;
  // This is the SIMD bit-width of the host system.
  size_t simdBitWidth = 0;
  std::vector<std::string> cpuFeatures;
  size_t numPhysicalCores = 0;
  // These represent either data or unified cache size -- they do not include
  // instruction-only caches, but may include cache size that are shared
  // between instruction and data.
  size_t l1CacheSize = 0;
  size_t l2CacheSize = 0;
  size_t l3CacheSize = 0;
  size_t l4CacheSize = 0;
  // Preferred CPU ids for numPhysicalCores threads if both CPUSystemInfo
  // and thread affinities are supported. Otherwise empty.
  std::optional<std::vector<size_t>> affinities;

  void print(llvm::raw_ostream &os) const;
  void print(llvm::json::OStream &json) const;
  void print(HostProperty property, llvm::raw_ostream &os) const;
  void print(HostProperty property, llvm::json::OStream &json) const;

  /// Print information excluding the ones that are likely to change with
  /// threading configuration, such as number of cores and affinities.
  void printStaticInfo(llvm::raw_ostream &os) const;

  /// Returns a HostMachineInfo matching target info. Only some fields
  /// are captured:
  ///  - triple (captured as triple and osName)
  ///  - cpu (captured as cpuArch)
  ///  - features (captured as cpuFeatures)
  ///
  /// CAUTION: Temporary while we unravel the TargetInfoAttr/HostMachineInfo
  ///          confusion.
  HostMachineInfo static fromTargetInfo(const TargetInfo &targetInfo);
};

/// Get information about the host machine.
///
/// CAUTION: Use getHostTargetInfo from MArchTarget if you only need a
/// triple, cpu and features.
ErrorOr<HostMachineInfo> getHostMachineInfo();

/// Get a flag indicating whether the host machine is running in a container
ErrorOr<bool> getHostIsInContainer();

/// Get the host machine total memory in kB
ErrorOr<std::string> getHostTotalMemoryKB();

/// Get the host machine OS version
ErrorOr<std::string> getHostOSVersion();

} // namespace M

#endif // SUPPORT_MARCHTARGET_HOST_H
