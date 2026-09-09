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

#include "Support/MArchTarget/Host.h"
#include "Support/CPUCache.h"
#include "Support/DeviceSpecs.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/Threading/HWInfo.h"
#include "Support/Threading/ThreadAffinity.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include <mach/mach_init.h>
#include <mach/task.h>
#include <sys/sysctl.h>
#endif // __APPLE__

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif // _MSC_VER

#define DEBUG_TYPE "host"

using namespace M;

//===----------------------------------------------------------------------===//
// CPU Model Info
//===----------------------------------------------------------------------===//

static ErrorOr<std::vector<std::string>> getAllHostCPUModelNames() {
#if defined(__APPLE__)
  size_t len = 0;
  if (sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0) ==
          -1 &&
      errno != ENOMEM)
    return Error("Unable to query the machdep.cpu.brand_string for length: " +
                 llvm::Twine(strerror(errno)));
  SmallString<128> result;
  result.resize(len);
  if (sysctlbyname("machdep.cpu.brand_string", result.data(), &len, nullptr, 0))
    return Error("Unable to query the machdep.cpu.brand_string for value: " +
                 llvm::Twine(strerror(errno)));
  result.resize(len);
  return std::vector<std::string>{std::string(result)};
#elif defined(__linux__)
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/cpuinfo");
  if (std::error_code ec = errOrBuf.getError())
    return Error("Can't open /proc/cpuinfo: " + llvm::Twine(ec.message()));
  SmallVector<StringRef> lines;
  (*errOrBuf)->getBuffer().split(lines, "\n", /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  std::vector<std::string> modelNames;
  for (StringRef line : lines) {
    auto fields = line.split(':');
    if (fields.first.trim() == "model name")
      modelNames.push_back(fields.second.trim().str());
  }
  return modelNames;
#elif defined(_MSC_VER)
  // TODO: Implement for Windows.  This is very involved -- see
  // https://github.com/Thomas-Sparber/wmi for a complete example.  (We'll need
  // to fetch Win32_Processor.Name.)  For now, return an empty vector instead
  // of returning an error.  If we return error, system-info.exe will fail even
  // if we don't care about model name.
  return std::vector<std::string>{};
#else
  return Error("Unsupported platform.");
#endif
}

ErrorOr<std::string> M::getHostCPUModelName() {
  auto allModelNamesOr = getAllHostCPUModelNames();
  if (allModelNamesOr.isError())
    return allModelNamesOr.takeError();
  auto allModelNames = std::move(*allModelNamesOr);
  std::sort(allModelNames.begin(), allModelNames.end());
  allModelNames.erase(std::unique(allModelNames.begin(), allModelNames.end()),
                      allModelNames.end());
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::interleave(allModelNames, os, ", ");
  return str;
}

//===----------------------------------------------------------------------===//
// HostMachineInfo
//===----------------------------------------------------------------------===//

static void dumpFeatures(raw_ostream &os,
                         const std::vector<std::string> &features) {
  llvm::interleaveComma(features, os);
}

static void
dumpAffinities(raw_ostream &os,
               const std::optional<std::vector<size_t>> &affinities) {
  if (affinities) {
    os << "[";
    llvm::interleave(*affinities, os, ", ");
    os << "]";
  } else {
    os << "none";
  }
}

void M::HostMachineInfo::print(llvm::raw_ostream &os) const {
  os << "target-triple: ";
  os << triple;
  os << "\nos: ";
  os << osName;
  os << "\narch: ";
  os << cpuArch;
  os << "\ncpu-model: ";
  os << cpuModelName;
  os << "\nsimd-bitwidth: ";
  os << simdBitWidth;
  os << "\nfeatures: ";
  dumpFeatures(os, cpuFeatures);
  os << "\ncore-count: ";
  os << numPhysicalCores;
  os << "\nl1-cache-size: ";
  os << l1CacheSize;
  os << "\nl2-cache-size: ";
  os << l2CacheSize;
  os << "\nl3-cache-size: ";
  os << l3CacheSize;
  os << "\nl4-cache-size: ";
  os << l4CacheSize;
  os << "\naffinities: ";
  dumpAffinities(os, affinities);
  os << "\n";
}

void M::HostMachineInfo::print(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("target-triple", triple);
  json.attribute("os", osName);
  json.attribute("arch", cpuArch);
  json.attribute("cpu-model", cpuModelName);
  json.attribute("simd-bitwidth", simdBitWidth);
  json.attribute("features", cpuFeatures);
  json.attribute("core-count", numPhysicalCores);
  json.attribute("l1-cache-size", l1CacheSize);
  json.attribute("l2-cache-size", l2CacheSize);
  json.attribute("l3-cache-size", l3CacheSize);
  json.attribute("l4-cache-size", l4CacheSize);
  if (affinities) {
    json.attribute("affinities", *affinities);
  }
  json.objectEnd();
}

void HostMachineInfo::print(HostProperty property,
                            llvm::json::OStream &json) const {
  switch (property) {
  case HostProperty::TargetTriple:
    json.attribute("target-triple", triple);
    break;
  case HostProperty::OS:
    json.attribute("os", osName);
    break;
  case HostProperty::Arch:
    json.attribute("arch", cpuArch);
    break;
  case HostProperty::CPUModel:
    json.attribute("cpu-model", cpuModelName);
    break;
  case HostProperty::SIMDBitWidth:
    json.attribute("simd-bitwidth", simdBitWidth);
    break;
  case HostProperty::Features:
    json.attribute("features", cpuFeatures);
    break;
  case HostProperty::CoreCount:
    json.attribute("core-count", numPhysicalCores);
    break;
  case HostProperty::L1CacheSize:
    json.attribute("l1-cache-size", l1CacheSize);
    break;
  case HostProperty::L2CacheSize:
    json.attribute("l2-cache-size", l2CacheSize);
    break;
  case HostProperty::L3CacheSize:
    json.attribute("l3-cache-size", l3CacheSize);
    break;
  case HostProperty::L4CacheSize:
    json.attribute("l4-cache-size", l4CacheSize);
    break;
  case HostProperty::Affinities:
    if (affinities)
      json.attribute("affinities", *affinities);
    break;
  }
}

void HostMachineInfo::print(HostProperty property,
                            llvm::raw_ostream &os) const {
  switch (property) {
  case HostProperty::TargetTriple:
    os << triple;
    break;
  case HostProperty::OS:
    os << osName;
    break;
  case HostProperty::Arch:
    os << cpuArch;
    break;
  case HostProperty::CPUModel:
    os << cpuModelName;
    break;
  case HostProperty::SIMDBitWidth:
    os << simdBitWidth;
    break;
  case HostProperty::Features:
    dumpFeatures(os, cpuFeatures);
    break;
  case HostProperty::CoreCount:
    os << numPhysicalCores;
    break;
  case HostProperty::L1CacheSize:
    os << l1CacheSize;
    break;
  case HostProperty::L2CacheSize:
    os << l2CacheSize;
    break;
  case HostProperty::L3CacheSize:
    os << l3CacheSize;
    break;
  case HostProperty::L4CacheSize:
    os << l4CacheSize;
    break;
  case HostProperty::Affinities:
    dumpAffinities(os, affinities);
    break;
  }
  os << "\n";
}

void HostMachineInfo::printStaticInfo(raw_ostream &os) const {
  print(HostProperty::TargetTriple, os);
  print(HostProperty::OS, os);
  print(HostProperty::Arch, os);
  print(HostProperty::CPUModel, os);
  print(HostProperty::SIMDBitWidth, os);
  print(HostProperty::Features, os);
  print(HostProperty::L1CacheSize, os);
  print(HostProperty::L2CacheSize, os);
  print(HostProperty::L3CacheSize, os);
  print(HostProperty::L4CacheSize, os);
}

HostMachineInfo HostMachineInfo::fromTargetInfo(const TargetInfo &targetInfo) {
  HostMachineInfo result;
  result.triple = targetInfo.triple.str();
  result.osName = llvm::Triple::getOSTypeName(targetInfo.triple.getOS());
  result.cpuArch = targetInfo.arch;
  result.cpuFeatures = targetInfo.features;
  return result;
}

static M::ErrorOr<HostMachineInfo> getHostMachineInfoImpl() {
  HostMachineInfo machineInfo;

  machineInfo.triple = llvm::sys::getDefaultTargetTriple();
  machineInfo.osName =
      llvm::Triple::getOSTypeName(llvm::Triple(machineInfo.triple).getOS());
  machineInfo.cpuArch = llvm::sys::getHostCPUName();
  auto cpuModelNameOr = getHostCPUModelName();
  if (cpuModelNameOr.isError())
    return cpuModelNameOr.takeError();
  machineInfo.cpuModelName = std::move(*cpuModelNameOr);

  // Get OS version.  Note some barebones setups might not set this.
  auto osVersion = getHostOSVersion();
  if (!osVersion.isError())
    machineInfo.osVersion = osVersion.takeValue();

  auto hostTargetInfoOr = getHostTargetInfo();
  if (hostTargetInfoOr.isError())
    return hostTargetInfoOr.takeError();
  machineInfo.cpuFeatures = hostTargetInfoOr->features;

  machineInfo.simdBitWidth = simdWidthFromFeatures(machineInfo.cpuFeatures);

  machineInfo.numPhysicalCores = M::getNumPhysicalCores();

  auto l1CacheSizeOr = getHostCPUCacheSize(1);
  if (l1CacheSizeOr.isError())
    return l1CacheSizeOr.takeError();
  machineInfo.l1CacheSize = std::move(*l1CacheSizeOr);

  auto l2CacheSizeOr = getHostCPUCacheSize(2);
  if (l2CacheSizeOr.isError())
    return l2CacheSizeOr.takeError();
  machineInfo.l2CacheSize = std::move(*l2CacheSizeOr);

  auto l3CacheSizeOr = getHostCPUCacheSize(3);
  if (l3CacheSizeOr.isError())
    return l3CacheSizeOr.takeError();
  machineInfo.l3CacheSize = std::move(*l3CacheSizeOr);

  auto l4CacheSizeOr = getHostCPUCacheSize(4);
  if (l4CacheSizeOr.isError())
    return l4CacheSizeOr.takeError();
  machineInfo.l4CacheSize = std::move(*l4CacheSizeOr);

  if (haveThreadAffinity()) {
    ErrorOr<CPUSystemInfo> errOrSysInfo = CPUSystemInfo::get();
    if (!errOrSysInfo.isError()) {
      machineInfo.affinities =
          errOrSysInfo->getPreferredCpuIDs(machineInfo.numPhysicalCores);
    }
    // else: ignore error, leave field empty to denote affinities are not avail.
  }

  return std::move(machineInfo);
}

M::ErrorOr<HostMachineInfo> M::getHostMachineInfo() {
  // We cache the host machine information to make query a lot faster, since it
  // will not change between invocations.
  static M::ErrorOr<HostMachineInfo> hostMachineInfo = getHostMachineInfoImpl();
  if (hostMachineInfo.isError())
    return M::Error(hostMachineInfo.getError());
  return *hostMachineInfo;
}

M::ErrorOr<bool> M::getHostIsInContainer() {
#if defined(__linux__)
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBufCgroup =
      llvm::MemoryBuffer::getFileAsStream("/proc/self/cgroup");
  if (std::error_code ec = errOrBufCgroup.getError())
    return Error("Can't open /proc/self/cgroup: " + llvm::Twine(ec.message()));
  SmallVector<StringRef> lines;
  (*errOrBufCgroup)
      ->getBuffer()
      .split(lines, "\n", /*MaxSplit=*/-1,
             /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    auto fields = line.split('/');
    if (!fields.second.empty())
      return true;
  }

  // The above may not work with cgroup v2, so we also try an alternative
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBufMountinfo =
      llvm::MemoryBuffer::getFileAsStream("/proc/self/mountinfo");
  if (std::error_code ec = errOrBufMountinfo.getError())
    return Error("Can't open /proc/self/mountinfo: " +
                 llvm::Twine(ec.message()));
  (*errOrBufMountinfo)
      ->getBuffer()
      .split(lines, "\n", /*MaxSplit=*/-1,
             /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    if (line.contains("/docker/containers/"))
      return true;
  }

  return false;
#else
  return false;
#endif
}

M::ErrorOr<std::string> M::getHostTotalMemoryKB() {
#if defined(__APPLE__)
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  int64_t bytes;
  size_t len = sizeof(int64_t);
  if (sysctl(mib, 2, &bytes, &len, nullptr, 0) == -1)
    return Error("Unable to query hw.memsize: " + llvm::Twine(strerror(errno)));
  return std::to_string(bytes / 1000);
#elif defined(__linux__)
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/meminfo");
  if (std::error_code ec = errOrBuf.getError())
    return Error("Can't open /proc/meminfo: " + llvm::Twine(ec.message()));
  SmallVector<StringRef> lines;
  (*errOrBuf)->getBuffer().split(lines, "\n", /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    auto fields = line.split(':');
    if (fields.first.trim() == "MemTotal")
      return fields.second.trim().drop_back(3).str();
  }
  return Error("Failed to find MemTotal field in /proc/meminfo.");
#elif defined(_MSC_VER)
  // TODO Windows implementation
  return "";
#else
  return Error("Unsupported platform.");
#endif
}

M::ErrorOr<std::string> M::getHostOSVersion() {
#if defined(__APPLE__)
  std::string osVersion;
  auto procTriple = llvm::Triple(llvm::sys::getProcessTriple());
  llvm::VersionTuple adjustedVersion;
  if (!procTriple.getMacOSXVersion(adjustedVersion))
    return Error("Failed to getMacOSXVersion");
  osVersion = adjustedVersion.getAsString();
  // Deal with LLVM sometimes missing minor version
  if (!adjustedVersion.getMinor().has_value() &&
      procTriple.getOSVersion().getMinor().has_value()) {
    llvm::VersionTuple fullVersion(
        adjustedVersion.getMajor(),
        procTriple.getOSVersion().getMinor().value());
    osVersion = fullVersion.getAsString();
  }

  return osVersion;
#elif defined(__linux__)
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/etc/os-release");
  if (std::error_code ec = errOrBuf.getError())
    return Error("Can't open /etc/os-release: " + llvm::Twine(ec.message()));
  SmallVector<StringRef> lines;
  (*errOrBuf)->getBuffer().split(lines, "\n", /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    auto fields = line.split('=');
    if (fields.first.trim() == "PRETTY_NAME")
      return fields.second.trim().drop_back(1).drop_front(1).str();
  }
  return Error("Failed to find PRETTY_NAME field in /etc/os-release.");
#elif defined(_MSC_VER)
  // TODO Windows implementation
  return "";
#else
  return Error("Unsupported platform.");
#endif
}
