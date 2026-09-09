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

#include "Support/Threading/HWInfo.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sched.h>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

// _WIN32 rather than _MSC_VER, so this also builds under the mingw cross
// toolchain. The x86 intrinsic headers have no aarch64 form, so they narrow to
// x86_64 instead of following the wider Windows check.
#ifdef _WIN32
#include "llvm/Support/WindowsError.h"
#if MODULAR_X86_64
#include <cpuid.h>
#include <intrin.h>
#endif // MODULAR_X86_64
#include <iphlpapi.h>
#include <windows.h>
#else
#include <ifaddrs.h>
#include <sys/socket.h>
#endif

#ifdef __APPLE__
#include <net/if_dl.h>
#include <sys/sysctl.h>
#endif

#include <filesystem>

#define DEBUG_TYPE "threading-hw-info"

using namespace M;

//===----------------------------------------------------------------------===//
// CPUSystemInfo
//===----------------------------------------------------------------------===//

namespace {
#if HAVE_LINUX_X86_SYSTEM_INFO
/// Function is very similar to what linux uses. We will mainly use
/// this to detect P and E cores on x86.
static inline void native_cpuid(unsigned int *eax, unsigned int *ebx,
                                unsigned int *ecx, unsigned int *edx) {
  /* ecx is often an input as well as an output. */
  asm volatile("cpuid"
               : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
               : "0"(*eax), "2"(*ecx));
}
std::unique_ptr<llvm::MemoryBuffer> fileBuffer(StringRef path) {
  auto errOrBuf = llvm::MemoryBuffer::getFileAsStream(path);
  if (std::error_code ec = errOrBuf.getError()) {
    LDBG() << "fileBuffer: Could not open " << path;
    return nullptr;
  }
  return std::move(errOrBuf.get());
}
#endif

} // namespace

#if HAVE_LINUX_X86_SYSTEM_INFO
ErrorOr<std::string>
Detail::parseV1CPUCgroupFile(const llvm::MemoryBuffer &buf) {
  std::string cgroup;
  SmallVector<StringRef> strs;
  buf.getBuffer().split(strs, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef line : strs) {
    SmallVector<StringRef> frags;
    line.split(frags, ":");
    StringRef sys = frags[1].trim();
    StringRef grp = frags[2].trim();
    // Cpuset settings are already reflected in the CPU affinity mask, but
    // further changes must be monitored to trigger CPUSystemInfo regeneration.
    if (sys == "cpu,cpuacct") {
      cgroup = grp;
      break;
    }
  }
  if (cgroup.empty())
    return Error("could not parse v1 CPU cgroup");
  return cgroup;
}

ErrorOr<Detail::linuxCPULimits>
Detail::parseV1CPULimits(const llvm::MemoryBuffer &quotaBuf,
                         const llvm::MemoryBuffer &periodBuf) {
  Detail::linuxCPULimits limits;
  if (quotaBuf.getBuffer().trim().getAsInteger(10, limits.quota_us))
    return Error("can't parse CPU quota as an int");
  if (periodBuf.getBuffer().trim().getAsInteger(10, limits.period_us))
    return Error("can't parse CPU period as an int");
  return limits;
}

ErrorOr<std::string>
Detail::parseV2CPUCgroupFile(const llvm::MemoryBuffer &buf,
                             const std::function<bool(StringRef)> &exists) {
  StringRef cgroup;
  SmallVector<StringRef> strs;
  buf.getBuffer().split(strs, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef line : strs) {
    if (line.consume_front("0::")) {
      cgroup = line;
      break;
    }
  }
  if (cgroup.empty())
    return Error("could not parse v2 CPU cgroup");

  StringRef curr = cgroup.trim();
  const size_t n = curr.count('/');
  // Check each level of the v2 filesystem to find the right file.
  for (size_t i = 0; i <= n; ++i) {
    const auto path = ("/sys/fs/cgroup/" + curr + "/cpu.max").str();
    if (!exists(path)) {
      auto pos = curr.rfind('/');
      if (pos != StringRef::npos) {
        curr = curr.slice(0, pos);
        continue;
      } else
        return Error("could not resolve CPU max file");
    }
    break;
  }
  return curr.str();
}

ErrorOr<Detail::linuxCPULimits>
Detail::parseV2CPULimits(const llvm::MemoryBuffer &maxBuf) {
  Detail::linuxCPULimits limits;
  SmallVector<StringRef> strs;
  maxBuf.getBuffer().split(strs, " ", /*MaxSplit=*/2, /*KeepEmpty=*/false);
  if (strs.empty())
    return Error("can't parse empty CPU max and period");
  if (strs[0] != "max" && strs[0].trim().getAsInteger(10, limits.quota_us))
    return Error("can't parse CPU max as an int");
  if (strs.size() == 2 && strs[1].trim().getAsInteger(10, limits.period_us))
    return Error("can't parse CPU period as an int");
  return limits;
}

/// Looks up various cgroup CPU limits for the current process.
Detail::linuxCPULimits Detail::getLinuxCPULimits() {
  // Detect cgroup version.
  auto errOrControllers =
      llvm::MemoryBuffer::getFileAsStream("/sys/fs/cgroup/cgroup.controllers");
  bool isV1 = true;
  if (errOrControllers.getError().value() == 0) {
    SmallVector<StringRef> strs;
    (*errOrControllers)
        ->getBuffer()
        .split(strs, " ", /*MaxSplit=*/2, /*KeepEmpty=*/false);
    for (const auto str : strs) {
      // When using the hybrid layout, the cpu controller might not be mounted
      // as v2; in which case we fallback to locating the v1 filesystem.
      if (str == "cpu") {
        isV1 = false;
        break;
      }
    }
  }

  // Read and parse /proc/self/cgroup
  auto cgroupBuf = fileBuffer("/proc/self/cgroup");
  if (!cgroupBuf)
    return {};
  Detail::linuxCPULimits limits;
  if (isV1) {
    const auto errOrCgroup = parseV1CPUCgroupFile(*cgroupBuf);
    if (errOrCgroup.isError()) {
      LDBG() << "getLinuxCPULimits: " << errOrCgroup.getError();
      return {};
    }
    const std::string &cgroup = *errOrCgroup;
    // Quota and period files found in membership-specific filesystems.
    const auto quotaBuf =
        fileBuffer("/sys/fs/cgroup/cpu/" + cgroup + "/cpu.cfs_quota_us");
    if (!quotaBuf)
      return {};
    const auto periodBuf =
        fileBuffer("/sys/fs/cgroup/cpu/" + cgroup + "/cpu.cfs_period_us");
    if (!periodBuf)
      return {};
    const auto errOrLimits = parseV1CPULimits(*quotaBuf, *periodBuf);
    if (errOrLimits.isError()) {
      LDBG() << "getLinuxCPULimits: " << errOrLimits.getError();
      return {};
    }
    limits = *errOrLimits;
  } else {
    const auto errOrCgroup =
        parseV2CPUCgroupFile(*cgroupBuf, [](StringRef path) {
          return std::filesystem::exists(path.str());
        });
    if (errOrCgroup.isError()) {
      LDBG() << "getLinuxCPULimits: " << errOrCgroup.getError();
      return {};
    }
    const std::string &cgroup = *errOrCgroup;
    // Lives at some level of the cgroup v2 unified filesystem as a combined
    // file.
    const auto maxBuf = fileBuffer("/sys/fs/cgroup/" + cgroup + "/cpu.max");
    if (!maxBuf)
      return {};
    const auto errOrLimits = parseV2CPULimits(*maxBuf);
    if (errOrLimits.isError()) {
      LDBG() << "getLinuxCPULimits: " << errOrLimits.getError();
      return {};
    }
    limits = *errOrLimits;
  }
  // The bounds and explanations for these values can be found at:
  // https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.rst
  if (limits.quota_us != -1 && limits.quota_us < 1000) {
    LDBG() << "getLinuxCPULimits: Expected cpu quota above 1ms";
    return {};
  }
  if (limits.period_us < 1000 || limits.period_us > 1000000) {
    LDBG() << "getLinuxCPULimits: Expected cpu period between 1ms and 1s";
    return {};
  }
  return limits;
}

/// Given contents of /proc/cpuinfo and the thread affinity mask describing
/// which CPUs should be considered available, returns CPUSystemInfo.
ErrorOr<CPUSystemInfo> M::Detail::getLinuxX86CPUSystemInfoImpl(
    const cpu_set_t &availableCpus, std::unique_ptr<llvm::MemoryBuffer> buf) {
  // Describes a virtual core.
  struct Entry {
    // Dense socket index.
    int physicalId = -1;
    // Non-dense index of physical core in socket.
    int coreId = -1;
    // Dense system-wide processor identifier, what we'll refer to as cpuID.
    int processor = -1;

    bool operator<(const Entry &that) const {
      return std::tie(physicalId, coreId, processor) <
             std::tie(that.physicalId, that.coreId, that.processor);
    }
  };

  SmallVector<StringRef> strs;
  buf->getBuffer().split(strs, "\n", /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);

  // Collect entries
  SmallVector<Entry> entries;
  Entry currEntry;
  for (StringRef line : strs) {
    std::pair<StringRef, StringRef> data = line.split(':');
    StringRef name = data.first.trim();
    StringRef val = data.second.trim();
    // These fields are available if the kernel is configured with CONFIG_SMP.
    if (name == "processor") {
      if (val.getAsInteger(10, currEntry.processor))
        return Error("ill-formed /proc/cpuinfo output");
    } else if (name == "physical id") {
      if (val.getAsInteger(10, currEntry.physicalId))
        return Error("ill-formed /proc/cpuinfo output");
    } else if (name == "core id") {
      if (val.getAsInteger(10, currEntry.coreId))
        return Error("ill-formed /proc/cpuinfo output");
      if (currEntry.physicalId < 0 || currEntry.coreId < 0 ||
          currEntry.processor < 0)
        return Error("ill-formed /proc/cpuinfo output");
      if (CPU_ISSET(currEntry.processor, &availableCpus)) {
        // Only include if processor already in affinity set.
        entries.push_back(currEntry);
        currEntry = Entry();
      } else {
        LDBG() << "getLinuxX86CPUSystemInfo: Ignoring processor "
               << std::to_string(currEntry.processor)
               << " since excluded from main thread's affinity set";
      }
    }
  }
  llvm::sort(entries);

  // Build system info.
  CPUSystemInfo systemInfo;
  Entry prevEntry;
  for (auto entry : entries) {
    if (entry.physicalId != prevEntry.physicalId)
      systemInfo.sockets.emplace_back();
    CPUSystemInfo::Socket &socket = systemInfo.sockets.back();
    if (entry.physicalId != prevEntry.physicalId ||
        entry.coreId != prevEntry.coreId)
      socket.physicalCores.emplace_back();
    CPUSystemInfo::PhysicalCore &physicalCore = socket.physicalCores.back();
    physicalCore.virtualCores.emplace_back(entry.processor);
    prevEntry = entry;
  }
  return systemInfo;
}

/// Helper to parse CPU list strings like "0-3,5,7-9" into a vector of CPU IDs.
static ErrorOr<std::vector<size_t>> parseCpuList(StringRef cpuList) {
  std::vector<size_t> cpuIds;
  SmallVector<StringRef> ranges;
  cpuList.trim().split(ranges, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  for (StringRef range : ranges) {
    if (range.contains('-')) {
      auto [startStr, endStr] = range.split('-');
      size_t start, end;
      if (startStr.getAsInteger(10, start) || endStr.getAsInteger(10, end))
        return Error("Failed to parse CPU range: " + range.str());
      for (size_t i = start; i <= end; ++i)
        cpuIds.push_back(i);
    } else {
      size_t cpuId;
      if (range.getAsInteger(10, cpuId))
        return Error("Failed to parse CPU ID: " + range.str());
      cpuIds.push_back(cpuId);
    }
  }

  return cpuIds;
}

/// On X86 Linux systems /proc/cpuinfo allows us to distinguish
/// virtual cores, physical cores and sockets.
///
/// Adapted from third-party/llvm-project/llvm/lib/Support/Unix/Threading.inc.
ErrorOr<CPUSystemInfo> M::Detail::getLinuxX86CPUSystemInfo() {
  // Only consider cpuIDs which are already in the affinity set of the
  // calling thread. This way we'll respect any restrictions already set.
  cpu_set_t callersAffinity;
  if (int rc = sched_getaffinity(0, sizeof(callersAffinity), &callersAffinity))
    return Error("can't retrieve schedule affinity for main thread: " +
                 std::to_string(rc));

  // Read /proc/cpuinfo as a stream (until EOF reached). It cannot be
  // mmapped because it appears to have 0 size.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/cpuinfo");
  if (std::error_code ec = errOrBuf.getError())
    return Error("can't open /proc/cpuinfo: " + ec.message());

  return getLinuxX86CPUSystemInfoImpl(callersAffinity, std::move(*errOrBuf));
}
#endif

ErrorOr<CPUSystemInfo> CPUSystemInfo::get() {
#if HAVE_LINUX_X86_SYSTEM_INFO
  return Detail::getLinuxX86CPUSystemInfo();
#endif
  return Error("CPUSystemInfo is not supported by this build");
}

size_t M::getNumPhysicalCores() {
  auto threadStrat = llvm::hardware_concurrency();
  threadStrat.UseHyperThreads = false;
  return threadStrat.compute_thread_count();
}

size_t M::getNumLogicalCores() {
  return llvm::hardware_concurrency().compute_thread_count();
}

/// Returns the number of physical performance cores across all CPU sockets. If
/// not known, will return the total number of physical cores.
/// TODO: add implementations for Windows and Linux
size_t M::getNumPerformanceCores() {
#if defined(__APPLE__)
  // Attempt to read the sysctl "hw.perflevel0.physicalcpu", which contains the
  // number of local performance cores. This is described here [1]. This can be
  // used to create a runtime if it is available to avoid contention and lagging
  // on the efficiency cores. We rely on the operating system to keep busy
  // threads running on the performance cores rather than any explicit affinity.
  //
  // Per sysctl(2), this type is expected to be int32_t.
  //
  // [1]
  // https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_system_capabilities
  int32_t pcores;
  size_t len = sizeof(pcores);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &len, nullptr, 0) ==
          0 &&
      pcores > 0)
    return static_cast<size_t>(pcores);
  return M::getNumPhysicalCores();
#elif HAVE_LINUX_X86_SYSTEM_INFO
  // Detect hybrid cores first.
  // https://www.intel.com/content/www/us/en/developer/articles/guide/12th-gen-intel-core-processor-gamedev-guide.html
  unsigned eax = 7, ecx = 0, ebx, edx;
  native_cpuid(&eax, &ebx, &ecx, &edx);
  if (((edx >> 15) & 0x1) == 0)
    return M::getNumPhysicalCores();
  cpu_set_t callersAffinity;
  if (sched_getaffinity(0, sizeof(callersAffinity), &callersAffinity))
    return M::getNumPhysicalCores();
  auto sysInfo = (Detail::getLinuxX86CPUSystemInfo()).get();
  int count = 0;
  for (auto socket : sysInfo.sockets)
    for (auto core : socket.physicalCores)
      for (auto vc : core.virtualCores) {
        cpu_set_t newAffinity;
        CPU_ZERO(&newAffinity);
        CPU_SET(vc.cpuID, &newAffinity);
        if (sched_setaffinity(0, sizeof(newAffinity), &newAffinity)) {
          sched_setaffinity(0, sizeof(callersAffinity), &callersAffinity);
          return M::getNumPhysicalCores();
        }
        unsigned eax = 26, ecx = 0, ebx, edx;
        native_cpuid(&eax, &ebx, &ecx, &edx);
        // bits 24-31 of eax determine the coreType.
        // https://www.intel.com/content/www/us/en/developer/articles/guide/12th-gen-intel-core-processor-gamedev-guide.html
        if (((eax >> 24) & 0xFF) == 64) {
          ++count;
          break;
        }
      }
  sched_setaffinity(0, sizeof(callersAffinity), &callersAffinity);
  if (count == 0)
    return M::getNumPhysicalCores();
  return count;
#else
  return M::getNumPhysicalCores();
#endif
}

void CPUSystemInfo::print(raw_ostream &os) const {
  os << "CPUSystemInfo(";
  llvm::interleave(
      sockets, os,
      [&](const Socket &s) {
        os << "Socket(";
        llvm::interleave(
            s.physicalCores, os,
            [&](const PhysicalCore &pc) {
              os << "{";
              llvm::interleave(
                  pc.virtualCores, os,
                  [&](const VirtualCore &vc) { os << vc.cpuID; }, ", ");
              os << "}";
            },
            ", ");
        os << ")";
      },
      ", ");
  os << ")";
}

std::vector<size_t> CPUSystemInfo::getPreferredCpuIDs(size_t numThreads) const {
  std::vector<size_t> cpuIDs;
  size_t virtualCoreIndex = 0;
  while (true) {
    // Pull all virtual cores matching this particular index. We will make a
    // decision whether to add this group at the group, based on whether this
    // creates an uneven distribution.
    std::vector<size_t> currentPass;
    for (const auto &socket : sockets) {
      for (const auto &physicalCore : socket.physicalCores) {
        if (virtualCoreIndex < physicalCore.virtualCores.size() &&
            currentPass.size() < numThreads) {
          currentPass.emplace_back(
              physicalCore.virtualCores[virtualCoreIndex].cpuID);
        }
      }
    }

    if (currentPass.size() == 0) {
      // If we added nothing in this pass, then start recycling.
      virtualCoreIndex = 0;
    } else if (currentPass.size() + cpuIDs.size() <= numThreads) {
      // If this is less than the intended size, we can add this batch.
      cpuIDs.insert(cpuIDs.end(), currentPass.begin(), currentPass.end());
      ++virtualCoreIndex;
    } else {
      // Otherwise, we will add a batch of unpinned CPUs at the end. We don't
      // generally expected this; it should happen only when the precise number
      // of cores has been specified up front (and is greater than the number of
      // physical cores, but does not segment them appropriately).
      for (size_t i = cpuIDs.size(); i < numThreads; ++i)
        cpuIDs.emplace_back(kNoAffinity);
    }
    assert(cpuIDs.size() <= numThreads);
    if (cpuIDs.size() == numThreads)
      return cpuIDs;
  }
}

static std::string bytesToHexStr(uint8_t *ptr, int count) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  for (int i = 0; i < count; ++i) {
    if (i > 0)
      ss << ":";
    ss << llvm::format_hex_no_prefix(ptr[i], 2, true);
  }
  return s;
}

std::vector<std::string> M::localMACs() {
  std::vector<std::string> macs;
#ifdef _WIN32
  IP_ADAPTER_INFO AdapterInfo[32];
  DWORD dwBufLen = sizeof(AdapterInfo);
  DWORD dwStatus = GetAdaptersInfo(AdapterInfo, &dwBufLen);
  if (dwStatus == ERROR_SUCCESS) {
    PIP_ADAPTER_INFO pAdapterInfo = AdapterInfo;
    while (pAdapterInfo) {
      macs.emplace_back(
          bytesToHexStr(pAdapterInfo->Address, pAdapterInfo->AddressLength));
      pAdapterInfo = pAdapterInfo->Next;
    }
  }
#else
#ifdef AF_PACKET
#define M_AF_TYPE AF_PACKET
#else
#define M_AF_TYPE AF_LINK
#endif
  struct ifaddrs *ifap, *ifaptr;
  if (getifaddrs(&ifap) == 0) {
    for (ifaptr = ifap; ifaptr != nullptr; ifaptr = (ifaptr)->ifa_next) {
      if (ifaptr->ifa_addr != nullptr &&
          ifaptr->ifa_addr->sa_family == M_AF_TYPE) {
        uint8_t *data = reinterpret_cast<uint8_t *>(
            (struct sockaddr_dl *)(ifaptr)->ifa_addr);
        macs.emplace_back(bytesToHexStr(data, 6));
      }
    }
    freeifaddrs(ifap);
  }
#undef M_AF_TYPE
#endif
  return macs;
}

#if HAVE_LINUX_X86_SYSTEM_INFO
std::optional<size_t> Detail::linuxCPULimits::toMillicores() const {
  if (quota_us >= 0 && period_us > 0)
    return (uint64_t(1000) * quota_us) / period_us;
  return std::nullopt;
}
#endif

ErrorOr<CPULimits> CPULimits::get() {
#if HAVE_LINUX_X86_SYSTEM_INFO
  CPULimits limits;
  limits.millicores = Detail::getLinuxCPULimits().toMillicores();
  return limits;
#endif
  return Error("CPULimits are not supported by this build");
}

static ErrorOr<NUMATopology> queryNumaTopology() {
#if HAVE_LINUX_X86_SYSTEM_INFO
  NUMATopology topology;

  // Enumerate NUMA nodes from /sys/devices/system/node/
  std::error_code err;
  for (const auto &entry :
       std::filesystem::directory_iterator("/sys/devices/system/node", err)) {
    if (err)
      break;
    std::string filename = entry.path().filename().string();
    if (filename.rfind("node", 0) == 0) {
      int nodeId;
      if (!StringRef(filename).substr(4).getAsInteger(10, nodeId))
        topology.numaNodes.push_back(nodeId);
    }
  }
  if (err)
    return Error("Failed to enumerate NUMA nodes: " + err.message());

  if (topology.numaNodes.empty())
    return Error("No NUMA nodes found");

  llvm::sort(topology.numaNodes);

  // For each NUMA node, get CPU IDs from cpulist file.
  for (int node : topology.numaNodes) {
    std::string cpuListPath =
        "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
    auto buf = fileBuffer(cpuListPath);
    if (buf) {
      auto cpuIdsOr = parseCpuList(buf->getBuffer());
      if (!cpuIdsOr.isError()) {
        topology.cpuIdsPerNumaNode[node] = std::move(*cpuIdsOr);
        for (size_t cpuId : topology.cpuIdsPerNumaNode[node])
          topology.cpuIdToNumaNode[cpuId] = node;
      }
    }
  }

  // Enumerate PCI devices and map them to NUMA nodes.
  for (const auto &entry :
       std::filesystem::directory_iterator("/sys/bus/pci/devices", err)) {
    if (err)
      break;

    std::string pciBusId = entry.path().filename().string();

    // Normalize to 8-digit domain format to match CUDA's format
    // e.g., "0000:17:00.0" -> "00000000:17:00.0"
    size_t firstColon = pciBusId.find(':');
    if (firstColon != std::string::npos && firstColon < 8)
      pciBusId = std::string(8 - firstColon, '0') + pciBusId;

    // Convert to uppercase to match CUDA's format
    // e.g., "00000000:1a:00.0" -> "00000000:1A:00.0"
    std::transform(pciBusId.begin(), pciBusId.end(), pciBusId.begin(),
                   ::toupper);

    std::string numaPath = entry.path().string() + "/numa_node";

    auto buf = fileBuffer(numaPath);
    if (!buf)
      continue;

    int deviceNumaNode;
    if (buf->getBuffer().trim().getAsInteger(10, deviceNumaNode))
      continue;

    topology.pciBusToNumaNode[pciBusId] = deviceNumaNode;
    topology.pciBusesPerNumaNode[deviceNumaNode].push_back(pciBusId);
  }

  // Sort PCI buses per NUMA node for consistent ordering.
  for (auto &[node, buses] : topology.pciBusesPerNumaNode)
    llvm::sort(buses);

  return topology;
#else
  return Error("NUMATopology is only supported on Linux x86");
#endif
}

const ErrorOr<NUMATopology> &NUMATopology::get() {
  static const ErrorOr<NUMATopology> cachedNUMATopology = queryNumaTopology();
  return cachedNUMATopology;
}

std::vector<size_t> NUMATopology::getCpuIdsForNumaNode(int numaNode) const {
  auto it = cpuIdsPerNumaNode.find(numaNode);
  if (it != cpuIdsPerNumaNode.end())
    return it->second;
  return {};
}

int NUMATopology::getNumaNodeForCpuId(size_t cpuId) const {
  auto it = cpuIdToNumaNode.find(cpuId);
  return it != cpuIdToNumaNode.end() ? it->second : kAnyNumaNode;
}

std::vector<std::string>
NUMATopology::getPciBusesForNumaNode(int numaNode) const {
  auto it = pciBusesPerNumaNode.find(numaNode);
  if (it != pciBusesPerNumaNode.end())
    return it->second;
  return {};
}

int NUMATopology::getNumaNodeForPciBus(StringRef pciBusId) const {
  auto it = pciBusToNumaNode.find(pciBusId.str());
  if (it != pciBusToNumaNode.end())
    return it->second;
  return -1;
}
