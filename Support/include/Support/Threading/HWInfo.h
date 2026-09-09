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

#ifndef SUPPORT_THREADING_HWINFO_H
#define SUPPORT_THREADING_HWINFO_H

#include "Support/ErrorOr.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <sched.h>
#include <string>
#include <vector>

#if defined(__linux__) && (defined(__i386__) || defined(__x86_64__))
#define HAVE_LINUX_X86_SYSTEM_INFO 1
#else
#define HAVE_LINUX_X86_SYSTEM_INFO 0
#endif

namespace M {

/// The distinguished CPU ID denoting 'no affinity to be set'.
constexpr size_t kNoAffinity = ~0;

//===----------------------------------------------------------------------===//
// CPUSystemInfo
//===----------------------------------------------------------------------===//

/// Describes the sockets, physical cores, and virtual cores in a CPU system
/// when supported by the host os. However does not capture sharing of caches
/// and plethora of other details. See https://www.open-mpi.org/projects/hwloc/
struct CPUSystemInfo {
  /// A 'virtual' core, generally without dedicated cache or ALU resources.
  /// Systems with hyperthreading can have multiple virtual cores per
  /// 'physical' core.
  struct VirtualCore {
    size_t cpuID;

    VirtualCore(size_t cpuID) : cpuID(cpuID) {}
  };

  /// A 'physical' core, generally with its own dedicated cache levels.
  struct PhysicalCore {
    llvm::SmallVector<VirtualCore, 2> virtualCores;
  };

  /// A 'socket', generally with its own NUMA memory area and dedicated cache
  /// levels.
  struct Socket {
    llvm::SmallVector<PhysicalCore, 16> physicalCores;
  };

  llvm::SmallVector<Socket, 1> sockets;

  /// Returns system info if it can be determined. Fidelity with respect to
  /// actual hardware may vary depending on host OS. Returns an error if system
  /// info cannot be determined.
  static ErrorOr<CPUSystemInfo> get();

  /// Returns numThreads cpuIDs drawn from this system info, following these
  /// heuristics (from strongest to weakest):
  ///  - prefer threads on distinct virtual cores
  ///  - prefer virtual cores on distinct physical cores
  ///  - prefer physical cores on the same socket
  /// If numThreads exceeds the number of virtual cores in the system then
  /// cpu IDs will be repeated in the result.
  std::vector<size_t> getPreferredCpuIDs(size_t numThreads) const;

  void print(raw_ostream &os) const;
};

/// Sentinel value used to indicate that a resource has no NUMA affinity.
constexpr int kAnyNumaNode = -1;

/// Describes the NUMA topology of the system, including NUMA nodes, CPU IDs
/// per node, and PCI bus to NUMA node mappings.
struct NUMATopology {
  std::vector<int> numaNodes;
  std::map<int, std::vector<size_t>> cpuIdsPerNumaNode;
  std::map<size_t, int> cpuIdToNumaNode;
  std::map<int, std::vector<std::string>> pciBusesPerNumaNode;
  std::map<std::string, int> pciBusToNumaNode;

  /// Returns the process-wide NUMA topology, querying the system on the first
  /// call and caching the result (success or error) for the remainder of the
  /// process lifetime. Thread-safe via C++11 static initialization.
  static const ErrorOr<NUMATopology> &get();

  /// Returns the list of NUMA node IDs available in the system.
  const std::vector<int> &getNumaNodes() const { return numaNodes; }

  /// Returns the list of CPU IDs belonging to a NUMA node.
  std::vector<size_t> getCpuIdsForNumaNode(int numaNode) const;

  /// Returns the NUMA node ID for a given CPU ID, or kAnyNumaNode if the CPU
  /// is not found in the topology.
  int getNumaNodeForCpuId(size_t cpuId) const;

  /// Returns the list of PCI bus addresses belonging to a NUMA node.
  std::vector<std::string> getPciBusesForNumaNode(int numaNode) const;

  /// Returns the NUMA node ID for a given PCI bus address, or -1 if the PCI
  /// bus is not found.
  int getNumaNodeForPciBus(StringRef pciBusId) const;
};

inline raw_ostream &operator<<(raw_ostream &os, const CPUSystemInfo &info) {
  info.print(os);
  return os;
}

/// Returns the number of physical cores across all CPU sockets
size_t getNumPhysicalCores();

/// Returns the number of hardware threads, including hyperthreads across all
/// CPU sockets
size_t getNumLogicalCores();

/// Returns the number of physical performance cores across all CPU sockets. If
/// not known, will return the total number of physical cores.
size_t getNumPerformanceCores();

/// Returns the set of local MAC addresses.
std::vector<std::string> localMACs();

/// Describes CPU limits in an OS-agnostic way.
struct CPULimits {
  /// Unfortunately, millicores are a canonical way of representing the limit,
  /// even though it has far more subtlety than this.
  std::optional<size_t> millicores;

  /// Returns local limits, if available.
  static ErrorOr<CPULimits> get();
};

//===----------------------------------------------------------------------===//
// OS and architecture-specific utilities, visible for testing only
//===----------------------------------------------------------------------===//

namespace Detail {
#if HAVE_LINUX_X86_SYSTEM_INFO
/// Specifies CPU quota per period of CPU time allotted by the Linux CFS.
struct linuxCPULimits {
  int quota_us = -1;
  int period_us = 100000;

  /// Converts to millicores, if a quota is set.
  std::optional<size_t> toMillicores() const;
};

/// Returns the cgroup v1 CPU membership from |buf|.
ErrorOr<std::string> parseV1CPUCgroupFile(const llvm::MemoryBuffer &buf);
ErrorOr<linuxCPULimits> parseV1CPULimits(const llvm::MemoryBuffer &quotaBuf,
                                         const llvm::MemoryBuffer &periodBuf);

/// Returns the effective cgroup v2 CPU membership from |buf|. This is
/// determined by searching /sys/fs/cgroup/ until a cpu.max file is found.
ErrorOr<std::string>
parseV2CPUCgroupFile(const llvm::MemoryBuffer &buf,
                     const std::function<bool(StringRef)> &exists);
ErrorOr<linuxCPULimits> parseV2CPULimits(const llvm::MemoryBuffer &maxBuf);

linuxCPULimits getLinuxCPULimits();

ErrorOr<CPUSystemInfo>
getLinuxX86CPUSystemInfoImpl(const cpu_set_t &availableCpus,
                             std::unique_ptr<llvm::MemoryBuffer> buf);
ErrorOr<CPUSystemInfo> getLinuxX86CPUSystemInfo();
#endif // HAVE_LINUX_X86_SYSTEM_INFO

} // namespace Detail
} // namespace M

#endif // SUPPORT_THREADING_HWINFO_H
