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

#include "Support/CPUCache.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <system_error>
#include <utility>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif // __APPLE__

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif // __linux__

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif // _MSC_VER

using namespace M;

//===----------------------------------------------------------------------===//
// Cache sizes
//===----------------------------------------------------------------------===//

#ifdef __linux__
namespace {
class FileDescriptorCloser final {
public:
  explicit FileDescriptorCloser(int fd) : fd(fd) {}
  FileDescriptorCloser(const FileDescriptorCloser &) = delete;
  ~FileDescriptorCloser() {
    [[maybe_unused]] std::error_code ec =
        llvm::sys::Process::SafelyCloseFileDescriptor(fd);
    assert(!ec && "Error encountered closing read-only file descriptor, which "
                  "should never fail");
  }
  FileDescriptorCloser &operator=(const FileDescriptorCloser &) = delete;

private:
  int fd;
};
} // namespace

static M::ErrorOr<size_t>
readSmallFileFromDirFD(int dirFD, const char *relPath,
                       const llvm::Twine &fileDescription, char *buffer,
                       size_t bufferSize) {
  int fd = openat(dirFD, relPath, O_RDONLY);
  if (fd == -1)
    return Error("Could not open " + fileDescription + ": " + strerror(errno));
  FileDescriptorCloser fdCloser(fd);
  ssize_t nRead = read(fd, buffer, bufferSize);
  if (nRead == -1)
    return Error("Could not read " + fileDescription + ": " + strerror(errno));
  if (static_cast<size_t>(nRead) == bufferSize)
    return Error("File for " + fileDescription +
                 " too large to read into fixed-size buffer");
  return static_cast<size_t>(nRead);
}
#endif // __linux__

M::ErrorOr<size_t> M::getHostCPUCacheSize(size_t cacheLevel) {
#if defined(__APPLE__)
  size_t result;
  size_t len = sizeof(result);
  switch (cacheLevel) {
  case 1:
    if (sysctlbyname("hw.l1dcachesize", &result, &len, nullptr, 0))
      return Error("unable to query the hw.l1dcachesize");
    return result;
  case 2:
    if (sysctlbyname("hw.l2cachesize", &result, &len, nullptr, 0))
      return Error("unable to query the hw.l2cachesize");
    return result;
  default:
    return 0;
  }
#elif defined(__linux__)
  int cacheDirFD =
      open("/sys/devices/system/cpu/cpu0/cache", O_DIRECTORY | O_PATH);
  if (cacheDirFD == -1) {
    if (errno == ENOENT) {
      // There are some times, for example inside of a Docker container on Mac,
      // where the file is not there as expected. For now, hardcoding returning
      // 0 in that case, with maybe a bit of smarts in the future where the
      // host can specify what the cache is.
      return 0;
    }
    return Error("Could not open CPU0 cache directory: " +
                 llvm::Twine(strerror(errno)));
  }
  FileDescriptorCloser cacheDirFDCloser(cacheDirFD);

  for (int index = 0;; ++index) {
    char relPath[32];
    sprintf(relPath, "index%d", index);
    int cacheDirIndexFD = openat(cacheDirFD, relPath, O_DIRECTORY | O_PATH);
    if (cacheDirIndexFD == -1) {
      if (errno == ENOENT)
        break;
      return Error("Could not open cache index directory at index " +
                   llvm::Twine(index) + ": " + strerror(errno));
    }
    FileDescriptorCloser cacheDirIndexFDCloser(cacheDirIndexFD);

    char levelBuf[32], typeBuf[32], sizeBuf[32];
    auto levelLenOr =
        readSmallFileFromDirFD(cacheDirIndexFD, "level",
                               "cache index " + llvm::Twine(index) + " level",
                               levelBuf, sizeof(levelBuf));
    if (levelLenOr.isError())
      return levelLenOr.takeError();
    auto levelLen = std::move(*levelLenOr);
    auto typeLenOr = readSmallFileFromDirFD(
        cacheDirIndexFD, "type", "cache index " + llvm::Twine(index) + " type",
        typeBuf, sizeof(typeBuf));
    if (typeLenOr.isError())
      return typeLenOr.takeError();
    auto typeLen = std::move(*typeLenOr);
    auto sizeLenOr = readSmallFileFromDirFD(
        cacheDirIndexFD, "size", "cache index " + llvm::Twine(index) + " size",
        sizeBuf, sizeof(sizeBuf));
    // The file might not exist on certain VMs (like multipass on Mac)
    if (sizeLenOr.isError())
      continue;
    auto sizeLen = std::move(*sizeLenOr);
    StringRef levelStr = StringRef(levelBuf, levelLen).trim();
    StringRef typeStr = StringRef(typeBuf, typeLen).trim();
    StringRef sizeStr = StringRef(sizeBuf, sizeLen).trim();

    size_t level;
    if (levelStr.getAsInteger(10, level))
      return Error("Could not parse cache index " + llvm::Twine(index) +
                   " level");
    if (level != cacheLevel)
      continue;

    if (typeStr != "Data" && typeStr != "Unified")
      continue;

    // Linux hard-codes the unit as K, so this should never trip unless the
    // interface is changed or something else is wrong.
    if (!sizeStr.consume_back("K"))
      return Error("Cache size at index " + llvm::Twine(index) +
                   " is not specified in K");
    size_t sizeInK;
    if (sizeStr.getAsInteger(10, sizeInK))
      return Error("Could not parse cache index " + llvm::Twine(index) +
                   " size");
    return sizeInK * 1024;
  }
  return 0;
#elif defined(_MSC_VER)

  // We can only get info for L1, L2 & L3 cache.
  if (cacheLevel >= 4)
    return 0;

  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> processorInfos;
  DWORD bufferLength = 0;

  DWORD returnCode =
      GetLogicalProcessorInformation(processorInfos.data(), &bufferLength);

  if (!returnCode) {

    // This is the only error where there is a reason for retry.
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
      return Error(llvm::mapWindowsError(GetLastError()).message());
    processorInfos.resize(llvm::divideCeil(
        bufferLength, sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)));

    // Try once again with the new buffer length and pre allocated buffer.
    returnCode =
        GetLogicalProcessorInformation(processorInfos.data(), &bufferLength);

    // We can recheck for insufficient buffer length and keep on doing this
    // but it should be pretty rare to fail twice with that reason. So we will
    // bail out.
    if (!returnCode)
      return Error(llvm::mapWindowsError(GetLastError()).message());
  }

  for (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION &processorInfo :
       processorInfos) {

    if (processorInfo.Relationship == RelationCache) {
      const CACHE_DESCRIPTOR &cache = processorInfo.Cache;
      if (cache.Level == cacheLevel &&
          (cache.Type == CacheData || cache.Type == CacheUnified))
        return cache.Size;
    }
  }

  return 0;
#else
  return Error("unsupported platform");
#endif
}
