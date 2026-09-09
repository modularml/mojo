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

#include "Support/Buffer.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/Filesystem/DiskUsage.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace M;
using mapped_file_region = llvm::sys::fs::mapped_file_region;

namespace M {

/// Open a file in read-only or read-write mode and return its file descriptor
/// and the status object for the file so we can get things like its size.
ErrorOr<std::pair<llvm::sys::fs::file_t, llvm::sys::fs::file_status>>
openFile(const std::filesystem::path &filepath, bool readOnly) {
  std::string filepathStr = filepath.string();

  llvm::sys::fs::file_t fd;
  // llvm::Expected doesn't have a default constructor...so we have to duplicate
  // some code.
  if (readOnly) {
    llvm::Expected<llvm::sys::fs::file_t> fdOr =
        llvm::sys::fs::openNativeFileForRead(filepathStr,
                                             llvm::sys::fs::OF_None);
    if (!fdOr)
      return Error(toString(fdOr.takeError()));
    fd = *fdOr;
  } else {
    llvm::Expected<llvm::sys::fs::file_t> fdOr =
        llvm::sys::fs::openNativeFileForReadWrite(
            filepathStr, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::OF_None);
    if (!fdOr)
      return Error(toString(fdOr.takeError()));
    fd = *fdOr;
  }

  llvm::sys::fs::file_status status;
  if (std::error_code err = llvm::sys::fs::status(fd, status))
    return Error(err.message());

  // If this not a file or a block device (e.g. it's a named pipe
  // or character device), we can't mmap it, so error out.
  llvm::sys::fs::file_type type = status.type();
  if (type != llvm::sys::fs::file_type::regular_file &&
      type != llvm::sys::fs::file_type::block_file)
    return Error("cannot map file that is not an actual file or block device");

  return std::make_pair(fd, status);
}

} // namespace M

//===----------------------------------------------------------------------===//
// Buffer
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> Buffer::getFile(const std::filesystem::path &filepath,
                                   std::optional<size_t> size,
                                   std::optional<size_t> offset) {
  auto fdOr = openFile(filepath, /*readOnly=*/true);
  if (fdOr.isError())
    return fdOr.takeError();
  llvm::sys::fs::file_t fd = fdOr->first;
  llvm::sys::fs::file_status status = fdOr->second;

  // If no size was provided, use the file's size.
  if (!size)
    size = status.getSize();

  // If the size is zero, then we have an empty buffer. Since Buffer is
  // read-only, we can simply return an empty string.
  if (size == 0)
    return BufferRef::create("");

  std::error_code ec;
  llvm::sys::fs::mapped_file_region mappedFile(
      fd, llvm::sys::fs::mapped_file_region::readonly, *size,
      offset.value_or(0), ec);
  if (ec)
    return Error(ec.message());

  // Close the file, mmap will hold a ref to the descriptor.
  llvm::sys::fs::closeFile(fd);

  return BufferRef::create(filepath, std::move(mappedFile));
}

std::optional<std::filesystem::path> Buffer::getFilePath() const {

  if (isa<MappedBufferStorage>(storage))
    return cast<MappedBufferStorage>(storage).filePath;

  if (isa<MemoryBufferStorage>(storage)) {
    StringRef filePath =
        cast<MemoryBufferStorage>(storage).memBuffer->getBufferIdentifier();
    if (filePath.empty())
      return std::nullopt;
    return std::filesystem::path(filePath.str());
  }

  return std::nullopt;
}

const char *Buffer::getBufferStart() const {
  if (isa<AllocatedBuffer>(storage))
    return cast<AllocatedBuffer>(storage).data.data();

  if (isa<MappedBufferStorage>(storage))
    return cast<MappedBufferStorage>(storage).mapping.const_data();

  if (isa<MemoryBufferStorage>(storage))
    return cast<MemoryBufferStorage>(storage).memBuffer->getBufferStart();

  if (isa<AliasedBufferStorage>(storage))
    return cast<AliasedBufferStorage>(storage).aliasContents.begin();

  llvm_unreachable("unknown storage type");
}

const char *Buffer::getBufferEnd() const {
  if (isa<AllocatedBuffer>(storage)) {
    auto &allocStorage = cast<AllocatedBuffer>(storage);
    return allocStorage.data.data() + allocStorage.data.size();
  }

  if (isa<MappedBufferStorage>(storage)) {
    auto &mappedStorage = cast<MappedBufferStorage>(storage).mapping;
    return mappedStorage.const_data() + mappedStorage.size();
  }

  if (isa<MemoryBufferStorage>(storage))
    return cast<MemoryBufferStorage>(storage).memBuffer->getBufferEnd();

  if (isa<AliasedBufferStorage>(storage))
    return cast<AliasedBufferStorage>(storage).aliasContents.end();

  llvm_unreachable("unknown storage type");
}

size_t Buffer::getBufferSize() const {
  if (isa<AllocatedBuffer>(storage))
    return cast<AllocatedBuffer>(storage).data.size();

  if (isa<MappedBufferStorage>(storage))
    return cast<MappedBufferStorage>(storage).mapping.size();

  if (isa<MemoryBufferStorage>(storage))
    return cast<MemoryBufferStorage>(storage).memBuffer->getBufferSize();

  if (isa<AliasedBufferStorage>(storage))
    return cast<AliasedBufferStorage>(storage).aliasContents.size();

  llvm_unreachable("unknown storage type");
}

size_t Buffer::getBufferCapacity() const {
  if (isa<AllocatedBuffer>(storage))
    return cast<AllocatedBuffer>(storage).data.capacity();

  if (isa<MappedBufferStorage>(storage))
    return cast<MappedBufferStorage>(storage).mapping.size();

  if (isa<MemoryBufferStorage>(storage))
    return cast<MemoryBufferStorage>(storage).memBuffer->getBufferSize();

  if (isa<AliasedBufferStorage>(storage))
    return cast<AliasedBufferStorage>(storage).aliasContents.size();

  llvm_unreachable("unknown storage type");
}

StringRef Buffer::getBuffer() const {
  return StringRef(getBufferStart(), getBufferSize());
}

llvm::MemoryBufferRef Buffer::getMemBufferRef() const {
  return llvm::MemoryBufferRef(getBuffer(), /*Identifier=*/"");
}

//===----------------------------------------------------------------------===//
// WriteableBuffer getFile
//===----------------------------------------------------------------------===//

ErrorOr<WriteableBufferRef>
WriteableBuffer::getFile(const std::filesystem::path &filepath, size_t size,
                         size_t offset) {
  auto parentPath = filepath.parent_path();
  // For cases where the file doesn't exist, the `openFile` below can create it,
  // but the root path will not exist, so we will default to the current working
  // directory to get the available size.
  if (parentPath.empty())
    parentPath = std::filesystem::current_path();

  auto availableSizeOr = M::getAvailableDiskSpace(parentPath);
  if (availableSizeOr.isError())
    return availableSizeOr.takeError();

  if (*availableSizeOr < size)
    return Error("available space in disk less than requested size");

  auto fdOr = openFile(filepath, /*readOnly=*/false);
  if (fdOr.isError())
    return fdOr.takeError();
  llvm::sys::fs::file_t fd = fdOr->first;
  llvm::sys::fs::file_status status = fdOr->second;

  // Handle the size. If no size was provided, use the file's size. Otherwise,
  // resize the file before we map it in.
  if (size == 0) {
    size = status.getSize();
  } else if (status.getSize() < size) {
    // On Windows, the resize_file_before_mapping_readwrite is a no-op which
    // takes an integer file handle (and not an llvm::fs::file_t). To avoid
    // compilation failure, we just skip calling the
    // resize_file_before_mapping_readwrite function.
#ifndef _WIN32
    if (auto err =
            llvm::sys::fs::resize_file_before_mapping_readwrite(fd, size))
      return Error(err.message());
#endif // _WIN32
  }

  std::error_code ec;
  llvm::sys::fs::mapped_file_region mappedFile(
      fd, llvm::sys::fs::mapped_file_region::readwrite, size, offset, ec);
  if (ec)
    return Error(ec.message());

  // Close the file, mmap will hold a ref to the descriptor.
  llvm::sys::fs::closeFile(fd);

  return WriteableBufferRef::create(filepath, std::move(mappedFile));
}

//===----------------------------------------------------------------------===//
// WriteableBuffer raw_pwrite_stream implementation
//===----------------------------------------------------------------------===//

void WriteableBuffer::write_impl(const char *ptr, size_t size) {
  assert(!isa<AliasedBufferStorage>(storage) &&
         "cannot `write` to an aliased buffer");
  assert(!isa<MemoryBufferStorage>(storage) &&
         "cannot `write` to an llvm::MemoryBuffer backed buffer");

  if (isa<AllocatedBuffer>(storage)) {
    auto &allocStorage = cast<AllocatedBuffer>(storage);
    allocStorage.data.append(ptr, size);
    return;
  }

  auto &mappedStorage = cast<MappedBufferStorage>(storage);
  // Ensure we have the space to do the write. This means that the number of
  // bytes we want to write must be less than the size remaining in the mapped
  // buffer, which is exactly the total mapping size minus the already-written
  // bytes.
  assert(size <= mappedStorage.mapping.size() -
                     (uintptr_t)(mappedStorage.write -
                                 mappedStorage.mapping.data()) &&
         "too many bytes to write to this mapping");
  memcpy(mappedStorage.write, ptr, size);
  // Increment the write pointer.
  mappedStorage.write += size;
}

uint64_t WriteableBuffer::current_pos() const {
  if (isa<MappedBufferStorage>(storage)) {
    auto &mappedStorage = cast<MappedBufferStorage>(storage);
    return mappedStorage.write - mappedStorage.mapping.data();
  }

  return getBufferSize();
}

// This implementation is essentially translated from the implementation of
// llvm::raw_svector_ostream.
void WriteableBuffer::pwrite_impl(const char *ptr, size_t size,
                                  uint64_t offset) {
  assert(!isa<AliasedBufferStorage>(storage) &&
         "cannot `pwrite` to an aliased buffer");
  assert(!isa<MemoryBufferStorage>(storage) &&
         "cannot `pwrite` to an llvm::MemoryBuffer backed buffer");

  // TODO: currently we don't resize the mmap'd buffer.
  assert(getBufferStart() + offset + size <= getBufferEnd() ||
         isa<AllocatedBuffer>(storage));

  if (isa<MappedBufferStorage>(storage)) {
    memcpy(cast<MappedBufferStorage>(storage).mapping.data() + offset, ptr,
           size);
    return;
  }

  // We currently don't support writing to a llvm::MemoryBuffer-backed buffer.
  auto &allocStorage = cast<AllocatedBuffer>(storage);
  memcpy(allocStorage.data.data() + offset, ptr, size);
}
