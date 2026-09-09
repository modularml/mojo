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

#ifndef SUPPORT_BUFFER_H
#define SUPPORT_BUFFER_H

#include "Support/ADT/SmartVariant.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace M {

/// Open a file in read-only or read-write mode and return its file descriptor
/// and the status object for the file so we can get things like its size.
ErrorOr<std::pair<llvm::sys::fs::file_t, llvm::sys::fs::file_status>>
openFile(const std::filesystem::path &filepath, bool readOnly);

class Buffer;
using BufferRef = RCRef<Buffer>;

/// Provides a reference-counted version of an LLVM memory buffer that owns its
/// data. This is useful for caching where you might want to store a buffer
/// that can't be deallocated until it's been (asynchronously) stored in the
/// cache. Buffer is read-only, for writing one should use WriteableBuffer
/// (defined below).
// TODO: Should this hold a reference to a AsyncRT::Allocator and use that to
//       allocate memory?
class Buffer : public ReferenceCounted<Buffer> {
public:
  /// Destroy a buffer. Releases any resources associated with that buffer.
  virtual ~Buffer() = default;

  /// Create a buffer from a StringRef of data. This will copy the data into the
  /// resulting BufferRef.
  static BufferRef get(StringRef data) { return BufferRef::create(data); }

  /// Create a buffer that is an alias of `other`, starting at `begin` with size
  /// `size`. Providing the default (0) for both parameters results in a full
  /// alias of the entire buffer. This will store a ref to `other` inside this
  /// buffer to ensure the lifetimes overlap sufficiently.
  static BufferRef getAlias(BufferRef other, size_t begin = 0,
                            size_t size = 0) {
    return BufferRef::create(std::move(other), begin, size);
  }

  /// Map in a file and use it as the backing storage for the BufferRef. If size
  /// and offset are provided, then a sub-range of the file is mapped in. This
  /// file is mapped read-only.
  static ErrorOr<BufferRef>
  getFile(const std::filesystem::path &filepath,
          std::optional<size_t> size = std::nullopt,
          std::optional<size_t> offset = std::nullopt);

  /// Return the location of file in filesystem if the buffer is backed by a
  /// file or the memory buffer backing the buffer is sourced from a file,
  /// std::nullopt otherwise.
  std::optional<std::filesystem::path> getFilePath() const;

  /// Take ownership of an `llvm::MemoryBuffer` and use that as the backing
  /// storage for the BufferRef.
  static BufferRef take(std::unique_ptr<llvm::MemoryBuffer> buffer) {
    return BufferRef::create(std::move(buffer));
  }

  //===-------------------------------------------------------------------===//
  // llvm::MemoryBuffer API
  //===-------------------------------------------------------------------===//

  /// Provide essentially the same API as llvm::MemoryBuffer.
  const char *getBufferStart() const;
  const char *getBufferEnd() const;
  size_t getBufferSize() const;
  size_t getBufferCapacity() const;
  StringRef getBuffer() const;
  llvm::MemoryBufferRef getMemBufferRef() const;

protected:
  /// So RCRef can access protected constructors.
  friend class RCRef<Buffer>;

  /// Create a Buffer of given size and alignment.
  Buffer(size_t size, std::optional<size_t> alignment,
         std::optional<size_t> capacity)
      : storage{AllocatedBuffer(size, alignment, capacity)} {}

  /// Construct the Buffer where it has to copy its data.
  Buffer(StringRef data) : storage{AllocatedBuffer(data)} {}

  /// Construct a buffer with a mapped file region. The buffer takes ownership
  /// of the mapped file region.
  Buffer(const std::filesystem::path &path,
         llvm::sys::fs::mapped_file_region &&mapped)
      : storage{std::move(mapped)} {
    cast<MappedBufferStorage>(storage).filePath = path;
  }

  /// Construct a `Buffer` from an `llvm::MemoryBuffer`. The buffer takes
  /// ownership of any storage owned by the `llvm::MemoryBuffer`.
  Buffer(std::unique_ptr<llvm::MemoryBuffer> buffer)
      : storage{std::move(buffer)} {}

  /// Given an offset and a size, construct an AliasedBufferStorage from
  /// the bytes of `other`.
  Buffer(BufferRef other, size_t begin, size_t size)
      : storage{AliasedBufferStorage(std::move(other), begin, size)} {}

  /// Buffers are not copy-constructible.
  Buffer(const Buffer &other) = delete;
  Buffer &operator=(const Buffer &other) = delete;

  /// A buffer whose memory is managed by a `std::basic_string`.
  struct AllocatedBuffer {
    AlignedAllocator<char> allocator;
    std::basic_string<char, std::char_traits<char>, AlignedAllocator<char>>
        data;

    /// Create the buffer with a given size, capacity, and alignment.
    AllocatedBuffer(size_t size = 0, std::optional<size_t> align = {},
                    std::optional<size_t> capacity = {})
        : allocator(align.value_or(alignof(std::max_align_t))),
          data(size, 0, allocator) {
      if (capacity)
        data.reserve(*capacity);
    }

    /// Construct a MallocdBuffer from a StringRef.
    AllocatedBuffer(StringRef str)
        : AllocatedBuffer(str.size(), alignof(std::max_align_t)) {
      data.assign(str.data(), str.size());
    }
  };

  /// Struct to hold the data we need if this is an llvm::MemoryBuffer.
  struct MemoryBufferStorage {
    MemoryBufferStorage(std::unique_ptr<llvm::MemoryBuffer> buffer)
        : memBuffer{std::move(buffer)} {
      assert(memBuffer && "expected a non-null memory buffer");
    }

    std::unique_ptr<llvm::MemoryBuffer> memBuffer;
  };

  /// A Buffer storage that aliases another buffer. This is a read-only slice -
  /// WriteableBuffer cannot use this storage kind (enforced with asserts, as
  /// well as not having a constructor that could result in an
  /// AliasedBufferStorage kind).
  struct AliasedBufferStorage {
    /// The parent buffer - this ensures we don't delete a buffer while it has
    /// live aliases.
    BufferRef parent;
    /// The alias to the bytes of `parent` that we care about here.
    StringRef aliasContents;

    /// Construct the storage object and build the alias StringRef.
    AliasedBufferStorage(BufferRef other, size_t begin, size_t size)
        : parent(std::move(other)) {
      // `size` == 0 implies we want the full buffer.
      if (size == 0)
        size = parent->getBufferSize();
      assert(begin + size <= parent->getBufferSize() &&
             "too many bytes selected");

      aliasContents = StringRef(parent->getBufferStart() + begin, size);
    }
  };

  struct MappedBufferStorage {
    /// The actual file mapping.
    llvm::sys::fs::mapped_file_region mapping;
    /// The current write pointer. Always starts at the beginning of the
    /// mapping.
    char *write;

    /// Path to mapped file.
    std::filesystem::path filePath;

    MappedBufferStorage(llvm::sys::fs::mapped_file_region &&mapped)
        : mapping(std::move(mapped)), write(mapping.data()) {}
  };

  /// The data owned by this buffer.
  /// AllocatedBuffer is the first type so that the default constructor is an
  /// empty AllocatedBuffer.
  SmartVariant<AllocatedBuffer, MappedBufferStorage, MemoryBufferStorage,
               AliasedBufferStorage>
      storage;
};

class WriteableBuffer;
using WriteableBufferRef = RCRef<WriteableBuffer>;

/// Subclass of Buffer that is write-able. It also owns its data in all cases.
class WriteableBuffer : public Buffer, public llvm::raw_pwrite_stream {
public:
  /// Create a WriteableBuffer with initial size (this sets both the capacity
  /// and the number of bytes stored in the buffer to `size`). The user can also
  /// provide an alignment for the underlying allocation.
  WriteableBuffer(size_t size = 0, std::optional<size_t> alignment = {},
                  std::optional<size_t> capacity = {})
      : Buffer(size, alignment, capacity) {
    SetUnbuffered();
  }

  static WriteableBufferRef get() { return WriteableBufferRef::create(); }
  /// Create a WriteableBuffer with initial size (this sets both the capacity
  /// and the number of bytes stored in the buffer to `size`). The user can also
  /// provide an alignment for the underlying allocation. When `size` is
  /// provided, that sets the current write pointer of the buffer to be begin()
  /// + `size` - meaning that `write` will append and the user should use
  /// `pwrite`. If the intent is to pre-allocate a buffer that will be
  /// filled-in, the caller should instead indicate a capacity, with a size of
  /// 0.
  static WriteableBufferRef get(size_t size,
                                std::optional<size_t> alignment = {},
                                std::optional<size_t> capacity = {}) {
    return WriteableBufferRef::create(size, alignment, capacity);
  }

  /// Map in a file and use it as the backing storage for the BufferRef. If
  /// size and offset are provided, then a sub-range of the file is mapped in.
  /// This file is mapped read/write.
  static ErrorOr<WriteableBufferRef>
  getFile(const std::filesystem::path &filepath, size_t size = 0,
          size_t offset = 0);

  /// Keep all the reader APIs.
  using Buffer::getBuffer;
  using Buffer::getBufferCapacity;
  using Buffer::getBufferEnd;
  using Buffer::getBufferSize;
  using Buffer::getBufferStart;

  char *getBufferStart() {
    return const_cast<char *>(Buffer::getBufferStart());
  }

  char *getBufferEnd() { return const_cast<char *>(Buffer::getBufferEnd()); }

  MutableArrayRef<char> getBuffer() {
    return {getBufferStart(), getBufferEnd()};
  }

  //===-------------------------------------------------------------------===//
  // raw_pwrite_stream implementation
  //===-------------------------------------------------------------------===//

  /// Copies `size` bytes from address `ptr` to the end of the buffer (this
  /// resizes the buffer to contain getBufferSize() + `size` bytes).
  void write_impl(const char *ptr, size_t size) override;
  uint64_t current_pos() const override;
  void pwrite_impl(const char *ptr, size_t size, uint64_t offset) override;

private:
  /// So RCRef can access protected constructors.
  friend class RCRef<WriteableBuffer>;

  /// Construct a WriteableBuffer from a mapped file region, and make sure to
  /// set to unbuffered.
  WriteableBuffer(const std::filesystem::path &path,
                  llvm::sys::fs::mapped_file_region &&mapped)
      : Buffer(path, std::move(mapped)) {
    SetUnbuffered();
  }
};
} // namespace M

#endif // SUPPORT_BUFFER_H
