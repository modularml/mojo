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
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;

TEST(BufferTest, RefCountingWorks) {
  auto buffer = Buffer::get("hello");
  EXPECT_TRUE(buffer->getNumReferences() == 1);
  // Move the buffer into a new one.
  auto buffer2 = std::move(buffer);
  EXPECT_TRUE(buffer2->getNumReferences() == 1);
  EXPECT_TRUE(buffer2->getBuffer() == "hello");

  {
    auto buffer3 = buffer2.copy();
    // These two should have the exact same data pointers.
    EXPECT_TRUE(buffer2->getBufferStart() == buffer3->getBufferStart());
    // There should now be 2 references.
    EXPECT_TRUE(buffer2->getNumReferences() == 2);
  }
  // And the data should still not have been freed.
  EXPECT_TRUE(buffer2->getBuffer() == "hello");
  EXPECT_TRUE(buffer2->getNumReferences() == 1);
  EXPECT_FALSE(buffer2->getFilePath()) << "buffer not backed by file";
}

TEST(BufferTest, TestWrite) {
  auto buffer = WriteableBuffer::get();
  *buffer << "hello";
  auto contents =
      StringRef(buffer->getBuffer().data(), buffer->getBuffer().size());
  EXPECT_TRUE(contents == "hello") << "Actually had: " << std::string(contents);

  auto buffer2 = std::move(buffer);
  contents =
      StringRef(buffer2->getBuffer().data(), buffer2->getBuffer().size());
  EXPECT_TRUE(contents == "hello") << "Actually had: " << std::string(contents);

  auto buffer3 = buffer2.copy();
  EXPECT_TRUE(buffer3->getBufferStart() == buffer2->getBufferStart());

  const char *data = "writetest";
  int initialSize = strlen(data);
  auto buffer4 = WriteableBuffer::get(initialSize);
  memcpy(buffer4->getBufferStart(), data, initialSize);
  *buffer4 << "hello";
  contents = StringRef(buffer4->getBufferStart(), initialSize);
  EXPECT_TRUE(contents == data) << "Actually had: " << std::string(contents);
  contents = StringRef(buffer4->getBufferStart() + initialSize,
                       buffer4->getBufferSize() - initialSize);
  EXPECT_TRUE(contents == "hello") << "Actually had: " << std::string(contents);
}

TEST(BufferTest, TestReadPWriteFile) {
  std::error_code ec;
  std::filesystem::path tmpFilePath = std::filesystem::temp_directory_path(ec);
  ASSERT_FALSE(ec) << ec.message();
  tmpFilePath /= "tmpFile";
  auto writeOr =
      WriteableBuffer::getFile(tmpFilePath, /*size=*/5, /*offset=*/0);
  EXPECT_FALSE(writeOr.isError()) << writeOr.getError();
  WriteableBufferRef write = std::move(*writeOr);
  // pwrite because we want to write to a particular offset.
  char hello[] = "hello";
  write->pwrite(hello, 5, 0);

  auto wrongBufferOr = Buffer::getFile("aSillyNamedTempFileThatShouldNotExist");
  EXPECT_TRUE(wrongBufferOr.isError()) << "No such file should exist...";

  auto rightBufferOr = Buffer::getFile(tmpFilePath);
  EXPECT_FALSE(rightBufferOr.isError()) << rightBufferOr.getError();
  EXPECT_TRUE((*rightBufferOr)->getBuffer() == "hello");

  // Clean up the file.
  std::filesystem::remove(tmpFilePath, ec);
  ASSERT_FALSE(ec) << ec.message();
}

TEST(BufferTest, TestReadWriteFile) {
  std::error_code ec;
  std::filesystem::path tmpFilePath = std::filesystem::temp_directory_path(ec);
  ASSERT_FALSE(ec) << ec.message();
  tmpFilePath /= "tmpFile";
  auto writeOr =
      WriteableBuffer::getFile(tmpFilePath, /*size=*/5, /*offset=*/0);
  EXPECT_FALSE(writeOr.isError()) << writeOr.getError();
  WriteableBufferRef write = std::move(*writeOr);
  // We want to write to the start of the file, it'll start at offset 0.
  char hello[] = "hello";
  write->write(hello, 5);

  auto wrongBufferOr = Buffer::getFile("aSillyNamedTempFileThatShouldNotExist");
  EXPECT_TRUE(wrongBufferOr.isError()) << "No such file should exist...";

  auto rightBufferOr = Buffer::getFile(tmpFilePath);
  EXPECT_FALSE(rightBufferOr.isError()) << rightBufferOr.getError();
  BufferRef rightBuffer = rightBufferOr.takeValue();
  EXPECT_TRUE(rightBuffer->getBuffer() == "hello");
  auto retrievedPath = rightBuffer->getFilePath();
  EXPECT_TRUE(retrievedPath) << "buffer backed by file";
  EXPECT_EQ(retrievedPath, tmpFilePath);

  // Indirect read through memory buffer.
  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(tmpFilePath.string(), {}, &errorMsg);
  ASSERT_TRUE(errorMsg.empty()) << errorMsg;
  BufferRef bufferFromMemoryBuffer = Buffer::take(std::move(file));
  EXPECT_TRUE(bufferFromMemoryBuffer->getBuffer() == "hello");
  retrievedPath = bufferFromMemoryBuffer->getFilePath();
  EXPECT_TRUE(retrievedPath) << "buffer backed by file";
  EXPECT_EQ(retrievedPath, tmpFilePath);

  // Clean up the file.
  std::filesystem::remove(tmpFilePath, ec);
}

TEST(BufferTest, AlignmentWorks) {
  auto buffer = WriteableBuffer::get(32, 1024);
  // Just make sure the memory buffer is aligned like we expect it to be.
  EXPECT_TRUE(((uintptr_t)buffer->getBufferStart() & 1023) == 0);
  EXPECT_TRUE(buffer->getBufferSize() == 32);
  std::string originalContents(buffer->getBufferStart(),
                               buffer->getBufferEnd());

  // Resize the buffer now.
  buffer->write("hello", 5);
  // Make sure it's still aligned correctly.
  EXPECT_TRUE(((uintptr_t)buffer->getBufferStart() & 1023) == 0);
  // And ensure we still have all the data we put in.
  originalContents += "hello";
  EXPECT_TRUE(((Buffer &)*buffer).getBuffer() == originalContents);
}

TEST(BufferTest, MemoryBufferConversion) {
  // Test that we can convert from a MemoryBuffer to a Buffer, and that after
  // the `llvm::MemoryBuffer` goes out of scope, the data is still there.
  auto buffer = Buffer::get("hello");
  {
    std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
        llvm::MemoryBuffer::getMemBuffer("goodbye");
    EXPECT_EQ(memoryBuffer->getBufferKind(),
              llvm::MemoryBuffer::MemoryBuffer_Malloc);

    buffer = Buffer::take(std::move(memoryBuffer));
    EXPECT_TRUE(memoryBuffer == nullptr);
  }
  EXPECT_TRUE(buffer->getBuffer() == "goodbye")
      << "Actually had: " << std::string(buffer->getBuffer());

#ifdef MODULAR_DEBUG
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer;
  ASSERT_DEATH_IF_SUPPORTED(Buffer::take(std::move(memoryBuffer)),
                            "expected a non-null memory buffer");
#endif // MODULAR_DEBUG
}

TEST(BufferTest, AliasedBuffer) {
  BufferRef alias;
  {
    auto buffer = Buffer::get("hello, world");
    alias = Buffer::getAlias(buffer.copy(), 0, 5);
  }
  // Check that the alias extends the original buffer's lifetime.
  EXPECT_EQ(alias->getBuffer(), "hello");
  auto aliasOfAlias = Buffer::getAlias(alias.copy());
  EXPECT_EQ(aliasOfAlias->getBuffer(), "hello");

  ASSERT_DEATH_IF_SUPPORTED(Buffer::getAlias(alias.copy(), 12, 348),
                            "too many bytes");
}
