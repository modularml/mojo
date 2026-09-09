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

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Cache/BlobCache.h"
#include "Cache/Support/Keys.h"
#include "Init/Init.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/Buffer.h"
#include "Support/CommandLine.h"
#include "Support/CommonCLOptions.h"
#include "Support/Context.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/URI.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace M;
using namespace AsyncRT;
using namespace Cache;

namespace {
/// This provides a zero-copy binary blob cache key struct. The idea is that it
/// should operate directly on BufferRef because that's what we use in
/// this tool, and it should be simple to read/write.
using BinaryBlobCacheKey = Keys::VariantTypeKey<BufferRef, StringRef>;

/// Provides the CLOptions for this tool.
class CLOptions : public OptionsBase {
private:
  CLOptionsBase parser;

public:
  CLOptions(int argc, char **argv, bool skipInitLLVM = false)
      : parser(argc, argv, *this, skipInitLLVM) {}

  /// Specify the input file or cached object.
  M::cl::MOpt<std::string> input{"input",
                                 cl::desc("<input file or CAS reference>")};

  M::cl::MOpt<std::string> key{
      "key",
      cl::desc("Explicitly Specify key. In this case instead of binary hash, "
               "this key will be used for adding object to cache.")};

  M::cl::MOpt<std::string> backendVersion{
      "backend-version", cl::desc("Set the version for the local backend."),
      cl::init("")};

  /// Specify the target path for the CAS backend.
  M::cl::MOpt<std::string> fsPath{
      "base-dir",
      cl::desc("URI for the CAS storage. Defaults to a temporary directory."),
      llvm::cl::init("")};

  M::cl::MOpt<std::string> outFile{
      "o", cl::desc("File path to use for program outputs."),
      llvm::cl::init("-")};

  M::cl::MOpt<bool> append{"append", cl::desc("Append to the output file."),
                           cl::init(false)};

  M::cl::MOpt<bool> outputHex{
      "output-hex",
      cl::desc("write the output hash in hex, rather than base64"),
      cl::init(false)};

  /// Get the backend path for the CAS. Defaults to a temporary directory.
  ErrorOr<URI> getBackendPath() const;
};
} // namespace

/// Attempt to decode a hash from the encoded string. If the incoming string is
/// hex or base64 this will return the raw bytes, otherwise it will return
/// std::nullopt.
static std::optional<std::string> decodeHash(StringRef encoded) {
  std::string hash;
  if (!llvm::tryGetFromHex(encoded, hash)) {
    std::vector<char> ref;
    ref.reserve(encoded.size());
    if (auto err = llvm::decodeBase64(encoded, ref)) {
      (void)llvm::toString(
          std::move(err)); // Consume the error without doing anything.
      return std::nullopt;
    }

    hash = std::string(ref.begin(), ref.end());
    hash.shrink_to_fit();
    return hash;
  }
  return hash;
}

//===----------------------------------------------------------------------===//
// CLOptions::getFsPath
//===----------------------------------------------------------------------===//

ErrorOr<URI> CLOptions::getBackendPath() const {
  auto uriOr = URI::parse(fsPath.getValue());
  if (uriOr.isError())
    return uriOr.takeError();

  // If the URI isn't a file:// URI, then just return it directly.
  if (uriOr->getScheme() != "file")
    return std::move(*uriOr);

  // Get the path provided to the command line if it exists.
  std::filesystem::path out(uriOr->getPath().str());
  std::error_code ec;
  if (!out.empty()) {
    out = std::filesystem::absolute(out, ec);
    if (ec) {
      reportError(ec.message());
      exit(1);
    }
    return out;
  }

  // Default to some temp directory.
  out = std::filesystem::temp_directory_path(ec) / "modular" / "cache";
  if (ec) {
    reportError(ec.message());
    exit(1);
  }
  llvm::errs() << "[WARNING] Using temporary file path at " << out.string()
               << " for CAS filesystem base path.\n";
  return out;
}

static AsyncValueRef<std::string>
putObjectsIntoCache(BinaryBlobCacheKey::KeyTy key, BufferRef value,
                    StringRef input,
                    RCRef<BlobCache<BinaryBlobCacheKey>> &cache,
                    AsyncRT::CPUDevice &cpuDevice, bool useHex) {

  AsyncValueRef<std::string> insert =
      cache->insert(cpuDevice, std::move(key), std::move(value));
  auto outCh = AsyncValueRef<std::string>::allocate(cpuDevice);
  std::move(insert).andThenSync([outCh = outCh.copy(),
                                 input = Buffer::get(input), useHex](
                                    AsyncValueRef<std::string> &&hash) mutable {
    // If we have an error, report it.
    if (hash.isError())
      return std::move(outCh).setToError(hash.takeDiagnostic());

    // Otherwise, emplace the string so that we can report it to the
    // user.
    if (useHex)
      return std::move(outCh).emplace(llvm::toHex(*hash, /*LowerCase=*/true));

    std::move(outCh).emplace(llvm::encodeBase64(*hash));
  });
  return outCh;
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Create our context.
  ErrorOr<ContextRef> ctxOr =
      Init::createContext("cache-mgr", Init::Options().withCPUDeviceOptions());
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  AsyncRT::CPUDevice &cpuDevice = *(*ctxOr)->get<AsyncRT::CPUDevice>();

  auto backendPathOr = clOptions.getBackendPath();
  if (backendPathOr.isError())
    return clOptions.reportError(backendPathOr.getError());

  auto backendChainOr =
      getDefaultBackendChain(*backendPathOr, clOptions.backendVersion);
  if (backendChainOr.isError())
    return clOptions.reportError(backendChainOr.getError());

  auto cache =
      RCRef<BlobCache<BinaryBlobCacheKey>>::create(std::move(*backendChainOr));

  // The result of the operation will always be either a buffer (raw contents
  // from a GET), string (hash from a PUT), or an error.
  SmartVariant<BufferRef, std::string, EncodedDiagnostic> operationResult;

  // If we were able to decode the hash, then it's a GET - otherwise it's a
  // PUT.
  auto hash = decodeHash(clOptions.input);
  std::string key = clOptions.key;
  if (hash || !key.empty()) {
    // Attempt to find the value in the cache, if it exists, write it out.
    // If the key is specified, use it directly.
    std::string keyToFind =
        key.empty() ? BinaryBlobCacheKey::hashKey(*hash) : key;
    auto result = cache->find(cpuDevice, keyToFind);
    auto outCh = AsyncValueRef<BufferRef>::allocate(cpuDevice);
    std::move(result).andThenSync(
        [outCh = outCh.copy(), input = Buffer::get(*hash)](
            AsyncValueRef<std::optional<BufferRef>> &&found) mutable {
          // If there was an error, produce that diagnostic.
          if (found.isError())
            return std::move(outCh).setToError(found.takeDiagnostic());

          // No value, emit an error.
          if (!found->has_value()) {
            return std::move(outCh).setToError(
                UnknownLocationDecoder::getDiagnostic(
                    Twine(llvm::encodeBase64(input->getBuffer())) +
                    ": value not found in the cache"));
          }

          // Has a value, write the value.
          BufferRef buf = std::move(**found);
          std::move(outCh).emplace(std::move(buf));
        });
    await(outCh);
    if (outCh.isError()) {
      operationResult.getUnderlyingStorage().emplace<EncodedDiagnostic>(
          outCh.takeDiagnostic());
    } else {
      operationResult.getUnderlyingStorage().emplace<BufferRef>(
          std::move(*outCh));
    }
  } else {
    auto bufOr = Buffer::getFile(clOptions.input.getValue());
    if (bufOr.isError())
      return clOptions.reportError(bufOr.getError());
    std::string keyToWrite =
        key.empty() ? BinaryBlobCacheKey::hashKey((*bufOr).copy()) : key;
    AsyncValueRef<std::string> outCh =
        putObjectsIntoCache(keyToWrite, (*bufOr).copy(), clOptions.input, cache,
                            cpuDevice, clOptions.outputHex);
    await(outCh);
    if (outCh.isError()) {
      operationResult.getUnderlyingStorage().emplace<EncodedDiagnostic>(
          outCh.takeDiagnostic());
    } else {
      operationResult.getUnderlyingStorage().emplace<std::string>(
          std::move(*outCh));
    }
  }

  std::error_code ec;
  llvm::ToolOutputFile outFile(clOptions.outFile, ec,
                               clOptions.append ? llvm::sys::fs::OF_Append
                                                : llvm::sys::fs::OF_None);
  if (ec)
    return clOptions.reportError(ec.message());

  // Report any errors we might have.
  if (isa<EncodedDiagnostic>(operationResult)) {
    auto &diag = cast<EncodedDiagnostic>(operationResult);
    return clOptions.reportError(diag.getMessage().get());
  }

  if (isa<std::string>(operationResult)) {
    auto &str = cast<std::string>(operationResult);
    outFile.os() << str;
  } else if (isa<BufferRef>(operationResult)) {
    auto &buf = cast<BufferRef>(operationResult);
    outFile.os() << buf->getBuffer();
  }
  if (clOptions.append)
    outFile.os() << ";";

  // Keep the output file.
  outFile.keep();
  return 0;
}
