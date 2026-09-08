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

#include "RPCClient.h"
#include "Mojo/Support/Configuration.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/Process.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include <filesystem>
#include <future>
#include <iostream>
#include <set>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
typedef int socklen_t;
#else
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using SOCKET = int;
#endif

using namespace M;

namespace json = llvm::json;

StringRef protocolSeparator = "\n----\n";

namespace {
struct ResponseConnect {
  std::string kind;
  bool success;
  std::optional<std::string> message;
  int64_t pid;
  int64_t lastTimeSeenActiveInSecs;
  std::string name;
};

struct RequestConnect {
  std::string kind = "connect";
};

struct ResponseDebug {
  std::string kind;
  bool success;
  std::optional<std::string> message;
};
} // namespace

namespace llvm::json {
bool fromJSON(const json::Value &value, ResponseConnect &response, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("success", response.success) &&
         o.map("kind", response.kind) && o.map("pid", response.pid) &&
         o.mapOptional("message", response.message) &&
         o.map("lastTimeSeenActiveInSecs", response.lastTimeSeenActiveInSecs) &&
         o.map("name", response.name);
}

bool fromJSON(const json::Value &value, ResponseDebug &response, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("success", response.success) &&
         o.map("kind", response.kind) &&
         o.mapOptional("message", response.message);
}

llvm::json::Value toJSON(const RequestConnect &request) {
  return llvm::json::Object{{"kind", request.kind}};
}
} // namespace llvm::json

namespace {
/// Move-only wrapper around an active socket and the information of
/// the connected RPC Server.
struct Connection {
  SOCKET sockfd;
  int port;
  ResponseConnect serverInfo;

  Connection(SOCKET sockfd, int port, ResponseConnect serverInfo)
      : sockfd(sockfd), port(port), serverInfo(std::move(serverInfo)) {}

  Connection(const Connection &) = delete;
  Connection &operator=(const Connection &) = delete;

  Connection(Connection &&o)
      : sockfd(o.sockfd), port(o.port), serverInfo(std::move(o.serverInfo)) {
    o.sockfd = -1;
  }

  /// Operator used to sort connections for display.
  bool operator<(const Connection &o) const {
    int64_t diff = serverInfo.lastTimeSeenActiveInSecs -
                   o.serverInfo.lastTimeSeenActiveInSecs;
    if (diff != 0)
      return diff < 0;
    if (serverInfo.name != o.serverInfo.name)
      return serverInfo.name.length() > o.serverInfo.name.length();
    return port < o.port;
  }

  ~Connection() {
    if (sockfd == -1)
      return;
#if defined(_WIN32)
    closesocket(sockfd);
#else
    close(sockfd);
#endif
  }
};
} // namespace

/// Create an object with the common fields of debug configurations.
static ErrorOr<json::Object>
createBasicDebugConfiguration(bool useCudaGdb,
                              ArrayRef<std::string> initCommands) {
  ErrorOr<std::filesystem::path> modularHome =
      Config::getModularConfigFolderPath();
  if (failed(modularHome))
    return modularHome.takeError();

  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr))
    return Error(Twine("failed to parse 'modular.cfg': ") +
                 configOr.getError());

  // For the cuda-gdb case, we use a custom mojo-cuda-gdb type that we massage
  // into the cuda-gdb type in the RPC server.
  json::Object payload{
      {"modularHomePath", modularHome->string()},
      {"modularConfigMojoSection", configOr->getMojoConfigSection().str()},
      {"mojoDriverPath", configOr->getDriverPath().str()},
      {"debuggerRoot", std::filesystem::current_path().string()},
      {"type", useCudaGdb ? "mojo-cuda-gdb" : "mojo-lldb"}};

  if (!initCommands.empty()) {
    payload.insert({"initCommands", json::Array(initCommands)});
  }

  return payload;
}

template <typename TResponse>
static ErrorOr<TResponse> doSendRequest(SOCKET sockfd, StringRef payloadStr,
                                        raw_ostream &extraLogStream, int port) {
  ssize_t sentBytes = send(sockfd, payloadStr.data(), payloadStr.size(), 0);
  if (sentBytes < 0)
    return Error(Twine("can't send data to the RPC debug server: ") +
                 strerror(errno));

  std::string rawResponse;
  while (true) {
    char buff[256];
    ssize_t recvBytes = 0;
    recvBytes = recv(sockfd, buff, sizeof(buff) - 1, 0);
    if (recvBytes < 0)
      return Error(Twine("can't receive response from the RPC debug server: ") +
                   strerror(errno));

    if (recvBytes == 0)
      break;

    rawResponse.append(buff, recvBytes);
    if (StringRef(rawResponse).ends_with(protocolSeparator))
      break;
  }

  extraLogStream << "[port=" << port << "] Got response:\n"
                 << rawResponse << "\n";

  StringRef response(rawResponse);
  if (!response.consume_back(protocolSeparator)) {
    return Error(Twine("response from the RPC server doesn't follow the "
                       "expected protocol: ") +
                 rawResponse);
  }

  llvm::Expected<TResponse> parsedResponse =
      llvm::json::parse<TResponse>(response);
  if (!parsedResponse) {
    llvm::consumeError(parsedResponse.takeError());
    return Error(Twine("invalid RPC server response: ") + response);
  }

  if (parsedResponse->success)
    return *parsedResponse;
  if (parsedResponse->message)
    return Error(Twine("RPC Server response:\n", *parsedResponse->message));
  return Error("couldn't get a valid response from the RPC server, see "
               "https://mojolang.org/docs/tools/debugging for possible fixes");
}

ErrorOr<Connection> tryToConnectToServer(int port,
                                         raw_ostream &extraLogStream) {
  SOCKET sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    return Error(llvm::formatv("can't open socket to communicate with the RPC "
                               "debug server with port {0}: {1}",
                               port, strerror(errno)));
  }

  struct sockaddr_in serverAddress;
  memset((char *)&serverAddress, 0, sizeof(serverAddress));
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  serverAddress.sin_port = htons(port);

  if (connect(sockfd, (struct sockaddr *)&serverAddress,
              sizeof(serverAddress)) < 0) {
    return Error(llvm::formatv(
        "can't connect to the RPC server with port {0}: {1}. "
        "See https://mojolang.org/docs/tools/debugging for possible fixes.",
        port, strerror(errno)));
  }

  extraLogStream << "[port=" << port << "] TCP connection successful\n";

  // We create a dangling thread to easily handle timeouts. The thread will
  // die anyway as soon as we close the socket.
  std::string request = llvm::formatv(
      "{0:2}{1}", json::toJSON(RequestConnect()), protocolSeparator);
  extraLogStream << "[port=" << port << "] Will send:\n" << request << "\n";
  auto future = new std::future<ErrorOr<ResponseConnect>>(
      std::async(doSendRequest<ResponseConnect>, sockfd, request,
                 std::ref(extraLogStream), port));
  auto timeout = std::chrono::seconds(5);
  if (future->wait_for(timeout) == std::future_status::timeout)
    return Error("timeout when waiting for the `connect` response");

  extraLogStream << "[port=" << port
                 << "] Server info successfully obtained.\n";
  ErrorOr<ResponseConnect> response = future->get();
  delete future;
  if (failed(response))
    return response.takeError();
  return Connection(sockfd, port, *response);
}

/// Send the given payload to the RPC server at one of the specified ports. If
/// `dryRun` is specified, then the payload is printed to the standard output
/// instead.
static ErrorOrSuccess invokeRPC(bool dryRun, ArrayRef<int> ports,
                                json::Object debugConfiguration) {

  json::Value request = json::Object{
      {"kind", "debug"}, {"debugConfiguration", std::move(debugConfiguration)}};
  std::string requestStr =
      llvm::formatv("{0:2}{1}", request, protocolSeparator);
  if (dryRun) {
    llvm::outs() << "payload: " << requestStr << "\n";
    return success();
  }

  // Create a temporary file to write additional logs.
  ErrorOr<TempFile> logFileOrErr =
      TempFile::create("mojo-debug-rpc-logs-%%%%%%.txt");
  if (failed(logFileOrErr))
    return Error(std::string("could not create file for additional logs: ") +
                 logFileOrErr.getError());
  logFileOrErr->keep();
  llvm::raw_fd_ostream extraLogStream(logFileOrErr->getFD(),
                                      /*shouldClose=*/true,
                                      /*unbuffered=*/true);
  llvm::errs() << "[INFO] Additional logs can be found in "
               << logFileOrErr->getPath()
               << ". Please include them when reporting bugs.\n\n";
  extraLogStream << "Server-side logs can be found in the `Mojo` section of "
                    "the `Output` tab of each VSCode Window.\n\n";

  std::set<Connection> connections;
  std::vector<std::pair<int, std::future<ErrorOr<Connection>>>>
      connectionFutures;
  for (int port : ports) {
    extraLogStream << "[port=" << port
                   << "] Will try to connect to its RPC server\n";
    connectionFutures.emplace_back(
        port, std::async(tryToConnectToServer, port, std::ref(extraLogStream)));
  }
  for (auto &connectionFuture : connectionFutures) {
    ErrorOr<Connection> connection = connectionFuture.second.get();
    if (failed(connection))
      extraLogStream << "[port=" << connectionFuture.first
                     << "] Error: " << connection.takeError() << "\n";
    else
      connections.insert(std::move(*connection));
  }

  if (connections.empty())
    return Error("couldn't connect to any VSCode windows. You might need to "
                 "restart the IDE or file a bug.");

  llvm::outs() << "Active VS Code windows:\n";
  for (auto [idx, conn] : llvm::enumerate(connections)) {
    std::string index = std::to_string(idx);
    llvm::outs() << index << ": "
                 << llvm::formatv(
                        "{0}\n{4}  Last activity identified {1} seconds "
                        "ago, pid={2}, port={3}\n",
                        conn.serverInfo.name,
                        conn.serverInfo.lastTimeSeenActiveInSecs,
                        conn.serverInfo.pid, conn.port,
                        std::string(index.size(), ' '));
  }

  int64_t index = 0;
  if (connections.size() == 1) {
    llvm::outs()
        << "\nOnly one VS Code window was found. The debug session will "
           "be launched in this window automatically.\n\n";
  } else {
    llvm::outs() << "\nMultiple VS Code windows found. Press <enter> to select "
                    "the window with index 0 or provide the index of the "
                    "window to use:\n";
    std::string rawInput;
    std::getline(std::cin, rawInput);
    StringRef input(rawInput);
    input = input.trim();
    if (!input.empty()) {
      if (input.consumeInteger(10, index)) {
        return Error("invalid input");
      }
    }
  }

  auto it = connections.begin();
  std::advance(it, index);

  extraLogStream << "[port=" << it->port << "] Will send debug request:\n"
                 << requestStr << "\n";

  ErrorOr<ResponseDebug> response = doSendRequest<ResponseDebug>(
      it->sockfd, requestStr, extraLogStream, it->port);
  if (failed(response))
    return response.takeError();
  return success();
}

ErrorOrSuccess M::invokeAttachRPC(bool dryRun, bool useCudaGdb,
                                  bool breakOnLaunch, ArrayRef<int> rpcPorts,
                                  const std::optional<StringRef> &pid,
                                  const std::optional<StringRef> &processName,
                                  ArrayRef<std::string> initCommands) {
  ErrorOr<json::Object> payload =
      createBasicDebugConfiguration(useCudaGdb, initCommands);
  if (failed(payload))
    return payload.takeError();
  payload->insert({"request", "attach"});
  if (pid)
    payload->insert({"pid", *pid});
  if (processName)
    payload->insert({"program", *processName});
  if (breakOnLaunch)
    payload->insert({"breakOnLaunch", true});
  return invokeRPC(dryRun, rpcPorts, *payload);
}

ErrorOrSuccess M::invokeLaunchRPC(bool dryRun, bool useCudaGdb,
                                  bool breakOnLaunch, ArrayRef<int> rpcPorts,
                                  StringRef target,
                                  ArrayRef<std::string> runArgs,
                                  StringRef rpcTerminal, bool stopOnEntry,
                                  ArrayRef<std::string> initCommands) {
  ErrorOr<json::Object> payload =
      createBasicDebugConfiguration(useCudaGdb, initCommands);
  if (failed(payload))
    return payload.takeError();

  std::error_code ec;
  std::filesystem::path fullTarget =
      std::filesystem::absolute(target.str(), ec);
  if (ec)
    return Error("failed to get absolute path to the target '" + target +
                 "': " + ec.message());

  std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (ec)
    return Error("failed to get the current working path: " + ec.message());

  if (useCudaGdb) {
    payload->insert({"name", "Mojo debug with cuda-gdb"});
    if (breakOnLaunch)
      payload->insert({"breakOnLaunch", true});
  }
  payload->insert({"program", fullTarget.string()});
  payload->insert({"request", "launch"});
  payload->insert({"cwd", cwd.string()});
  payload->insert({"debuggerRoot", cwd.string()});

  if (useCudaGdb)
    payload->insert({"stopAtEntry", stopOnEntry});
  else
    payload->insert({"stopOnEntry", stopOnEntry});

  json::Array env;
  for (StringRef entry : getEnv())
    env.push_back(entry);
  payload->insert({"env", std::move(env)});
  payload->insert({"args", json::Array{runArgs}});
  payload->insert({"runInTerminal", rpcTerminal == "dedicated"});

  return invokeRPC(dryRun, rpcPorts, *payload);
}
