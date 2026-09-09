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
// GPU Device API tests - requires GPU hardware
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "max/c/common.h"
#include "max/c/device.h"
#include "max/c/tensor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace M;

TEST_F(GPUTest, CreateGPUDevice) {
  EXPECT_EQ(M_getDeviceType(gpuDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(gpuDevice), 0);
  EXPECT_EQ(M_isHostDevice(gpuDevice), 0);
  EXPECT_STREQ(M_getDeviceLabel(gpuDevice), "gpu");
}

TEST_F(GPUTest, SynchronizeGPUDevice) {
  M_synchronizeDevice(gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to synchronize GPU device");
}

// Test for borrowing a pointer that already lives on the
// device must alias it in place (zero-copy), not allocate a fresh device
// buffer and copy into it. If it copies, in-place mutation of the borrowed
// buffer is lost because the model runs against the private copy.
TEST_F(GPUTest, BorrowDevicePointerAliasesInPlace) {
  std::vector<float> hostData = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[] = {4};

  // Stage data onto the GPU and capture its device pointer.
  M_TensorSpec *hostSpec = M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_AsyncTensorMap *hostMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(hostMap, hostData.data(), hostSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow host tensor");
  M_AsyncTensor *hostTensor = M_getTensorByNameFrom(hostMap, "input0", status);
  M_AsyncTensor *deviceTensor =
      M_copyTensorToDevice(hostTensor, gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to GPU");
  const void *devicePtr = M_getTensorData(deviceTensor);

  // Borrowing that device pointer into a GPU spec must reuse it in place
  // rather than allocate an alignment-padded copy at a different address.
  M_TensorSpec *gpuSpec =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_AsyncTensorMap *gpuMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(gpuMap, (void *)(uintptr_t)devicePtr, gpuSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow device tensor");
  M_AsyncTensor *borrowed = M_getTensorByNameFrom(gpuMap, "input0", status);
  EXPECT_SUCCESS(status, "Failed to get borrowed device tensor");
  EXPECT_EQ(M_getTensorData(borrowed), devicePtr);

  // Free the borrowing map first so its no-op destructor runs before the
  // tensor that actually owns the device allocation.
  M_freeAsyncTensorMap(gpuMap);
  M_freeTensorSpec(gpuSpec);
  M_freeTensor(deviceTensor);
  M_freeTensor(hostTensor);
  M_freeAsyncTensorMap(hostMap);
  M_freeTensorSpec(hostSpec);
}

// End-to-end regression test for when a model with a `BufferType`
// input mutates that input in place and the caller borrowed a device pointer
// for it, the mutation must be visible through the caller's pointer.
TEST_F(GPUTest, BorrowDevicePointerReflectsInPlaceMutation) {
  const char *mefPath = getenv("MUTATE_MEF_PATH");
  ASSERT_NE(mefPath, nullptr) << "MUTATE_MEF_PATH not set";

  // Compile the GPU mutate model: buffer[...] = buffer[...] * 2 + 1.
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath);
  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile mutate model");
  M_AsyncModel *model = M_initModel(context, compiledModel, nullptr, status);
  EXPECT_SUCCESS(status, "Failed to initialize mutate model");

  // Stage known data onto the GPU and capture its device pointer. This stands
  // in for the caller's device-resident buffer that they expect mutated.
  std::vector<float> original = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int64_t shape[] = {10};
  M_TensorSpec *hostSpec = M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_AsyncTensorMap *hostMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(hostMap, original.data(), hostSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow host tensor");
  M_AsyncTensor *hostTensor = M_getTensorByNameFrom(hostMap, "input0", status);
  M_AsyncTensor *deviceTensor =
      M_copyTensorToDevice(hostTensor, gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to GPU");
  void *devicePtr = const_cast<void *>(M_getTensorData(deviceTensor));

  // Borrow that device pointer (zero-copy) as the model's `BufferType` input.
  M_TensorSpec *gpuSpec =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, devicePtr, gpuSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow device tensor");

  // Run the model; it mutates the borrowed buffer in place.
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute mutate model");

  // Copy the *same* device allocation back to host and confirm the mutation
  // landed in it. If the borrow had copied instead of aliased, this allocation
  // would still hold the original 1..10 values.
  M_AsyncTensor *result = M_copyTensorToDevice(deviceTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to copy result back to host");
  const float *resultData = static_cast<const float *>(M_getTensorData(result));
  for (size_t i = 0; i < original.size(); ++i)
    EXPECT_FLOAT_EQ(resultData[i], original[i] * 2.0f + 1.0f)
        << "In-place mutation not reflected at index " << i;

  // Free the borrowing map first so its no-op destructor runs before the
  // tensor that owns the device allocation.
  M_freeTensor(result);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(gpuSpec);
  M_freeTensor(deviceTensor);
  M_freeTensor(hostTensor);
  M_freeAsyncTensorMap(hostMap);
  M_freeTensorSpec(hostSpec);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(GPUTest, MoveTensorToGPU) {
  // Create a tensor on host
  std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
  int64_t shape[] = {2, 4};
  M_TensorSpec *spec = M_newTensorSpec(shape, 2, M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensor *hostTensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get host tensor");
  EXPECT_THAT(hostTensor, ::testing::NotNull());

  // Verify tensor is on host
  const M_Device *tensorDevice = M_getTensorDevice(hostTensor);
  EXPECT_THAT(tensorDevice, ::testing::NotNull());
  EXPECT_EQ(M_isHostDevice(tensorDevice), 1);

  // Move tensor to GPU
  M_AsyncTensor *gpuTensor =
      M_copyTensorToDevice(hostTensor, gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to GPU");
  EXPECT_THAT(gpuTensor, ::testing::NotNull());

  // Verify tensor is on GPU
  const M_Device *gpuTensorDevice = M_getTensorDevice(gpuTensor);
  EXPECT_THAT(gpuTensorDevice, ::testing::NotNull());
  EXPECT_EQ(M_isHostDevice(gpuTensorDevice), 0);
  EXPECT_EQ(M_getDeviceType(gpuTensorDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(gpuTensorDevice), M_getDeviceId(gpuDevice));

  M_freeTensor(gpuTensor);
  M_freeTensor(hostTensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, MoveTensorFromGPU) {
  // Create a tensor and move it to GPU first
  std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[] = {4};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 1, M_FLOAT32, "testTensor", gpuDevice);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensor *gpuTensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get gpu tensor");
  M_Device *gpuTensorDevice = M_getTensorDevice(gpuTensor);
  EXPECT_EQ(M_isHostDevice(gpuTensorDevice), 0);
  EXPECT_EQ(M_getDeviceType(gpuTensorDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(gpuTensorDevice), M_getDeviceId(gpuDevice));
  M_freeDevice(gpuTensorDevice);

  M_AsyncTensor *hostTensor = M_copyTensorToDevice(gpuTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor back to host");
  EXPECT_THAT(hostTensor, ::testing::NotNull());

  M_Device *device = M_getTensorDevice(hostTensor);
  EXPECT_THAT(device, ::testing::NotNull());
  EXPECT_EQ(M_isHostDevice(device), 1);
  M_freeDevice(device);

  M_freeTensor(hostTensor);
  M_freeTensor(gpuTensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, RoundTripTensorData) {
  // Create a tensor with known data
  std::vector<float> originalData = {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f};
  int64_t shape[] = {2, 4};
  M_TensorSpec *spec = M_newTensorSpec(shape, 2, M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, originalData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensor *hostTensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get host tensor");

  M_AsyncTensor *gpuTensor =
      M_copyTensorToDevice(hostTensor, gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to GPU");

  M_AsyncTensor *roundTripped = M_copyTensorToDevice(gpuTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor back to host");

  EXPECT_EQ(M_getTensorNumElements(roundTripped), originalData.size());
  const float *resultData =
      static_cast<const float *>(M_getTensorData(roundTripped));
  EXPECT_THAT(resultData, ::testing::NotNull());

  for (size_t i = 0; i < originalData.size(); ++i)
    EXPECT_FLOAT_EQ(resultData[i], originalData[i])
        << "Data mismatch at index " << i;

  M_freeTensor(roundTripped);
  M_freeTensor(gpuTensor);
  M_freeTensor(hostTensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, TensorSpecWithGPUDevice) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 3, M_FLOAT32, "gpuTensor", gpuDevice);
  EXPECT_THAT(spec, ::testing::NotNull());

  EXPECT_EQ(M_getDeviceTypeFromSpec(spec), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceIdFromSpec(spec), M_getDeviceId(gpuDevice));

  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, BorrowTensorIntoWithGPUSpec) {
  // Create a tensor spec on GPU device
  std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  int64_t shape[] = {2, 5};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 2, M_FLOAT32, "gpuTensor", gpuDevice);
  EXPECT_THAT(spec, ::testing::NotNull());

  // Verify the spec has GPU device info
  EXPECT_EQ(M_getDeviceTypeFromSpec(spec), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceIdFromSpec(spec), 0);

  // Borrow tensor into the tensor map with GPU spec
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(tensorMap, inputData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor with GPU spec");

  // Retrieve the tensor and verify its properties
  M_AsyncTensor *tensor = M_getTensorByNameFrom(tensorMap, "gpuTensor", status);
  EXPECT_SUCCESS(status, "Failed to get GPU tensor by name");
  EXPECT_THAT(tensor, ::testing::NotNull());

  EXPECT_EQ(M_getTensorNumElements(tensor), 10u);
  EXPECT_EQ(M_getTensorType(tensor), M_FLOAT32);

  // Verify the tensor is associated with a GPU device
  M_Device *tensorDevice = M_getTensorDevice(tensor);
  EXPECT_THAT(tensorDevice, ::testing::NotNull());
  EXPECT_EQ(M_getDeviceType(tensorDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(tensorDevice), 0);
  EXPECT_EQ(M_isHostDevice(tensorDevice), 0);

  M_freeDevice(tensorDevice);
  M_freeTensor(tensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, ExecuteModelWithGPUTensors) {
  // This test requires the MEF file from the MEFTest
  const char *mefPath = getenv("TEST_MEF_PATH");

  // Load the MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath);

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  M_AsyncModel *model = M_initModel(context, compiledModel, nullptr, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");
  EXPECT_THAT(model, ::testing::NotNull());

  // Create input tensors on host first
  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", gpuDevice);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to add second input");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to retrieve output tensor");

  EXPECT_EQ(M_getTensorNumElements(outputTensor), 8u);

  M_Device *outputDevice = M_getTensorDevice(outputTensor);
  EXPECT_EQ(M_getDeviceType(outputDevice), M_ACCELERATOR);
  M_freeDevice(outputDevice);

  M_AsyncTensor *outputHostTensor =
      M_copyTensorToDevice(outputTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to transfer result to host");
  const float *outputData =
      static_cast<const float *>(M_getTensorData(outputHostTensor));
  EXPECT_THAT(outputData, ::testing::NotNull());

  for (int i = 0; i < 8; i++)
    EXPECT_FLOAT_EQ(outputData[i], 9.0f);

  M_freeTensor(outputTensor);
  M_freeTensor(outputHostTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(GPUTest, ExecuteModelWithIncorrectDeviceTensors) {
  // This test requires the MEF file from the MEFTest
  const char *mefPath = getenv("TEST_MEF_PATH");

  // Load the MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath);

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  M_AsyncModel *model = M_initModel(context, compiledModel, nullptr, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");
  EXPECT_THAT(model, ::testing::NotNull());

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to add second input");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_FAILURE(status, "input 'input0' on incorrect device");
  EXPECT_THAT(outputs, ::testing::IsNull());

  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

// ===== Device Graph Capture/Replay Tests ===== //

// Helper to load the GPU vector_add MEF and init a model.
static M_AsyncModel *loadVectorAddModel(M_RuntimeContext *context,
                                        M_AsyncCompiledModel **compiledModel,
                                        M_Status *status) {
  const char *mefPath = getenv("TEST_MEF_PATH");
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath);
  *compiledModel = M_compileModelSync(context, &compileConfig, status);
  if (M_isError(status))
    return nullptr;
  M_AsyncModel *model = M_initModel(context, *compiledModel, nullptr, status);
  return model;
}

#define ASSERT_SUCCESS(status, ...)                                            \
  {                                                                            \
    ASSERT_FALSE(M_isError(status))                                            \
        << "" __VA_ARGS__ << ": " << M_getError(status);                       \
  }

TEST_F(GPUTest, CaptureAndReplayModel) {
  M_AsyncCompiledModel *compiledModel = nullptr;
  M_AsyncModel *model = loadVectorAddModel(context, &compiledModel, status);
  ASSERT_SUCCESS(status, "Failed to load model");
  ASSERT_THAT(model, ::testing::NotNull());

  // Create GPU input tensors via borrow + extract.
  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t shape[1] = {8};

  M_TensorSpec *spec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_TensorSpec *spec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", gpuDevice);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(tensorMap, vector1, spec1, status);
  ASSERT_SUCCESS(status, "Failed to borrow input0");
  M_borrowTensorInto(tensorMap, vector2, spec2, status);
  ASSERT_SUCCESS(status, "Failed to borrow input1");

  M_AsyncTensor *input0 = M_getTensorByNameFrom(tensorMap, "input0", status);
  ASSERT_SUCCESS(status, "Failed to get input0");
  M_AsyncTensor *input1 = M_getTensorByNameFrom(tensorMap, "input1", status);
  ASSERT_SUCCESS(status, "Failed to get input1");

  // Capture with graph key 1. Graph capture requires CUDA or HIP -- skip on
  // other accelerators (e.g. Apple GPU on macOS).
  uint64_t graphKey = 1;
  M_AsyncTensor *inputArray[2] = {input0, input1};
  size_t numOutputs = 0;

  M_AsyncTensor **capturedOutputs = M_captureModelSync(
      context, model, &graphKey, 1, inputArray, 2, &numOutputs, status);
  if (M_isError(status)) {
    GTEST_SKIP() << "Graph capture not supported: " << M_getError(status);
  }
  ASSERT_THAT(capturedOutputs, ::testing::NotNull());
  ASSERT_EQ(numOutputs, 1u);

  // Verify captured output: vector1 + vector2 = {9, 9, 9, 9, 9, 9, 9, 9}
  M_AsyncTensor *outputHost =
      M_copyTensorToDevice(capturedOutputs[0], host, status);
  ASSERT_SUCCESS(status, "Failed to copy output to host");
  const float *outputData =
      static_cast<const float *>(M_getTensorData(outputHost));
  for (int i = 0; i < 8; i++)
    EXPECT_FLOAT_EQ(outputData[i], 9.0f);
  M_freeTensor(outputHost);

  // Replay with same inputs and verify outputs are still correct.
  M_replayModelSync(context, model, &graphKey, 1, inputArray, 2, status);
  ASSERT_SUCCESS(status, "Replay failed");

  outputHost = M_copyTensorToDevice(capturedOutputs[0], host, status);
  ASSERT_SUCCESS(status, "Failed to copy output to host after replay");
  outputData = static_cast<const float *>(M_getTensorData(outputHost));
  for (int i = 0; i < 8; i++)
    EXPECT_FLOAT_EQ(outputData[i], 9.0f);
  M_freeTensor(outputHost);

  // Clean up.
  M_freeTensor(capturedOutputs[0]);
  free(capturedOutputs);
  M_freeTensor(input0);
  M_freeTensor(input1);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec1);
  M_freeTensorSpec(spec2);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(GPUTest, ReplayWithoutCapture) {
  M_AsyncCompiledModel *compiledModel = nullptr;
  M_AsyncModel *model = loadVectorAddModel(context, &compiledModel, status);
  ASSERT_SUCCESS(status, "Failed to load model");
  ASSERT_THAT(model, ::testing::NotNull());

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t shape[1] = {8};

  M_TensorSpec *spec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_TensorSpec *spec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", gpuDevice);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(tensorMap, vector1, spec1, status);
  ASSERT_SUCCESS(status, "Failed to borrow input0");
  M_borrowTensorInto(tensorMap, vector2, spec2, status);
  ASSERT_SUCCESS(status, "Failed to borrow input1");

  M_AsyncTensor *input0 = M_getTensorByNameFrom(tensorMap, "input0", status);
  ASSERT_SUCCESS(status, "Failed to get input0");
  M_AsyncTensor *input1 = M_getTensorByNameFrom(tensorMap, "input1", status);
  ASSERT_SUCCESS(status, "Failed to get input1");

  // Attempt replay with a key that was never captured.
  uint64_t graphKey = 99;
  M_AsyncTensor *inputArray[2] = {input0, input1};
  M_replayModelSync(context, model, &graphKey, 1, inputArray, 2, status);
  EXPECT_TRUE(M_isError(status));

  M_freeTensor(input0);
  M_freeTensor(input1);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec1);
  M_freeTensorSpec(spec2);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(GPUTest, CaptureRejectsUnsafeKeyReuse) {
  M_AsyncCompiledModel *compiledModel = nullptr;
  M_AsyncModel *model = loadVectorAddModel(context, &compiledModel, status);
  ASSERT_SUCCESS(status, "Failed to load model");
  ASSERT_THAT(model, ::testing::NotNull());

  int64_t shape[1] = {8};

  // Create two distinct sets of input buffers with the same shape.
  float data1a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float data1b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
  M_TensorSpec *specA0 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "a_input0", gpuDevice);
  M_TensorSpec *specA1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "a_input1", gpuDevice);
  M_AsyncTensorMap *mapA = M_newAsyncTensorMap(context);
  M_borrowTensorInto(mapA, data1a, specA0, status);
  ASSERT_SUCCESS(status, "Failed to borrow a_input0");
  M_borrowTensorInto(mapA, data1b, specA1, status);
  ASSERT_SUCCESS(status, "Failed to borrow a_input1");
  M_AsyncTensor *inputA0 = M_getTensorByNameFrom(mapA, "a_input0", status);
  ASSERT_SUCCESS(status, "Failed to get a_input0");
  M_AsyncTensor *inputA1 = M_getTensorByNameFrom(mapA, "a_input1", status);
  ASSERT_SUCCESS(status, "Failed to get a_input1");

  float data2a[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  float data2b[8] = {80, 70, 60, 50, 40, 30, 20, 10};
  M_TensorSpec *specB0 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "b_input0", gpuDevice);
  M_TensorSpec *specB1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "b_input1", gpuDevice);
  M_AsyncTensorMap *mapB = M_newAsyncTensorMap(context);
  M_borrowTensorInto(mapB, data2a, specB0, status);
  ASSERT_SUCCESS(status, "Failed to borrow b_input0");
  M_borrowTensorInto(mapB, data2b, specB1, status);
  ASSERT_SUCCESS(status, "Failed to borrow b_input1");
  M_AsyncTensor *inputB0 = M_getTensorByNameFrom(mapB, "b_input0", status);
  ASSERT_SUCCESS(status, "Failed to get b_input0");
  M_AsyncTensor *inputB1 = M_getTensorByNameFrom(mapB, "b_input1", status);
  ASSERT_SUCCESS(status, "Failed to get b_input1");

  // Capture with key=1 using buffer set A. Skip if graph capture is not
  // supported on this accelerator.
  uint64_t graphKey = 1;
  M_AsyncTensor *inputsA[2] = {inputA0, inputA1};
  size_t numOutputs = 0;
  M_AsyncTensor **capturedOutputs = M_captureModelSync(
      context, model, &graphKey, 1, inputsA, 2, &numOutputs, status);
  if (M_isError(status)) {
    GTEST_SKIP() << "Graph capture not supported: " << M_getError(status);
  }

  // Attempt replay with key=1 but different buffers (set B).
  M_AsyncTensor *inputsB[2] = {inputB0, inputB1};
  M_replayModelSync(context, model, &graphKey, 1, inputsB, 2, status);
  EXPECT_FAILURE(status, "Unsafe graph key reuse");

  M_freeTensor(capturedOutputs[0]);
  free(capturedOutputs);
  M_freeTensor(inputA0);
  M_freeTensor(inputA1);
  M_freeTensor(inputB0);
  M_freeTensor(inputB1);
  M_freeAsyncTensorMap(mapA);
  M_freeAsyncTensorMap(mapB);
  M_freeTensorSpec(specA0);
  M_freeTensorSpec(specA1);
  M_freeTensorSpec(specB0);
  M_freeTensorSpec(specB1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(GPUTest, DebugVerifyReplayModel) {
  M_AsyncCompiledModel *compiledModel = nullptr;
  M_AsyncModel *model = loadVectorAddModel(context, &compiledModel, status);
  ASSERT_SUCCESS(status, "Failed to load model");
  ASSERT_THAT(model, ::testing::NotNull());

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t shape[1] = {8};

  M_TensorSpec *spec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_TensorSpec *spec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", gpuDevice);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(tensorMap, vector1, spec1, status);
  ASSERT_SUCCESS(status, "Failed to borrow input0");
  M_borrowTensorInto(tensorMap, vector2, spec2, status);
  ASSERT_SUCCESS(status, "Failed to borrow input1");

  M_AsyncTensor *input0 = M_getTensorByNameFrom(tensorMap, "input0", status);
  ASSERT_SUCCESS(status, "Failed to get input0");
  M_AsyncTensor *input1 = M_getTensorByNameFrom(tensorMap, "input1", status);
  ASSERT_SUCCESS(status, "Failed to get input1");

  // Capture the graph first. Skip if graph capture is not supported.
  uint64_t graphKey = 1;
  M_AsyncTensor *inputArray[2] = {input0, input1};
  size_t numOutputs = 0;
  M_AsyncTensor **capturedOutputs = M_captureModelSync(
      context, model, &graphKey, 1, inputArray, 2, &numOutputs, status);
  if (M_isError(status)) {
    GTEST_SKIP() << "Graph capture not supported: " << M_getError(status);
  }

  // Debug verify should succeed since the graph matches eager execution.
  M_debugVerifyReplayModelSync(context, model, &graphKey, 1, inputArray, 2,
                               status);
  EXPECT_SUCCESS(status, "Debug verify replay failed");

  M_freeTensor(capturedOutputs[0]);
  free(capturedOutputs);
  M_freeTensor(input0);
  M_freeTensor(input1);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec1);
  M_freeTensorSpec(spec2);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}
