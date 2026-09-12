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

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/device.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void printTensor(const float *vector, size_t numElements) {
  printf("[");
  for (size_t i = 0; i < numElements; ++i) {
    printf("%.1f", vector[i]);
    if (i < (numElements - 1))
      printf(", ");
  }
  printf("]\n");
}

int main() {
  printf("Loading MEF file and running vector addition...\n");

  int result = EXIT_SUCCESS;

  // Initialize MAX runtime
  M_Status *status = M_newStatus();
  M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();

  // Verify accelerator is available
  int acceleratorCount = M_getAcceleratorCount();
  if (acceleratorCount == 0) {
    printf("Error: No accelerator detected. This example requires a GPU.\n");
    result = EXIT_FAILURE;
    goto cleanupRuntimeConfig;
  }

  printf("Accelerator detected (%d available), using accelerator for "
         "execution.\n",
         acceleratorCount);

  // Create the host device (needed for borrowing data from CPU memory)
  M_Device *host = M_newDevice(M_HOST, 0, status);
  if (M_isError(status)) {
    printf("Error creating host device: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupRuntimeConfig;
  }
  M_runtimeConfigAddDevice(runtimeConfig, host);

  // Create the accelerator device
  M_Device *accelerator = M_newDevice(M_ACCELERATOR, 0, status);
  if (M_isError(status)) {
    printf("Error creating accelerator device: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupHost;
  }
  M_runtimeConfigAddDevice(runtimeConfig, accelerator);

  M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
  if (M_isError(status)) {
    printf("Error creating runtime context: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupAccelerator;
  }

  printf("Running on device: %s\n", M_getDeviceLabel(accelerator));

  // Load the cached model from a MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, "graph.mef");

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  if (M_isError(status)) {
    printf("Error compiling model: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupContext;
  }

  // Initialize the model
  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  if (M_isError(status)) {
    printf("Error initializing model: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupCompiledModel;
  }

  // Create input data - two vectors of 8 elements each
  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t shape[1] = {8};

  // Create tensor specs for inputs on host to borrow the data
  M_TensorSpec *hostInputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *hostInputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);

  // Borrow input data from host memory into a staging tensor map
  M_AsyncTensorMap *hostInputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(hostInputs, vector1, hostInputSpec1, status);
  if (M_isError(status)) {
    printf("Error adding vector1 to tensor map: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupHostInputs;
  }

  M_borrowTensorInto(hostInputs, vector2, hostInputSpec2, status);
  if (M_isError(status)) {
    printf("Error adding vector2 to tensor map: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupHostInputs;
  }

  // Get tensors from the host map
  M_AsyncTensor *hostInputTensor1 =
      M_getTensorByNameFrom(hostInputs, "input0", status);
  if (M_isError(status)) {
    printf("Error getting input tensor 1: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupHostInputs;
  }

  M_AsyncTensor *hostInputTensor2 =
      M_getTensorByNameFrom(hostInputs, "input1", status);
  if (M_isError(status)) {
    printf("Error getting input tensor 2: %s\n", M_getError(status));
    M_freeTensor(hostInputTensor1);
    result = EXIT_FAILURE;
    goto cleanupHostInputs;
  }

  // Copy tensors to accelerator
  printf("Copying input tensors to accelerator...\n");
  M_AsyncTensor *acceleratorInputTensor1 =
      M_copyTensorToDevice(hostInputTensor1, accelerator, status);
  if (M_isError(status)) {
    printf("Error copying input tensor 1 to accelerator: %s\n",
           M_getError(status));
    M_freeTensor(hostInputTensor1);
    M_freeTensor(hostInputTensor2);
    result = EXIT_FAILURE;
    goto cleanupHostInputs;
  }

  M_AsyncTensor *acceleratorInputTensor2 =
      M_copyTensorToDevice(hostInputTensor2, accelerator, status);
  if (M_isError(status)) {
    printf("Error copying input tensor 2 to accelerator: %s\n",
           M_getError(status));
    M_freeTensor(acceleratorInputTensor1);
    M_freeTensor(hostInputTensor1);
    M_freeTensor(hostInputTensor2);
    result = EXIT_FAILURE;
    goto cleanupHostInputs;
  }

  // Free host input tensors (no longer needed after copy)
  M_freeTensor(hostInputTensor1);
  M_freeTensor(hostInputTensor2);

  // Create input tensor map with accelerator tensor specs for model execution
  M_TensorSpec *acceleratorInputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", accelerator);
  M_TensorSpec *acceleratorInputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", accelerator);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  const void *acceleratorData1 = M_getTensorData(acceleratorInputTensor1);
  M_borrowTensorInto(inputs, (void *)(uintptr_t)acceleratorData1,
                     acceleratorInputSpec1, status);
  if (M_isError(status)) {
    printf("Error adding accelerator tensor 1 to input map: %s\n",
           M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupInputs;
  }

  const void *acceleratorData2 = M_getTensorData(acceleratorInputTensor2);
  M_borrowTensorInto(inputs, (void *)(uintptr_t)acceleratorData2,
                     acceleratorInputSpec2, status);
  if (M_isError(status)) {
    printf("Error adding accelerator tensor 2 to input map: %s\n",
           M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupInputs;
  }

  // Execute the model
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  if (M_isError(status)) {
    printf("Error executing model: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupInputs;
  }

  // Get the output tensor (on accelerator)
  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  if (M_isError(status)) {
    printf("Error getting output tensor: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupOutputs;
  }

  // Copy output tensor to host for reading
  M_AsyncTensor *hostOutputTensor =
      M_copyTensorToDevice(outputTensor, host, status);
  if (M_isError(status)) {
    printf("Error copying output tensor to host: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupOutputTensor;
  }

  // Get the output data from host tensor
  const float *outputData = (const float *)M_getTensorData(hostOutputTensor);
  if (outputData == NULL) {
    printf("Error: Output tensor data is NULL\n");
    result = EXIT_FAILURE;
    goto cleanupHostOutputTensor;
  }

  // Verify the results
  size_t numElements = M_getTensorNumElements(hostOutputTensor);
  if (numElements != 8) {
    printf("Error: wrong number of output elements, expected 8, got %zu",
           numElements);
    result = EXIT_FAILURE;
    goto cleanupHostOutputTensor;
  }

  printf("\nInput vectors:\n");
  printf("Vector 1: ");
  printTensor(vector1, numElements);

  printf("Vector 2: ");
  printTensor(vector2, numElements);

  printf("\nOutput vector (%zu elements):\n", numElements);
  printf("Result:   ");
  printTensor(outputData, numElements);

  printf("\nExpected: [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]\n");

  int correct = 1;
  for (size_t i = 0; i < numElements; ++i) {
    if (outputData[i] != 9.0f) {
      correct = 0;
      break;
    }
  }

  if (correct) {
    printf("\nVector addition successful! All results are correct.\n");
  } else {
    printf("\nVector addition failed - results don't match expected values.\n");
    result = EXIT_FAILURE;
  }

  // Clean up in reverse order of allocation
cleanupHostOutputTensor:
  M_freeTensor(hostOutputTensor);
cleanupOutputTensor:
  M_freeTensor(outputTensor);
cleanupOutputs:
  M_freeAsyncTensorMap(outputs);
cleanupInputs:
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(acceleratorInputSpec2);
  M_freeTensorSpec(acceleratorInputSpec1);
  M_freeTensor(acceleratorInputTensor2);
  M_freeTensor(acceleratorInputTensor1);
cleanupHostInputs:
  M_freeAsyncTensorMap(hostInputs);
  M_freeTensorSpec(hostInputSpec2);
  M_freeTensorSpec(hostInputSpec1);
  M_freeModel(model);
cleanupCompiledModel:
  M_freeCompiledModel(compiledModel);
cleanupContext:
  M_freeCompileConfig(compileConfig);
  M_freeRuntimeContext(context);
cleanupAccelerator:
  M_freeDevice(accelerator);
cleanupHost:
  M_freeDevice(host);
cleanupRuntimeConfig:
  M_freeRuntimeConfig(runtimeConfig);
  M_freeStatus(status);

  return result;
}
