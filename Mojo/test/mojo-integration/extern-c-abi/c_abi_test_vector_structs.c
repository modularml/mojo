// C reference for vector struct ABI tests.
// Mojo test: test_struct_arguments_vectors.mojo
//
// Bug being tested: a struct field of type LLVMFixedVectorType (e.g., the LLVM
// dialect type that Mojo's SIMD lowers to) is not recognized by
// isAllFloatStruct or classifyEightbyte because those functions check
// dyn_cast<mlir::VectorType> (MLIR built-in) but not
// dyn_cast<LLVMFixedVectorType> (LLVM dialect). Result: vector field → INTEGER
// coercion instead of SSE.

// GCC vector extension: 2-element float vector (8 bytes, naturally 8-byte
// aligned). Matches the ABI layout of Mojo's SIMD[DType.float32, 2].
typedef float float2 __attribute__((vector_size(8)));

struct VectorStruct8 {
  float2 v;
};

struct VectorStruct8 c_func_vec_8byte(struct VectorStruct8 s) {
  s.v[0] += 1.0f;
  s.v[1] += 1.0f;
  return s;
}
