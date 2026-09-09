# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from max.gpu import warp_id, lane_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import thread_idx
from max.gpu.intrinsics import threadfence
from max.gpu.compute.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from std.memory import bitcast


def wgmma_kernel[
    M: Int,
    N: Int,
    K: Int,
    WMMA_M: Int,
    WMMA_N: Int,
    WMMA_K: Int,
    smem_operand_a_layout: Layout,
    smem_operand_b_layout: Layout,
    a_type: DType,
    b_type: DType,
](
    operand_a: LayoutTensor[a_type, Layout.row_major(M, K), MutAnyOrigin],
    operand_b: LayoutTensor[b_type, Layout.row_major(K, N), MutAnyOrigin],
    result_c: LayoutTensor[.int32, Layout.row_major(M, N), MutAnyOrigin],
):
    var smem_operand_a = LayoutTensor[
        a_type,
        smem_operand_a_layout,
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    var smem_operand_b = LayoutTensor[
        b_type,
        smem_operand_b_layout,
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    var c_reg = SIMD[.uint32, 4](0)

    for k_i in range(K // WMMA_K):
        var operand_a_tile = operand_a.tile[M, WMMA_K](0, k_i)
        var operand_b_tile = operand_b.tile[WMMA_K, N](k_i, 0)
        var operand_a_sm_tile = smem_operand_a.tile[M, WMMA_K](0, k_i)
        var operand_b_sm_tile = smem_operand_b.tile[WMMA_K, N](k_i, 0)

        if thread_idx.x == 0:
            operand_a_sm_tile.copy_from(operand_a_tile)
            operand_b_sm_tile.copy_from(operand_b_tile)

        barrier()

        var mat_a_desc = WGMMADescriptor.create[8, 64](operand_a_sm_tile.ptr)
        var mat_b_desc = WGMMADescriptor.create[1, 8](operand_b_sm_tile.ptr)

        wgmma_fence_aligned()

        c_reg = wgmma_async[
            WMMA_M,
            WMMA_N,
            WMMA_K,
            a_type=a_type,
            b_type=b_type,
        ](mat_a_desc, mat_b_desc, c_reg)
        wgmma_commit_group_sync()
        wgmma_wait_group_sync()
        threadfence()
        wgmma_fence_aligned()

    # Refer to this layout:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-D.png
    # Each warp updates a 16x8 tile, and within each tile,
    # every thread updates a 1x2 vector. The resulting distribution layout
    # is as follows:
    var c0 = bitcast[.int32, 4](c_reg)
    var th_local_res = (
        result_c.tile[16, 8](warp_id(), 0)
        .vectorize[1, 2]()
        .distribute[Layout.row_major(8, 4)](lane_id())
    )
    th_local_res[0, 0][0] = c0[0]
    th_local_res[0, 0][1] = c0[1]
    th_local_res[1, 0][0] = c0[2]
    th_local_res[1, 0][1] = c0[3]


# CHECK-LABEL: wgmma_s8_s8_s32_64x8x32
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 286 259 292 270 273 286 259 292
def wgmma_s8_s8_s32_64x8x32(ctx: DeviceContext) raises:
    print("== wgmma_s8_s8_s32_64x8x32")
    comptime M = 64
    comptime N = 8
    comptime K = 32
    comptime a_type = DType.int8
    comptime b_type = DType.int8

    var lhs = ManagedLayoutTensor[a_type, Layout.row_major(M, K)](ctx)
    arange(lhs.tensor(), end=9)
    # print(lhs.tensor())

    var rhs = ManagedLayoutTensor[b_type, Layout.row_major(K, N)](ctx)
    arange(rhs.tensor(), end=5)
    # print(rhs.tensor())

    var res = ManagedLayoutTensor[.int32, Layout.row_major(M, N)](ctx)

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-A.png
    comptime a_smem_layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(16, 2)),
        IntTuple(IntTuple(16, 128), IntTuple(1, 1024)),
    )
    # print_layout(a_smem_layout)
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-B.png
    comptime b_smem_layout = Layout(
        IntTuple(IntTuple(16, 2), 8), IntTuple(IntTuple(1, 128), 16)
    )
    # print_layout(b_smem_layout)

    comptime kernel = wgmma_kernel[
        M,
        N,
        K,
        64,
        8,
        32,
        a_smem_layout,
        b_smem_layout,
        a_type=a_type,
        b_type=b_type,
    ]
    ctx.enqueue_function[kernel](
        lhs.device_tensor(),
        rhs.device_tensor(),
        res.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(res.tensor())
    _ = lhs^
    _ = rhs^
    _ = res^


# CHECK-LABEL: wgmma_u8_u8_s32_64x8x32
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 263 256 234 272 255 263 256
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
def wgmma_u8_u8_s32_64x8x32(ctx: DeviceContext) raises:
    print("== wgmma_u8_u8_s32_64x8x32")
    comptime M = 64
    comptime N = 8
    comptime K = 32
    comptime a_type = DType.uint8
    comptime b_type = DType.uint8

    var lhs = ManagedLayoutTensor[a_type, Layout.row_major(M, K)](ctx)
    arange(lhs.tensor(), end=9)
    # print(lhs.tensor())

    var rhs = ManagedLayoutTensor[b_type, Layout.row_major(K, N)](ctx)
    arange(rhs.tensor(), end=5)
    # print(rhs.tensor())

    var res = ManagedLayoutTensor[.int32, Layout.row_major(M, N)](ctx)

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-A.png
    comptime a_smem_layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(16, 2)),
        IntTuple(IntTuple(16, 128), IntTuple(1, 1024)),
    )
    # print_layout(a_smem_layout)
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-B.png
    comptime b_smem_layout = Layout(
        IntTuple(IntTuple(16, 2), 8), IntTuple(IntTuple(1, 128), 16)
    )
    # print_layout(b_smem_layout)

    comptime kernel = wgmma_kernel[
        M,
        N,
        K,
        64,
        8,
        32,
        a_smem_layout,
        b_smem_layout,
        a_type=a_type,
        b_type=b_type,
    ]
    ctx.enqueue_function[kernel](
        lhs.device_tensor(),
        rhs.device_tensor(),
        res.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(res.tensor())
    _ = lhs^
    _ = rhs^
    _ = res^


# CHECK-LABEL: wgmma_s8_u8_s32_64x8x32
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
# CHECK: 237 250 213 241 239 237 250 213
# CHECK: 255 269 253 282 281 255 269 253
# CHECK: 228 261 239 242 260 228 261 239
# CHECK: 246 271 261 256 266 246 271 261
# CHECK: 255 246 242 248 269 255 246 242
# CHECK: 273 256 264 262 275 273 256 264
# CHECK: 237 239 241 258 245 237 239 241
# CHECK: 282 285 263 281 269 282 285 263
def wgmma_s8_u8_s32_64x8x32(ctx: DeviceContext) raises:
    print("== wgmma_s8_u8_s32_64x8x32")
    comptime M = 64
    comptime N = 8
    comptime K = 32
    comptime a_type = DType.int8
    comptime b_type = DType.uint8

    var lhs = ManagedLayoutTensor[a_type, Layout.row_major(M, K)](ctx)
    var lhs_tensor = lhs.tensor()
    arange(lhs_tensor, end=9)
    print(lhs_tensor)

    var rhs = ManagedLayoutTensor[b_type, Layout.row_major(K, N)](ctx)
    var rhs_tensor = rhs.tensor()
    arange(rhs_tensor, end=5)
    print(rhs_tensor)

    var res = ManagedLayoutTensor[.int32, Layout.row_major(M, N)](ctx)

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-A.png
    comptime a_smem_layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(16, 2)),
        IntTuple(IntTuple(16, 128), IntTuple(1, 1024)),
    )
    # print_layout(a_smem_layout)
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-B.png
    comptime b_smem_layout = Layout(
        IntTuple(IntTuple(16, 2), 8), IntTuple(IntTuple(1, 128), 16)
    )
    # print_layout(b_smem_layout)

    comptime kernel = wgmma_kernel[
        M,
        N,
        K,
        64,
        8,
        32,
        a_smem_layout,
        b_smem_layout,
        a_type=a_type,
        b_type=b_type,
    ]
    ctx.enqueue_function[kernel](
        lhs.device_tensor(),
        rhs.device_tensor(),
        res.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(res.tensor())
    _ = lhs^
    _ = rhs^
    _ = res^


# CHECK-LABEL: wgmma_u8_s8_s32_64x8x32
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
# CHECK: 236 219 262 225 238 236 219 262
# CHECK: 276 260 259 288 257 276 260 259
# CHECK: 244 247 265 243 231 244 247 265
# CHECK: 239 279 244 279 259 239 279 244
# CHECK: 243 266 259 252 260 243 266 259
# CHECK: 220 271 247 243 279 220 271 247
# CHECK: 269 267 280 243 271 269 267 280
# CHECK: 219 236 268 225 272 219 236 268
def wgmma_u8_s8_s32_64x8x32(ctx: DeviceContext) raises:
    print("== wgmma_u8_s8_s32_64x8x32")
    comptime M = 64
    comptime N = 8
    comptime K = 32
    comptime a_type = DType.uint8
    comptime b_type = DType.int8

    var lhs = ManagedLayoutTensor[a_type, Layout.row_major(M, K)](ctx)
    var lhs_tensor = lhs.tensor()
    arange(lhs_tensor, end=9)
    print(lhs_tensor)

    var rhs = ManagedLayoutTensor[b_type, Layout.row_major(K, N)](ctx)
    var rhs_tensor = rhs.tensor()
    arange(rhs_tensor, end=5)
    print(rhs_tensor)

    var res = ManagedLayoutTensor[.int32, Layout.row_major(M, N)](ctx)

    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-A.png
    comptime a_smem_layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(16, 2)),
        IntTuple(IntTuple(16, 128), IntTuple(1, 1024)),
    )
    # print_layout(a_smem_layout)
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N32-core-matrices-B.png
    comptime b_smem_layout = Layout(
        IntTuple(IntTuple(16, 2), 8), IntTuple(IntTuple(1, 128), 16)
    )
    # print_layout(b_smem_layout)

    comptime kernel = wgmma_kernel[
        M,
        N,
        K,
        64,
        8,
        32,
        a_smem_layout,
        b_smem_layout,
        a_type=a_type,
        b_type=b_type,
    ]
    ctx.enqueue_function[kernel](
        lhs.device_tensor(),
        rhs.device_tensor(),
        res.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()
    print(res.tensor())
    _ = lhs^
    _ = rhs^
    _ = res^


def main() raises:
    with DeviceContext() as ctx:
        wgmma_s8_s8_s32_64x8x32(ctx)
        wgmma_u8_u8_s32_64x8x32(ctx)
        wgmma_s8_u8_s32_64x8x32(ctx)
        wgmma_u8_s8_s32_64x8x32(ctx)
