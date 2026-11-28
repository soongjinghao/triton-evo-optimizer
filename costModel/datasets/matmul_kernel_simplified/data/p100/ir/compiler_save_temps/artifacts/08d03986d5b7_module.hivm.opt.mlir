#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<()[s0] -> (32, s0)>
#map2 = affine_map<()[s0] -> (64, s0)>
#map3 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map4 = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
#map5 = affine_map<()[s0] -> (48, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c20_i8 = arith.constant 20 : i8
    return %c20_i8 : i8
  }
  func.func private @set_l1_2d_float(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c-1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %true = arith.constant true
    %c48 = arith.constant 48 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    hivm.hir.set_mask_norm
    %0 = arith.index_cast %arg9 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 64], strides: [%0, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %1 = arith.index_cast %arg10 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 48], strides: [%1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x48xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %2 = arith.index_cast %arg11 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32, 48], strides: [%2, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x48xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %4 = arith.index_cast %arg6 : i32 to index
    %5 = affine.max #map()[%4]
    %6 = arith.index_cast %arg8 : i32 to index
    %7 = affine.max #map()[%6]
    %8 = affine.min #map1()[%5]
    %9 = affine.max #map()[%8]
    %10 = affine.min #map2()[%7]
    %11 = affine.max #map()[%10]
    %12 = arith.cmpi slt, %9, %c32 : index
    %13 = arith.cmpi slt, %11, %c64 : index
    %14 = arith.ori %12, %13 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%9, %11] [1, 1] : memref<32x64xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %15 = affine.apply #map3()[%9]
    %16 = affine.apply #map4()[%11]
    %subview_2 = memref.subview %3[0, 0, 0, 0] [%16, %15, 16, 8] [1, 1, 1, 1] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>>
    scf.if %14 {
      %collapse_shape = memref.collapse_shape %3 [[0, 1, 2, 3]] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
      %cast_15 = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @set_l1_2d_float(%cst, %cast_15) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    %cast = memref.cast %subview : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %subview_2 : memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_float(%cast, %cast_3) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %17 = hivm.hir.pointer_cast(%c8192_i64) : memref<6x4x16x8xf32, #hivm.address_space<cbuf>>
    %18 = arith.index_cast %arg7 : i32 to index
    %19 = affine.max #map()[%18]
    %20 = affine.min #map5()[%19]
    %21 = affine.max #map()[%20]
    %22 = arith.cmpi slt, %21, %c48 : index
    %23 = arith.ori %13, %22 : i1
    %subview_4 = memref.subview %reinterpret_cast_0[0, 0] [%11, %21] [1, 1] : memref<64x48xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    %24 = affine.apply #map3()[%11]
    %25 = affine.apply #map4()[%21]
    %subview_5 = memref.subview %17[0, 0, 0, 0] [%25, %24, 16, 8] [1, 1, 1, 1] : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
    scf.if %23 {
      %collapse_shape = memref.collapse_shape %17 [[0, 1, 2, 3]] : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> into memref<3072xf32, #hivm.address_space<cbuf>>
      %cast_15 = memref.cast %collapse_shape : memref<3072xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @set_l1_2d_float(%cst, %cast_15) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    %cast_6 = memref.cast %subview_4 : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_7 = memref.cast %subview_5 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_float(%cast_6, %cast_7) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %26 = hivm.hir.pointer_cast(%c0_i64) : memref<3x2x16x16xf32, #hivm.address_space<cc>>
    %cast_8 = memref.cast %3 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_9 = memref.cast %17 : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_10 = memref.cast %26 : memref<3x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_float_to_float(%cast_8, %cast_9, %true, %9, %11, %21, %cast_10, %c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %27 = affine.apply #map3()[%21]
    %subview_11 = memref.subview %26[0, 0, 0, 0] [%27, %15, 16, 16] [1, 1, 1, 1] : memref<3x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>>
    %subview_12 = memref.subview %reinterpret_cast_1[0, 0] [%9, %21] [1, 1] : memref<32x48xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %cast_13 = memref.cast %subview_11 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_14 = memref.cast %subview_12 : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_13, %cast_14, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
