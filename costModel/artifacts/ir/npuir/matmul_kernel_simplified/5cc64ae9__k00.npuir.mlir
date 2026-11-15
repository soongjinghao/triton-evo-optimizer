// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %true = arith.constant true
  %c48 = arith.constant 48 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = arith.index_cast %arg9 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 64], strides: [%2, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %3 = arith.index_cast %arg10 : i32 to index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 48], strides: [%3, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x48xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %4 = arith.index_cast %arg11 : i32 to index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32, 48], strides: [%4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x48xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %5 = hivm.hir.pointer_cast(%c0_i64_0) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
  %cast = memref.cast %5 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
  %6 = arith.index_cast %arg6 : i32 to index
  %7 = affine.max affine_map<()[s0] -> (0, s0)>()[%6]
  %8 = arith.index_cast %arg8 : i32 to index
  %9 = affine.max affine_map<()[s0] -> (0, s0)>()[%8]
  %10 = affine.min affine_map<()[s0] -> (32, s0)>()[%7]
  %11 = affine.max affine_map<()[s0] -> (0, s0)>()[%10]
  %12 = affine.min affine_map<()[s0] -> (64, s0)>()[%9]
  %13 = affine.max affine_map<()[s0] -> (0, s0)>()[%12]
  %14 = arith.cmpi slt, %11, %c32 : index
  %15 = arith.cmpi slt, %13, %c64 : index
  %16 = arith.ori %14, %15 : i1
  %subview = memref.subview %reinterpret_cast[0, 0] [%11, %13] [1, 1] : memref<32x64xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %17 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%11]
  %18 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%13]
  %subview_3 = memref.subview %5[0, 0, 0, 0] [%18, %17, 16, 8] [1, 1, 1, 1] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>>
  %cast_4 = memref.cast %subview_3 : memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  scf.if %16 {
    %collapse_shape = memref.collapse_shape %5 [[0, 1, 2, 3]] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>>)
    hivm.hir.pipe_barrier[<PIPE_MTE2>]
  } {hivm.unlikely_condition}
  hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
  %19 = hivm.hir.pointer_cast(%c8192_i64) : memref<6x4x16x8xf32, #hivm.address_space<cbuf>>
  %cast_5 = memref.cast %19 : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
  %20 = arith.index_cast %arg7 : i32 to index
  %21 = affine.max affine_map<()[s0] -> (0, s0)>()[%20]
  %22 = affine.min affine_map<()[s0] -> (48, s0)>()[%21]
  %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
  %24 = arith.cmpi slt, %23, %c48 : index
  %25 = arith.ori %15, %24 : i1
  %subview_6 = memref.subview %reinterpret_cast_1[0, 0] [%13, %23] [1, 1] : memref<64x48xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  %26 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%13]
  %27 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%23]
  %subview_7 = memref.subview %19[0, 0, 0, 0] [%27, %26, 16, 8] [1, 1, 1, 1] : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
  %cast_8 = memref.cast %subview_7 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  scf.if %25 {
    %collapse_shape = memref.collapse_shape %19 [[0, 1, 2, 3]] : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> into memref<3072xf32, #hivm.address_space<cbuf>>
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<3072xf32, #hivm.address_space<cbuf>>)
    hivm.hir.pipe_barrier[<PIPE_MTE2>]
  } {hivm.unlikely_condition}
  hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
  %28 = hivm.hir.pointer_cast(%c0_i64_0) : memref<3x2x16x16xf32, #hivm.address_space<cc>>
  %cast_9 = memref.cast %28 : memref<3x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  %c-1_i64 = arith.constant -1 : i64
  hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_5, %true, %11, %13, %23 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_9 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
  hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  %29 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%23]
  %subview_10 = memref.subview %28[0, 0, 0, 0] [%29, %17, 16, 16] [1, 1, 1, 1] : memref<3x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>>
  %cast_11 = memref.cast %subview_10 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
  %subview_12 = memref.subview %reinterpret_cast_2[0, 0] [%11, %23] [1, 1] : memref<32x48xf32, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_11 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_12 : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
