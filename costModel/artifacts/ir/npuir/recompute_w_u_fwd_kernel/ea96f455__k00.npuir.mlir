// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @recompute_w_u_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c24_i32 = arith.constant 24 : i32
  %true = arith.constant true
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c8192 = arith.constant 8192 : index
  %c2 = arith.constant 2 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  %4 = arith.muli %arg13, %arg12 : i32
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c24_i32  : i32 {
    %5 = arith.index_cast %arg14 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c24_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i64
    %11 = arith.index_cast %arg14 : i32 to index
    %12 = arith.index_cast %3 : i32 to index
    %13 = arith.index_cast %1 : i32 to index
    %14 = arith.index_cast %c24_i32 : i32 to index
    %15 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%11, %12, %14]
    %16 = arith.index_cast %15 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64_3 = arith.constant 1 : i64
    %17 = arith.select %16, %c0_i64_2, %c1_i64_3 : i64
    %18 = hivm.hir.pointer_cast(%c4096_i64, %c24576_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %19 = hivm.hir.pointer_cast(%c0_i64_1, %c20480_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg14, %arg13 : i32
    %21 = arith.remsi %20, %arg12 : i32
    %22 = arith.divsi %arg14, %4 : i32
    %23 = arith.remsi %22, %arg11 : i32
    %24 = arith.divsi %21, %c8_i32 : i32
    %25 = arith.remsi %21, %c8_i32 : i32
    %26 = arith.muli %24, %arg10 : i32
    %27 = arith.muli %26, %c8_i32 : i32
    %28 = arith.muli %23, %c32_i32 : i32
    %29 = arith.maxsi %28, %c0_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.addi %27, %25 : i32
    %32 = arith.muli %31, %c32_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%33, %30]
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%34], sizes: [32, 32], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %cast = memref.cast %19 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %17]
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %35 = arith.muli %31, %c64_i32 : i32
    %36 = arith.index_cast %35 : i32 to index
    %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%36, %30]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%37], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %38 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_5 = memref.cast %18 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%38 : memref<32x64xf32, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    %39 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    %cast_6 = memref.cast %39 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_5, %true, %c32, %c32, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_6 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %10, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_6 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%reinterpret_cast_4 : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %40 = arith.muli %31, %c128_i32 : i32
    %41 = arith.index_cast %40 : i32 to index
    %42 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
    scf.for %arg15 = %c0 to %c2 step %c1 {
      %43 = arith.index_cast %arg14 : i32 to index
      %44 = arith.index_cast %3 : i32 to index
      %45 = arith.index_cast %1 : i32 to index
      %46 = arith.index_cast %c24_i32 : i32 to index
      %47 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + ((s1 - s2) floordiv s3) * 2)>()[%arg15, %43, %44, %46]
      %48 = arith.index_cast %47 : index to i64
      %49 = arith.index_cast %arg14 : i32 to index
      %50 = arith.index_cast %3 : i32 to index
      %51 = arith.index_cast %1 : i32 to index
      %52 = arith.index_cast %c24_i32 : i32 to index
      %53 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 mod 2)>()[%arg15, %49, %50, %52]
      %54 = arith.index_cast %53 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %55 = arith.select %54, %c2_i64, %c3_i64 : i64
      %56 = hivm.hir.pointer_cast(%c12288_i64, %c32768_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %56 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      %57 = arith.index_cast %arg15 : index to i32
      %58 = arith.muli %57, %c64_i32 : i32
      %59 = arith.maxsi %58, %c0_i32 : i32
      %60 = arith.index_cast %59 : i32 to index
      %61 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%60, %41, %30]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%61], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %42[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_8 = memref.cast %56 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %55]
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %62 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
      %cast_9 = memref.cast %62 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64_10 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_8, %true, %c32, %c32, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_9 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_10, %c0_i64_0, %c-1_i64_10, %55, %48, %c-1_i64_10, %c-1_i64_10 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_9 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%reinterpret_cast_7 : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %17]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @recompute_w_u_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c24320_i64 = arith.constant 24320 : i64
  %c20224_i64 = arith.constant 20224 : i64
  %c19712_i64 = arith.constant 19712 : i64
  %c15616_i64 = arith.constant 15616 : i64
  %c11520_i64 = arith.constant 11520 : i64
  %c10496_i64 = arith.constant 10496 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c2176_i64 = arith.constant 2176 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c24_i32 = arith.constant 24 : i32
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8192 = arith.constant 8192 : index
  %c2 = arith.constant 2 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  %5 = hivm.hir.get_sub_block_idx -> i64
  %6 = arith.index_cast %5 : i64 to index
  %7 = arith.cmpi eq, %6, %c0 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg14 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %8 = arith.divsi %arg14, %arg13 : i32
    %9 = arith.remsi %8, %arg12 : i32
    %10 = arith.divsi %arg14, %4 : i32
    %11 = arith.remsi %10, %arg11 : i32
    %12 = arith.divsi %9, %c8_i32 : i32
    %13 = arith.remsi %9, %c8_i32 : i32
    %14 = arith.muli %12, %arg10 : i32
    %15 = arith.muli %14, %c8_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.index_cast %13 : i32 to index
    %18 = arith.muli %11, %c32_i32 : i32
    %19 = arith.maxsi %18, %c0_i32 : i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 8 + s1 + s2)>()[%20, %16, %17]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [32, 1], strides: [8, 8] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %22 = arith.addi %15, %13 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8)>()[%23, %20]
    %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%24], sizes: [32], strides: [8] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %25 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c0_i64) : memref<32x1x8xf32, #hivm.address_space<ub>>
    %subview = memref.subview %26[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x1x8xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> into memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_1 = memref.collapse_shape %subview [[0, 1]] : memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%collapse_shape : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_1 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %collapse_shape_2 = memref.collapse_shape %25 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdeinterleave ins(%collapse_shape_1 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<32xf32, #hivm.address_space<ub>>) channel_num = 8 index_mode = <CHANNEL_0>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    %27 = hivm.hir.pointer_cast(%c2176_i64) : memref<32xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c1152_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_3 = memref.subview %28[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vdeinterleave ins(%subview_3 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%27 : memref<32xf32, #hivm.address_space<ub>>) channel_num = 8 index_mode = <CHANNEL_0>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %29 = hivm.hir.pointer_cast(%c2176_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%27 : memref<32xf32, #hivm.address_space<ub>>) outs(%29 : memref<32xf32, #hivm.address_space<ub>>)
    %30 = arith.muli %22, %c64_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%31, %20]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %33 = hivm.hir.pointer_cast(%c2304_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%reinterpret_cast_4 : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%33 : memref<32x64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %34 = hivm.hir.pointer_cast(%c2304_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    %35 = hivm.hir.pointer_cast(%c10496_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%33, %25 : memref<32x64xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%34 : memref<32x64xf32, #hivm.address_space<ub>>) temp_buffer(%35 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %36 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %7 {
      %collapse_shape_5 = memref.collapse_shape %34 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_6 = memref.collapse_shape %36 [[0, 1]] : memref<32x64xf32, #hivm.address_space<gm>> into memref<2048xf32, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_5 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<2048xf32, #hivm.address_space<gm>>) {"inserted-store"}
    }
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    annotation.mark %36 : memref<32x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %37 = arith.muli %22, %c128_i32 : i32
    %38 = arith.index_cast %37 : i32 to index
    %expand_shape = memref.expand_shape %29 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %39 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
    scf.for %arg15 = %c0 to %c2 step %c1 {
      %40 = arith.index_cast %arg15 : index to i32
      %41 = arith.muli %40, %c64_i32 : i32
      %42 = arith.maxsi %41, %c0_i32 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%43, %38, %20]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %39[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_7 = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      scf.for %arg16 = %c0 to %c32 step %c16 {
        %subview_8 = memref.subview %reinterpret_cast_5[%arg16, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %45 = hivm.hir.pointer_cast(%c11520_i64) : memref<16x64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%subview_8 : memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%45 : memref<16x64xf32, #hivm.address_space<ub>>) {vector_producer_to_fuse_0} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %subview_9 = memref.subview %25[%arg16, 0] [16, 1] [1, 1] : memref<32x1xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        %46 = hivm.hir.pointer_cast(%c15616_i64) : memref<16x64xf32, #hivm.address_space<ub>>
        %47 = hivm.hir.pointer_cast(%c19712_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul {vector_producer_to_fuse_0} ins(%45, %subview_9 : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%46 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%47 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        %subview_10 = memref.subview %expand_shape[%arg16, 0] [16, 1] [1, 1] : memref<32x1xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        %48 = hivm.hir.pointer_cast(%c20224_i64) : memref<16x64xf32, #hivm.address_space<ub>>
        %49 = hivm.hir.pointer_cast(%c24320_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul {vector_producer_to_fuse_0} ins(%46, %subview_10 : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%48 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%49 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_11 = memref.subview %collapse_shape_7[%arg16, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        scf.if %7 {
          %collapse_shape_12 = memref.collapse_shape %48 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
          %collapse_shape_13 = memref.collapse_shape %subview_11 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%collapse_shape_12 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_0}
        } {limit_sub_block_id0}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      }
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
