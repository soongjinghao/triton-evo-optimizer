// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @recompute_w_u_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c14336_i64 = arith.constant 14336 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c96_i32 = arith.constant 96 : i32
  %c16384 = arith.constant 16384 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
  %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  %6 = arith.index_cast %1 : i32 to index
  %7 = arith.muli %arg13, %arg12 : i32
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c96_i32  : i32 {
    %8 = arith.index_cast %arg14 : i32 to index
    %9 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%8)[%6]
    scf.for %arg15 = %c0 to %9 step %c1 {
      %10 = arith.index_cast %arg14 : i32 to index
      %11 = arith.index_cast %3 : i32 to index
      %12 = arith.index_cast %1 : i32 to index
      %13 = arith.index_cast %c96_i32 : i32 to index
      %14 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> (s0 + ((s2 - s3) floordiv s4) * s1)>()[%arg15, %9, %10, %11, %13]
      %15 = arith.index_cast %14 : index to i64
      %16 = arith.index_cast %arg14 : i32 to index
      %17 = arith.index_cast %3 : i32 to index
      %18 = arith.index_cast %1 : i32 to index
      %19 = arith.index_cast %c96_i32 : i32 to index
      %20 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> ((s0 + ((s2 - s3) floordiv s4) * s1) mod 2)>()[%arg15, %9, %16, %17, %19]
      %21 = arith.index_cast %20 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %22 = arith.select %21, %c2_i64, %c3_i64 : i64
      %23 = arith.index_cast %20 : index to i1
      %c0_i64_2 = arith.constant 0 : i64
      %c1_i64_3 = arith.constant 1 : i64
      %24 = arith.select %23, %c0_i64_2, %c1_i64_3 : i64
      %25 = hivm.hir.pointer_cast(%c5120_i64, %c14336_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      %26 = hivm.hir.pointer_cast(%c1024_i64, %c10240_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      %27 = hivm.hir.pointer_cast(%c0_i64_1, %c9216_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %28 = arith.index_cast %arg15 : index to i64
      %29 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg15)[%8]
      %30 = arith.index_cast %29 : index to i32
      %31 = arith.divsi %30, %arg13 : i32
      %32 = arith.remsi %31, %arg12 : i32
      %33 = arith.divsi %30, %7 : i32
      %34 = arith.remsi %33, %arg11 : i32
      %35 = arith.divsi %32, %c4_i32 : i32
      %36 = arith.remsi %32, %c4_i32 : i32
      %37 = arith.muli %35, %arg10 : i32
      %38 = arith.muli %37, %c4_i32 : i32
      %39 = arith.muli %34, %c16_i32 : i32
      %40 = arith.maxsi %39, %c0_i32 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = arith.addi %38, %36 : i32
      %43 = arith.muli %42, %c16_i32 : i32
      %44 = arith.index_cast %43 : i32 to index
      %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%44, %41]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%45], sizes: [16, 16], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %cast = memref.cast %27 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %24]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %46 = arith.muli %42, %c32_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%47, %41]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%48], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %4[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
      %cast_5 = memref.cast %26 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %49 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
      %cast_6 = memref.cast %49 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_5, %true, %c16, %c16, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_6 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c1_i64, %c-1_i64, %c-1_i64, %15, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_6 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%reinterpret_cast_4 : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %50 = arith.muli %42, %c64_i32 : i32
      %51 = arith.index_cast %50 : i32 to index
      %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%51, %41]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%52], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %5[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_9 = memref.cast %25 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %22]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %28
      %53 = hivm.hir.pointer_cast(%c4096_i64) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
      %cast_10 = memref.cast %53 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %c-1_i64_11 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_9, %true, %c16, %c16, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_10 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_11, %c0_i64, %24, %22, %15, %c-1_i64_11, %c-1_i64_11 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_10 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%reinterpret_cast_7 : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
  }
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
  %c5184_i64 = arith.constant 5184 : i64
  %c4896_i64 = arith.constant 4896 : i64
  %c5152_i64 = arith.constant 5152 : i64
  %c4640_i64 = arith.constant 4640 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2336_i64 = arith.constant 2336 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c0_i64 = arith.constant 0 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c96_i32 = arith.constant 96 : i32
  %c8 = arith.constant 8 : index
  %c16384 = arith.constant 16384 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
  %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
  %6 = arith.index_cast %1 : i32 to index
  %7 = arith.muli %arg13, %arg12 : i32
  %8 = hivm.hir.get_sub_block_idx -> i64
  %9 = arith.index_cast %8 : i64 to index
  %10 = arith.cmpi eq, %9, %c0 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c96_i32  : i32 {
    %11 = arith.index_cast %arg14 : i32 to index
    %12 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%11)[%6]
    scf.for %arg15 = %c0 to %12 step %c1 {
      %13 = arith.index_cast %arg15 : index to i64
      %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg15)[%11]
      %15 = arith.index_cast %14 : index to i32
      %16 = arith.divsi %15, %arg13 : i32
      %17 = arith.remsi %16, %arg12 : i32
      %18 = arith.divsi %15, %7 : i32
      %19 = arith.remsi %18, %arg11 : i32
      %20 = arith.divsi %17, %c4_i32 : i32
      %21 = arith.remsi %17, %c4_i32 : i32
      %22 = arith.muli %20, %arg10 : i32
      %23 = arith.muli %22, %c4_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %21 : i32 to index
      %26 = arith.muli %19, %c16_i32 : i32
      %27 = arith.maxsi %26, %c0_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 + s2)>()[%28, %24, %25]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [16, 1], strides: [4, 4] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %30 = arith.addi %23, %21 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4)>()[%31, %28]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%32], sizes: [16], strides: [4] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %33 = arith.muli %30, %c32_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%34, %28]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %4[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %36 = arith.muli %30, %c64_i32 : i32
      %37 = arith.index_cast %36 : i32 to index
      %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%37, %28]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %5[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %13
      scf.for %arg16 = %c0 to %c16 step %c8 {
        %subview_5 = memref.subview %reinterpret_cast_1[%arg16, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %39 = hivm.hir.pointer_cast(%c0_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%subview_5 : memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%39 : memref<8x64xf32, #hivm.address_space<ub>>) {vector_producer_to_fuse_0} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        %subview_6 = memref.subview %reinterpret_cast[%arg16, 0] [8, 1] [1, 1] : memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
        %40 = hivm.hir.pointer_cast(%c2304_i64) : memref<8x1xf32, #hivm.address_space<ub>>
        %41 = hivm.hir.pointer_cast(%c2048_i64) : memref<8x1x8xf32, #hivm.address_space<ub>>
        %subview_7 = memref.subview %41[0, 0, 0] [8, 1, 1] [1, 1, 1] : memref<8x1x8xf32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
        %collapse_shape_8 = memref.collapse_shape %subview_6 [[0, 1]] : memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> into memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_9 = memref.collapse_shape %subview_7 [[0, 1]] : memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> into memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape_8 : memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_9 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>>) {vector_producer_to_fuse_0} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_10 = memref.collapse_shape %40 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vdeinterleave ins(%collapse_shape_9 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%collapse_shape_10 : memref<8xf32, #hivm.address_space<ub>>) channel_num = 8 index_mode = <CHANNEL_0>
        %42 = hivm.hir.pointer_cast(%c0_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %43 = hivm.hir.pointer_cast(%c2336_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul {vector_producer_to_fuse_0} ins(%39, %40 : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%42 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%43 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_11 = memref.subview %collapse_shape[%arg16, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        scf.if %10 {
          %collapse_shape_16 = memref.collapse_shape %42 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
          %collapse_shape_17 = memref.collapse_shape %subview_11 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%collapse_shape_16 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_0}
        } {limit_sub_block_id0}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        %subview_12 = memref.subview %reinterpret_cast_2[%arg16, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %44 = hivm.hir.pointer_cast(%c2592_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%subview_12 : memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%44 : memref<8x64xf32, #hivm.address_space<ub>>) {vector_producer_to_fuse_0} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %45 = hivm.hir.pointer_cast(%c2592_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %46 = hivm.hir.pointer_cast(%c4640_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul {vector_producer_to_fuse_0} ins(%44, %40 : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%45 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%46 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
        %subview_13 = memref.subview %reinterpret_cast_0[%arg16] [8] [1] : memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
        %47 = hivm.hir.pointer_cast(%c5152_i64) : memref<8xf32, #hivm.address_space<ub>>
        %48 = hivm.hir.pointer_cast(%c4896_i64) : memref<8x8xf32, #hivm.address_space<ub>>
        %subview_14 = memref.subview %48[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%subview_13 : memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>>) {vector_producer_to_fuse_0} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vdeinterleave ins(%subview_14 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%47 : memref<8xf32, #hivm.address_space<ub>>) channel_num = 8 index_mode = <CHANNEL_0>
        %49 = hivm.hir.pointer_cast(%c5152_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vexp {vector_producer_to_fuse_0} ins(%47 : memref<8xf32, #hivm.address_space<ub>>) outs(%49 : memref<8xf32, #hivm.address_space<ub>>)
        %expand_shape = memref.expand_shape %49 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
        %50 = hivm.hir.pointer_cast(%c2592_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %51 = hivm.hir.pointer_cast(%c5184_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul {vector_producer_to_fuse_0} ins(%45, %expand_shape : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%50 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_15 = memref.subview %collapse_shape_4[%arg16, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        scf.if %10 {
          %collapse_shape_16 = memref.collapse_shape %50 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
          %collapse_shape_17 = memref.collapse_shape %subview_15 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%collapse_shape_16 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_1}
        } {limit_sub_block_id0}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      }
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 4
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
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
