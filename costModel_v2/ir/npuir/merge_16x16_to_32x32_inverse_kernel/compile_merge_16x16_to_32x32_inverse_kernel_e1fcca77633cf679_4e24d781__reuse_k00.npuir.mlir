// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_32x32_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c24_i32 = arith.constant 24 : i32
  %true = arith.constant true
  %c-16_i32 = arith.constant -16 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  %c3072 = arith.constant 3072 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = arith.index_cast %arg5 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
    %6 = arith.index_cast %arg9 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c24_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = arith.index_cast %3 : i32 to index
    %14 = arith.index_cast %1 : i32 to index
    %15 = arith.index_cast %c24_i32 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%12, %13, %15]
    %17 = arith.index_cast %16 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64_3 = arith.constant 1 : i64
    %18 = arith.select %17, %c0_i64_2, %c1_i64_3 : i64
    %19 = hivm.hir.pointer_cast(%c3072_i64, %c7168_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %20 = hivm.hir.pointer_cast(%c2048_i64, %c6144_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %21 = hivm.hir.pointer_cast(%c1024_i64, %c5120_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %22 = hivm.hir.pointer_cast(%c0_i64_1, %c4096_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_mask_norm
    %23 = arith.divsi %arg9, %arg8 : i32
    %24 = arith.remsi %23, %arg7 : i32
    %25 = arith.divsi %arg9, %4 : i32
    %26 = arith.remsi %25, %arg6 : i32
    %27 = arith.divsi %24, %c4_i32 : i32
    %28 = arith.remsi %24, %c4_i32 : i32
    %29 = arith.muli %27, %arg5 : i32
    %30 = arith.muli %29, %c4_i32 : i32
    %31 = arith.addi %30, %28 : i32
    %32 = arith.muli %31, %c32_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = arith.muli %26, %c32_i32 : i32
    %35 = arith.addi %34, %c16_i32 : i32
    %36 = arith.maxsi %35, %c0_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%33, %37]
    %39 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %37]
    %40 = affine.min affine_map<()[s0] -> (16, s0)>()[%39]
    %41 = arith.subi %c-16_i32, %34 : i32
    %42 = arith.maxsi %41, %c0_i32 : i32
    %43 = arith.index_cast %42 : i32 to index
    %44 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%43, %39]
    %45 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %44]
    %46 = arith.cmpi slt, %45, %c16 : index
    %47 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast = memref.cast %22 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
    hivm.hir.nd2nz {dst_continuous} ins(%47 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    %48 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_4 = memref.cast %21 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%48 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_5 = memref.cast %20 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%45, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %49 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%40, %44]
    %50 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%44]
    %51 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%44]
    %subview_6 = memref.subview %20[0, %50, %51, 0] [2, %49, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %subview_6 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %46 {
      %collapse_shape = memref.collapse_shape %20 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %52 = hivm.hir.pointer_cast(%c0_i64_1) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_8 = memref.cast %52 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_4, %cast_5, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c1_i64, %c-1_i64, %c-1_i64, %11, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %53 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%53 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %cast_9 = memref.cast %19 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%53 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    %54 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_10 = memref.cast %54 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %c-1_i64_11 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_9, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_10 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64_11, %c-1_i64_11, %18, %11, %c-1_i64_11, %c-1_i64_11 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %55 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_10 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%55 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    annotation.mark %55 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_32x32_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c9248_i64 = arith.constant 9248 : i64
  %c8736_i64 = arith.constant 8736 : i64
  %c8704_i64 = arith.constant 8704 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c8160_i64 = arith.constant 8160 : i64
  %c7648_i64 = arith.constant 7648 : i64
  %c7136_i64 = arith.constant 7136 : i64
  %c7072_i64 = arith.constant 7072 : i64
  %c6560_i64 = arith.constant 6560 : i64
  %c6528_i64 = arith.constant 6528 : i64
  %c6016_i64 = arith.constant 6016 : i64
  %c5984_i64 = arith.constant 5984 : i64
  %c5472_i64 = arith.constant 5472 : i64
  %c4960_i64 = arith.constant 4960 : i64
  %c4896_i64 = arith.constant 4896 : i64
  %c4832_i64 = arith.constant 4832 : i64
  %c4320_i64 = arith.constant 4320 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c3232_i64 = arith.constant 3232 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c576_i64 = arith.constant 576 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c3072 = arith.constant 3072 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %c2_i32 = arith.constant 2 : i32
  %c128_i32 = arith.constant 128 : i32
  %c18_i32 = arith.constant 18 : i32
  %c-16_i32 = arith.constant -16 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c576_i64) : memref<0xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%6 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%7 : memref<0xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %8 = arith.index_cast %arg5 : i32 to index
  %9 = hivm.hir.get_sub_block_idx -> i64
  %10 = arith.index_cast %9 : i64 to index
  %11 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%10]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  %subview = memref.subview %5[%11] [8] [1] {to_be_bubbled_slice} : memref<16xi32, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %expand_shape_1 = memref.expand_shape %subview [[0, 1]] output_shape [1, 8] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
  %12 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%12 : memref<16x8xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %13 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %14 = hivm.hir.pointer_cast(%c1600_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%13 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%14 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %15 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %13 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %12 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %15 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<128xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %6 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %17 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c2112_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %18 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  %19 = hivm.hir.pointer_cast(%c2624_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%19 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg9, %arg8 : i32
    %21 = arith.remsi %20, %arg7 : i32
    %22 = arith.divsi %arg9, %4 : i32
    %23 = arith.remsi %22, %arg6 : i32
    %24 = arith.divsi %21, %c4_i32 : i32
    %25 = arith.remsi %21, %c4_i32 : i32
    %26 = arith.muli %24, %arg5 : i32
    %27 = arith.muli %26, %c4_i32 : i32
    %28 = arith.addi %27, %25 : i32
    %29 = arith.muli %28, %c32_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %23, %c32_i32 : i32
    %32 = arith.maxsi %31, %c0_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%30, %33]
    %35 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %33]
    %36 = affine.min affine_map<()[s0] -> (16, s0)>()[%35]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %37 = arith.addi %31, %c16_i32 : i32
    %38 = arith.maxsi %37, %c0_i32 : i32
    %39 = arith.index_cast %38 : i32 to index
    %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%30, %39]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%30, %39]
    %42 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %39]
    %43 = affine.min affine_map<()[s0] -> (16, s0)>()[%42]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %44 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %45 = arith.subi %c0_i32, %31 : i32
    %46 = arith.maxsi %45, %c0_i32 : i32
    %47 = arith.index_cast %46 : i32 to index
    %48 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%47, %35]
    %49 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%36, %48]
    %50 = arith.cmpi slt, %49, %c16 : index
    %subview_9 = memref.subview %reinterpret_cast[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %subview_9[0, 0] [%49, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %44[%48, 0] [%49, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %50 {
      %collapse_shape_44 = memref.collapse_shape %44 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%subview_10 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %51 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %52 = arith.subi %c-16_i32, %31 : i32
    %53 = arith.maxsi %52, %c0_i32 : i32
    %54 = arith.index_cast %53 : i32 to index
    %55 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%54, %42]
    %56 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%43, %55]
    %57 = arith.cmpi slt, %56, %c16 : index
    %subview_12 = memref.subview %reinterpret_cast_8[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_13 = memref.subview %subview_12[0, 0] [%56, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_14 = memref.subview %51[%55, 0] [%56, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %57 {
      %collapse_shape_44 = memref.collapse_shape %51 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.load ins(%subview_13 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %58 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %44 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %58 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c4256_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_15, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %60 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %60 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_16, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<128xf32, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %61 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_17, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_18 : memref<128xf32, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c4320_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %51 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %62 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c4832_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_19, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%63 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    %64 = hivm.hir.pointer_cast(%c4320_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %64 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_20, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c4320_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %65 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_21, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>)
    %66 = arith.subi %arg5, %31 : i32
    %67 = arith.minsi %66, %c16_i32 : i32
    scf.for %arg10 = %c2_i32 to %67 step %c1_i32  : i32 {
      %84 = arith.addi %31, %arg10 : i32
      %85 = arith.muli %84, %c128_i32 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %86]
      %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %88 = hivm.hir.pointer_cast(%c4896_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%88 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %89 = hivm.hir.pointer_cast(%c4896_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%88, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%89 : memref<16xf32, #hivm.address_space<ub>>)
      %90 = hivm.hir.pointer_cast(%c4896_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%89, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%90 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_45 = memref.expand_shape %90 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %91 = hivm.hir.pointer_cast(%c4960_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %92 = hivm.hir.pointer_cast(%c5472_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_45, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%91 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%92 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %93 = hivm.hir.pointer_cast(%c5984_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %94 = hivm.hir.pointer_cast(%c6016_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%91 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%93 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%94 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_46 = memref.collapse_shape %93 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_47 = memref.subview %90[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %95 = hivm.hir.pointer_cast(%c6528_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%95 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %expand_shape_48 = memref.expand_shape %95 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %96 = hivm.hir.pointer_cast(%c6560_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%96 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %97 = arith.index_cast %arg10 : i32 to index
      %98 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%97]
      %99 = arith.cmpi slt, %97, %c0 : index
      %100 = arith.cmpi sge, %97, %c16 : index
      %101 = arith.ori %99, %100 : i1
      scf.if %101 {
      } else {
        %subview_49 = memref.subview %61[0, 0] [%97, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_50 = memref.subview %96[0, 0] [%97, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_53 = memref.subview %96[0, 0] [%98, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_54 = memref.subview %61[0, 0] [%98, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %68 = arith.minsi %66, %c32_i32 : i32
    scf.for %arg10 = %c18_i32 to %68 step %c1_i32  : i32 {
      %84 = arith.addi %31, %arg10 : i32
      %85 = arith.muli %84, %c128_i32 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %86]
      %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %88 = hivm.hir.pointer_cast(%c7072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%88 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %89 = hivm.hir.pointer_cast(%c7072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%88, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%89 : memref<16xf32, #hivm.address_space<ub>>)
      %90 = hivm.hir.pointer_cast(%c7072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%89, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%90 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_45 = memref.expand_shape %90 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %91 = hivm.hir.pointer_cast(%c7136_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %92 = hivm.hir.pointer_cast(%c7648_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_45, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%91 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%92 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %93 = hivm.hir.pointer_cast(%c8160_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %94 = hivm.hir.pointer_cast(%c8192_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%91 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%93 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%94 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_46 = memref.collapse_shape %93 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_47 = memref.subview %90[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %95 = hivm.hir.pointer_cast(%c8704_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%95 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %96 = arith.subi %arg10, %c16_i32 : i32
      %expand_shape_48 = memref.expand_shape %95 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %97 = hivm.hir.pointer_cast(%c8736_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%97 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %98 = arith.index_cast %96 : i32 to index
      %99 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%98]
      %100 = arith.cmpi slt, %98, %c0 : index
      %101 = arith.cmpi sge, %98, %c16 : index
      %102 = arith.ori %100, %101 : i1
      scf.if %102 {
      } else {
        %subview_49 = memref.subview %65[0, 0] [%98, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_50 = memref.subview %97[0, 0] [%98, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_53 = memref.subview %97[0, 0] [%99, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_54 = memref.subview %65[0, 0] [%99, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %69 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_18, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %70 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_24 = memref.subview %70[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%69 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %71 = hivm.hir.pointer_cast(%c4320_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %71 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_22, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %72 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_26 = memref.subview %72[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%71 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %73 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_27 = memref.subview %73[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %74 = hivm.hir.pointer_cast(%c9248_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_29 = memref.collapse_shape %74 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape_28 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    %75 = hivm.hir.pointer_cast(%c9248_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %75 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_29, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
    %76 = hivm.hir.pointer_cast(%c9248_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %76 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_33 = memref.subview %reinterpret_cast_32[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_34 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_35 = memref.subview %reinterpret_cast_34[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_37 = memref.subview %reinterpret_cast_36[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_38 = memref.subview %69[%48, 0] [%49, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_39 = memref.subview %subview_33[0, 0] [%49, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_38 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_40 = memref.subview %71[%55, 0] [%56, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_41 = memref.subview %subview_37[0, 0] [%56, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_40 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    %77 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%55, %10]
    %78 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%42, %10]
    %79 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%78, %55, %10]
    %80 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%79, %77]
    %81 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%77, %10]
    %subview_42 = memref.subview %76[%81, 0] [%80, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %82 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %43, %55]
    %83 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%82, %43, %55]
    %subview_43 = memref.subview %subview_35[0, 0] [%83, 16] [1, 1] : memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_42 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_43 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
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
