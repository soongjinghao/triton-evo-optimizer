// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_32x32_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
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
func.func @merge_16x16_to_32x32_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c12288_i64 = arith.constant 12288 : i64
  %c11776_i64 = arith.constant 11776 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c11232_i64 = arith.constant 11232 : i64
  %c10208_i64 = arith.constant 10208 : i64
  %c9952_i64 = arith.constant 9952 : i64
  %c9856_i64 = arith.constant 9856 : i64
  %c9824_i64 = arith.constant 9824 : i64
  %c9312_i64 = arith.constant 9312 : i64
  %c9248_i64 = arith.constant 9248 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c8704_i64 = arith.constant 8704 : i64
  %c8672_i64 = arith.constant 8672 : i64
  %c8160_i64 = arith.constant 8160 : i64
  %c7648_i64 = arith.constant 7648 : i64
  %c7584_i64 = arith.constant 7584 : i64
  %c7552_i64 = arith.constant 7552 : i64
  %c6528_i64 = arith.constant 6528 : i64
  %c6272_i64 = arith.constant 6272 : i64
  %c6176_i64 = arith.constant 6176 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c5632_i64 = arith.constant 5632 : i64
  %c5568_i64 = arith.constant 5568 : i64
  %c5536_i64 = arith.constant 5536 : i64
  %c5024_i64 = arith.constant 5024 : i64
  %c4992_i64 = arith.constant 4992 : i64
  %c4480_i64 = arith.constant 4480 : i64
  %c3968_i64 = arith.constant 3968 : i64
  %c3904_i64 = arith.constant 3904 : i64
  %c10720_i64 = arith.constant 10720 : i64
  %c3840_i64 = arith.constant 3840 : i64
  %c3328_i64 = arith.constant 3328 : i64
  %c7040_i64 = arith.constant 7040 : i64
  %c3264_i64 = arith.constant 3264 : i64
  %c2752_i64 = arith.constant 2752 : i64
  %c2240_i64 = arith.constant 2240 : i64
  %c1728_i64 = arith.constant 1728 : i64
  %c1632_i64 = arith.constant 1632 : i64
  %c1120_i64 = arith.constant 1120 : i64
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
  %c2_i32 = arith.constant 2 : i32
  %c128_i32 = arith.constant 128 : i32
  %c18_i32 = arith.constant 18 : i32
  %c16 = arith.constant 16 : index
  %c-16_i32 = arith.constant -16 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant -1.000000e+00 : f32
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
  %6 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%5 : memref<16xi32, #hivm.address_space<ub>>) outs(%6 : memref<16xf32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %6 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c64_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %8 = hivm.hir.pointer_cast(%c576_i64) : memref<0xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xf32, #hivm.address_space<ub>>) outs(%7 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%8 : memref<0xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %9 = arith.index_cast %arg5 : i32 to index
  %10 = hivm.hir.get_sub_block_idx -> i64
  %11 = arith.index_cast %10 : i64 to index
  %12 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%11]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  %subview = memref.subview %6[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %expand_shape_2 = memref.expand_shape %subview [[0, 1]] output_shape [1, 8] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
  %13 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_2 : memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%13 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %14 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %13 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %14 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>) compare_mode = <gt>
  %15 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %15 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<128xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c1120_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  %17 = hivm.hir.pointer_cast(%c1632_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_5 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%17 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg9, %arg8 : i32
    %19 = arith.remsi %18, %arg7 : i32
    %20 = arith.divsi %arg9, %4 : i32
    %21 = arith.remsi %20, %arg6 : i32
    %22 = arith.divsi %19, %c4_i32 : i32
    %23 = arith.remsi %19, %c4_i32 : i32
    %24 = arith.muli %22, %arg5 : i32
    %25 = arith.muli %24, %c4_i32 : i32
    %26 = arith.addi %25, %23 : i32
    %27 = arith.muli %26, %c32_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29 = arith.muli %21, %c32_i32 : i32
    %30 = arith.maxsi %29, %c0_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%28, %31]
    %33 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%9, %31]
    %34 = affine.min affine_map<()[s0] -> (16, s0)>()[%33]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%32], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %35 = arith.addi %29, %c16_i32 : i32
    %36 = arith.maxsi %35, %c0_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%28, %37]
    %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%28, %37]
    %40 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%9, %37]
    %41 = affine.min affine_map<()[s0] -> (16, s0)>()[%40]
    %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %42 = hivm.hir.pointer_cast(%c1728_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %43 = arith.subi %c0_i32, %29 : i32
    %44 = arith.maxsi %43, %c0_i32 : i32
    %45 = arith.index_cast %44 : i32 to index
    %46 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%45, %33]
    %47 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%34, %46]
    %48 = arith.cmpi slt, %47, %c16 : index
    %subview_8 = memref.subview %reinterpret_cast[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %subview_8[0, 0] [%47, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %42[%46, 0] [%47, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %48 {
      %collapse_shape_45 = memref.collapse_shape %42 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%subview_9 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %49 = hivm.hir.pointer_cast(%c2240_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %50 = arith.subi %c-16_i32, %29 : i32
    %51 = arith.maxsi %50, %c0_i32 : i32
    %52 = arith.index_cast %51 : i32 to index
    %53 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%52, %40]
    %54 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%41, %53]
    %55 = arith.cmpi slt, %54, %c16 : index
    %subview_11 = memref.subview %reinterpret_cast_7[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %subview_11[0, 0] [%54, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_13 = memref.subview %49[%53, 0] [%54, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %55 {
      %collapse_shape_45 = memref.collapse_shape %49 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.load ins(%subview_12 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %56 = hivm.hir.pointer_cast(%c2752_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %42 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %56 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%c3264_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%57 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %58 = hivm.hir.pointer_cast(%c2752_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %58 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_15, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<128xf32, #hivm.address_space<ub>>)
    %59 = hivm.hir.pointer_cast(%c7040_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_16, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<128xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c3328_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %49 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %60 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c3840_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_18, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    %62 = hivm.hir.pointer_cast(%c3328_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %62 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_19, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c10720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %63 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_20, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
    %64 = arith.subi %arg5, %29 : i32
    %65 = arith.minsi %64, %c16_i32 : i32
    %66 = arith.minsi %64, %c32_i32 : i32
    %67 = scf.for %arg10 = %c2_i32 to %65 step %c1_i32 iter_args(%arg11 = %59) -> (memref<16x8xf32, #hivm.address_space<ub>>)  : i32 {
      %84 = arith.sitofp %arg10 : i32 to f32
      %85 = arith.addi %29, %arg10 : i32
      %86 = arith.muli %85, %c128_i32 : i32
      %87 = arith.index_cast %86 : i32 to index
      %88 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %87]
      %reinterpret_cast_45 = memref.reinterpret_cast %arg3 to offset: [%88], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %89 = hivm.hir.pointer_cast(%c3904_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%89 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %90 = hivm.hir.pointer_cast(%c3904_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%89, %cst_1 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%90 : memref<16xf32, #hivm.address_space<ub>>)
      %91 = hivm.hir.pointer_cast(%c3904_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%90, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%91 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_46 = memref.expand_shape %91 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %92 = hivm.hir.pointer_cast(%c3968_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %93 = hivm.hir.pointer_cast(%c4480_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_46, %arg11 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%92 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%93 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %94 = hivm.hir.pointer_cast(%c4992_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %95 = hivm.hir.pointer_cast(%c5024_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%92 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%94 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%95 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_47 = memref.collapse_shape %94 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_48 = memref.subview %91[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %96 = hivm.hir.pointer_cast(%c5536_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_48, %collapse_shape_47 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%96 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %97 = hivm.hir.pointer_cast(%c5568_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%84 : f32) outs(%97 : memref<16xf32, #hivm.address_space<ub>>)
      %98 = hivm.hir.pointer_cast(%c5568_i64) : memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%6, %97 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%98 : memref<16xi1, #hivm.address_space<ub>>)
      %expand_shape_49 = memref.expand_shape %96 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %99 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_49 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%99 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %100 = hivm.hir.pointer_cast(%c6144_i64) : memref<16xf16, #hivm.address_space<ub>>
      %101 = hivm.hir.pointer_cast(%c6176_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%98 : memref<16xi1, #hivm.address_space<ub>>) outs(%100 : memref<16xf16, #hivm.address_space<ub>>) temp_buffer(%101 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %expand_shape_50 = memref.expand_shape %100 [[0, 1]] output_shape [16, 1] : memref<16xf16, #hivm.address_space<ub>> into memref<16x1xf16, #hivm.address_space<ub>>
      %102 = hivm.hir.pointer_cast(%c6272_i64) : memref<16x8xf16, #hivm.address_space<ub>>
      %103 = hivm.hir.pointer_cast(%c6528_i64) : memref<256xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_50 : memref<16x1xf16, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf16, #hivm.address_space<ub>>) temp_buffer(%103 : memref<256xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
      %104 = hivm.hir.pointer_cast(%c6272_i64) : memref<16x8xi1, #hivm.address_space<ub>>
      %collapse_shape_51 = memref.collapse_shape %102 [[0, 1]] : memref<16x8xf16, #hivm.address_space<ub>> into memref<128xf16, #hivm.address_space<ub>>
      %collapse_shape_52 = memref.collapse_shape %104 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_51, %cst_0 : memref<128xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_52 : memref<128xi1, #hivm.address_space<ub>>)
      %105 = hivm.hir.pointer_cast(%c6272_i64) : memref<16x8xi1, #hivm.address_space<ub>>
      %collapse_shape_53 = memref.collapse_shape %105 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_52 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<128xi1, #hivm.address_space<ub>>)
      %106 = hivm.hir.pointer_cast(%c7040_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_54 = memref.collapse_shape %99 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_55 = memref.collapse_shape %arg11 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_56 = memref.collapse_shape %106 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %107 = hivm.hir.pointer_cast(%c7552_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_53, %collapse_shape_54, %collapse_shape_55 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%107 : memref<8xf32, #hivm.address_space<ub>>)
      scf.yield %106 : memref<16x8xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %68 = scf.for %arg10 = %c18_i32 to %66 step %c1_i32 iter_args(%arg11 = %63) -> (memref<16x8xf32, #hivm.address_space<ub>>)  : i32 {
      %84 = arith.subi %arg10, %c16_i32 : i32
      %85 = arith.sitofp %84 : i32 to f32
      %86 = arith.addi %29, %arg10 : i32
      %87 = arith.muli %86, %c128_i32 : i32
      %88 = arith.index_cast %87 : i32 to index
      %89 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%28, %88]
      %reinterpret_cast_45 = memref.reinterpret_cast %arg3 to offset: [%89], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %90 = hivm.hir.pointer_cast(%c7584_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%90 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %91 = hivm.hir.pointer_cast(%c7584_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%90, %cst_1 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%91 : memref<16xf32, #hivm.address_space<ub>>)
      %92 = hivm.hir.pointer_cast(%c7584_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%91, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%92 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_46 = memref.expand_shape %92 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %93 = hivm.hir.pointer_cast(%c7648_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %94 = hivm.hir.pointer_cast(%c8160_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_46, %arg11 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%93 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%94 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %95 = hivm.hir.pointer_cast(%c8672_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %96 = hivm.hir.pointer_cast(%c8704_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%93 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%95 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%96 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_47 = memref.collapse_shape %95 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_48 = memref.subview %92[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %97 = hivm.hir.pointer_cast(%c9216_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_48, %collapse_shape_47 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%97 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %98 = hivm.hir.pointer_cast(%c9248_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%85 : f32) outs(%98 : memref<16xf32, #hivm.address_space<ub>>)
      %99 = hivm.hir.pointer_cast(%c9248_i64) : memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%6, %98 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%99 : memref<16xi1, #hivm.address_space<ub>>)
      %expand_shape_49 = memref.expand_shape %97 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %100 = hivm.hir.pointer_cast(%c9312_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_49 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%100 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %101 = hivm.hir.pointer_cast(%c9824_i64) : memref<16xf16, #hivm.address_space<ub>>
      %102 = hivm.hir.pointer_cast(%c9856_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%99 : memref<16xi1, #hivm.address_space<ub>>) outs(%101 : memref<16xf16, #hivm.address_space<ub>>) temp_buffer(%102 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %expand_shape_50 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf16, #hivm.address_space<ub>> into memref<16x1xf16, #hivm.address_space<ub>>
      %103 = hivm.hir.pointer_cast(%c9952_i64) : memref<16x8xf16, #hivm.address_space<ub>>
      %104 = hivm.hir.pointer_cast(%c10208_i64) : memref<256xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_50 : memref<16x1xf16, #hivm.address_space<ub>>) outs(%103 : memref<16x8xf16, #hivm.address_space<ub>>) temp_buffer(%104 : memref<256xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
      %105 = hivm.hir.pointer_cast(%c9952_i64) : memref<16x8xi1, #hivm.address_space<ub>>
      %collapse_shape_51 = memref.collapse_shape %103 [[0, 1]] : memref<16x8xf16, #hivm.address_space<ub>> into memref<128xf16, #hivm.address_space<ub>>
      %collapse_shape_52 = memref.collapse_shape %105 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_51, %cst_0 : memref<128xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_52 : memref<128xi1, #hivm.address_space<ub>>)
      %106 = hivm.hir.pointer_cast(%c9952_i64) : memref<16x8xi1, #hivm.address_space<ub>>
      %collapse_shape_53 = memref.collapse_shape %106 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_52 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<128xi1, #hivm.address_space<ub>>)
      %107 = hivm.hir.pointer_cast(%c10720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_54 = memref.collapse_shape %100 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_55 = memref.collapse_shape %arg11 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_56 = memref.collapse_shape %107 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %108 = hivm.hir.pointer_cast(%c11232_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_53, %collapse_shape_54, %collapse_shape_55 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%108 : memref<8xf32, #hivm.address_space<ub>>)
      scf.yield %107 : memref<16x8xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %69 = hivm.hir.pointer_cast(%c11264_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %67 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_22, %collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %70 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_24 = memref.subview %70[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%69 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %71 = hivm.hir.pointer_cast(%c11776_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %71 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vadd ins(%collapse_shape_25, %collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %72 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_27 = memref.subview %72[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%71 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_27 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_27 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %73 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_28 = memref.subview %73[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %74 = hivm.hir.pointer_cast(%c12288_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_30 = memref.collapse_shape %74 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape_29 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    %75 = hivm.hir.pointer_cast(%c12288_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %75 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_30, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
    %76 = hivm.hir.pointer_cast(%c12288_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %76 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_31, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_32 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_33 = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_34 = memref.subview %reinterpret_cast_33[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_35 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_36 = memref.subview %reinterpret_cast_35[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_37 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_38 = memref.subview %reinterpret_cast_37[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_39 = memref.subview %69[%46, 0] [%47, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_40 = memref.subview %subview_34[0, 0] [%47, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_39 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_40 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_41 = memref.subview %71[%53, 0] [%54, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_42 = memref.subview %subview_38[0, 0] [%54, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_41 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_42 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    %77 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%53, %11]
    %78 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%40, %11]
    %79 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%78, %53, %11]
    %80 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%79, %77]
    %81 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%77, %11]
    %subview_43 = memref.subview %76[%81, 0] [%80, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %82 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%11, %41, %53]
    %83 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%82, %41, %53]
    %subview_44 = memref.subview %subview_36[0, 0] [%83, 16] [1, 1] : memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_43 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_44 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
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
