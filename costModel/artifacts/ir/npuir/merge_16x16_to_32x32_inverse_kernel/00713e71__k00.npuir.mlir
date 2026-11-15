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
  %c111392_i64 = arith.constant 111392 : i64
  %c110368_i64 = arith.constant 110368 : i64
  %c109344_i64 = arith.constant 109344 : i64
  %c109280_i64 = arith.constant 109280 : i64
  %c108768_i64 = arith.constant 108768 : i64
  %c108704_i64 = arith.constant 108704 : i64
  %c107680_i64 = arith.constant 107680 : i64
  %c106656_i64 = arith.constant 106656 : i64
  %c106592_i64 = arith.constant 106592 : i64
  %c106080_i64 = arith.constant 106080 : i64
  %c105056_i64 = arith.constant 105056 : i64
  %c104992_i64 = arith.constant 104992 : i64
  %c103968_i64 = arith.constant 103968 : i64
  %c102944_i64 = arith.constant 102944 : i64
  %c102880_i64 = arith.constant 102880 : i64
  %c102368_i64 = arith.constant 102368 : i64
  %c101344_i64 = arith.constant 101344 : i64
  %c101280_i64 = arith.constant 101280 : i64
  %c100256_i64 = arith.constant 100256 : i64
  %c99232_i64 = arith.constant 99232 : i64
  %c99168_i64 = arith.constant 99168 : i64
  %c98656_i64 = arith.constant 98656 : i64
  %c97632_i64 = arith.constant 97632 : i64
  %c97568_i64 = arith.constant 97568 : i64
  %c96544_i64 = arith.constant 96544 : i64
  %c95520_i64 = arith.constant 95520 : i64
  %c95456_i64 = arith.constant 95456 : i64
  %c94944_i64 = arith.constant 94944 : i64
  %c93920_i64 = arith.constant 93920 : i64
  %c93856_i64 = arith.constant 93856 : i64
  %c92832_i64 = arith.constant 92832 : i64
  %c91808_i64 = arith.constant 91808 : i64
  %c91744_i64 = arith.constant 91744 : i64
  %c91232_i64 = arith.constant 91232 : i64
  %c90208_i64 = arith.constant 90208 : i64
  %c90144_i64 = arith.constant 90144 : i64
  %c89120_i64 = arith.constant 89120 : i64
  %c88096_i64 = arith.constant 88096 : i64
  %c88032_i64 = arith.constant 88032 : i64
  %c87520_i64 = arith.constant 87520 : i64
  %c86496_i64 = arith.constant 86496 : i64
  %c86432_i64 = arith.constant 86432 : i64
  %c85408_i64 = arith.constant 85408 : i64
  %c84384_i64 = arith.constant 84384 : i64
  %c84320_i64 = arith.constant 84320 : i64
  %c83808_i64 = arith.constant 83808 : i64
  %c82784_i64 = arith.constant 82784 : i64
  %c82720_i64 = arith.constant 82720 : i64
  %c81696_i64 = arith.constant 81696 : i64
  %c80672_i64 = arith.constant 80672 : i64
  %c80608_i64 = arith.constant 80608 : i64
  %c80096_i64 = arith.constant 80096 : i64
  %c79072_i64 = arith.constant 79072 : i64
  %c79008_i64 = arith.constant 79008 : i64
  %c77984_i64 = arith.constant 77984 : i64
  %c76960_i64 = arith.constant 76960 : i64
  %c76896_i64 = arith.constant 76896 : i64
  %c76384_i64 = arith.constant 76384 : i64
  %c75360_i64 = arith.constant 75360 : i64
  %c75296_i64 = arith.constant 75296 : i64
  %c74272_i64 = arith.constant 74272 : i64
  %c73248_i64 = arith.constant 73248 : i64
  %c73184_i64 = arith.constant 73184 : i64
  %c72672_i64 = arith.constant 72672 : i64
  %c71648_i64 = arith.constant 71648 : i64
  %c71584_i64 = arith.constant 71584 : i64
  %c70560_i64 = arith.constant 70560 : i64
  %c69536_i64 = arith.constant 69536 : i64
  %c69472_i64 = arith.constant 69472 : i64
  %c68960_i64 = arith.constant 68960 : i64
  %c67936_i64 = arith.constant 67936 : i64
  %c67872_i64 = arith.constant 67872 : i64
  %c66848_i64 = arith.constant 66848 : i64
  %c65824_i64 = arith.constant 65824 : i64
  %c65760_i64 = arith.constant 65760 : i64
  %c65248_i64 = arith.constant 65248 : i64
  %c64224_i64 = arith.constant 64224 : i64
  %c64160_i64 = arith.constant 64160 : i64
  %c63136_i64 = arith.constant 63136 : i64
  %c62112_i64 = arith.constant 62112 : i64
  %c62048_i64 = arith.constant 62048 : i64
  %c61536_i64 = arith.constant 61536 : i64
  %c60512_i64 = arith.constant 60512 : i64
  %c60448_i64 = arith.constant 60448 : i64
  %c59424_i64 = arith.constant 59424 : i64
  %c58400_i64 = arith.constant 58400 : i64
  %c58336_i64 = arith.constant 58336 : i64
  %c57824_i64 = arith.constant 57824 : i64
  %c57760_i64 = arith.constant 57760 : i64
  %c56736_i64 = arith.constant 56736 : i64
  %c55712_i64 = arith.constant 55712 : i64
  %c55648_i64 = arith.constant 55648 : i64
  %c55136_i64 = arith.constant 55136 : i64
  %c54112_i64 = arith.constant 54112 : i64
  %c54048_i64 = arith.constant 54048 : i64
  %c53024_i64 = arith.constant 53024 : i64
  %c52000_i64 = arith.constant 52000 : i64
  %c51936_i64 = arith.constant 51936 : i64
  %c51424_i64 = arith.constant 51424 : i64
  %c50400_i64 = arith.constant 50400 : i64
  %c50336_i64 = arith.constant 50336 : i64
  %c49312_i64 = arith.constant 49312 : i64
  %c48288_i64 = arith.constant 48288 : i64
  %c48224_i64 = arith.constant 48224 : i64
  %c47712_i64 = arith.constant 47712 : i64
  %c46688_i64 = arith.constant 46688 : i64
  %c46624_i64 = arith.constant 46624 : i64
  %c45600_i64 = arith.constant 45600 : i64
  %c44576_i64 = arith.constant 44576 : i64
  %c44512_i64 = arith.constant 44512 : i64
  %c44000_i64 = arith.constant 44000 : i64
  %c42976_i64 = arith.constant 42976 : i64
  %c42912_i64 = arith.constant 42912 : i64
  %c41888_i64 = arith.constant 41888 : i64
  %c40864_i64 = arith.constant 40864 : i64
  %c40800_i64 = arith.constant 40800 : i64
  %c40288_i64 = arith.constant 40288 : i64
  %c39264_i64 = arith.constant 39264 : i64
  %c39200_i64 = arith.constant 39200 : i64
  %c38176_i64 = arith.constant 38176 : i64
  %c37152_i64 = arith.constant 37152 : i64
  %c37088_i64 = arith.constant 37088 : i64
  %c36576_i64 = arith.constant 36576 : i64
  %c35552_i64 = arith.constant 35552 : i64
  %c35488_i64 = arith.constant 35488 : i64
  %c34464_i64 = arith.constant 34464 : i64
  %c33440_i64 = arith.constant 33440 : i64
  %c33376_i64 = arith.constant 33376 : i64
  %c32864_i64 = arith.constant 32864 : i64
  %c31840_i64 = arith.constant 31840 : i64
  %c31776_i64 = arith.constant 31776 : i64
  %c30752_i64 = arith.constant 30752 : i64
  %c29728_i64 = arith.constant 29728 : i64
  %c29664_i64 = arith.constant 29664 : i64
  %c29152_i64 = arith.constant 29152 : i64
  %c28128_i64 = arith.constant 28128 : i64
  %c28064_i64 = arith.constant 28064 : i64
  %c27040_i64 = arith.constant 27040 : i64
  %c26016_i64 = arith.constant 26016 : i64
  %c25952_i64 = arith.constant 25952 : i64
  %c25440_i64 = arith.constant 25440 : i64
  %c24416_i64 = arith.constant 24416 : i64
  %c24352_i64 = arith.constant 24352 : i64
  %c23328_i64 = arith.constant 23328 : i64
  %c22304_i64 = arith.constant 22304 : i64
  %c22240_i64 = arith.constant 22240 : i64
  %c21728_i64 = arith.constant 21728 : i64
  %c20704_i64 = arith.constant 20704 : i64
  %c20640_i64 = arith.constant 20640 : i64
  %c19616_i64 = arith.constant 19616 : i64
  %c18592_i64 = arith.constant 18592 : i64
  %c18528_i64 = arith.constant 18528 : i64
  %c18016_i64 = arith.constant 18016 : i64
  %c16992_i64 = arith.constant 16992 : i64
  %c16928_i64 = arith.constant 16928 : i64
  %c15904_i64 = arith.constant 15904 : i64
  %c14880_i64 = arith.constant 14880 : i64
  %c14816_i64 = arith.constant 14816 : i64
  %c14304_i64 = arith.constant 14304 : i64
  %c13280_i64 = arith.constant 13280 : i64
  %c13216_i64 = arith.constant 13216 : i64
  %c12192_i64 = arith.constant 12192 : i64
  %c11168_i64 = arith.constant 11168 : i64
  %c11104_i64 = arith.constant 11104 : i64
  %c10592_i64 = arith.constant 10592 : i64
  %c9568_i64 = arith.constant 9568 : i64
  %c9504_i64 = arith.constant 9504 : i64
  %c9440_i64 = arith.constant 9440 : i64
  %c8416_i64 = arith.constant 8416 : i64
  %c8352_i64 = arith.constant 8352 : i64
  %c7328_i64 = arith.constant 7328 : i64
  %c6304_i64 = arith.constant 6304 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c5184_i64 = arith.constant 5184 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c24_i32 = arith.constant 24 : i32
  %cst = arith.constant -1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c-16_i32 = arith.constant -16 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c128_i32 = arith.constant 128 : i32
  %c31_i32 = arith.constant 31 : i32
  %c30_i32 = arith.constant 30 : i32
  %c29_i32 = arith.constant 29 : i32
  %c28_i32 = arith.constant 28 : i32
  %c27_i32 = arith.constant 27 : i32
  %c26_i32 = arith.constant 26 : i32
  %c25_i32 = arith.constant 25 : i32
  %c23_i32 = arith.constant 23 : i32
  %c22_i32 = arith.constant 22 : i32
  %c21_i32 = arith.constant 21 : i32
  %c20_i32 = arith.constant 20 : i32
  %c19_i32 = arith.constant 19 : i32
  %c18_i32 = arith.constant 18 : i32
  %c15_i32 = arith.constant 15 : i32
  %c14_i32 = arith.constant 14 : i32
  %c13_i32 = arith.constant 13 : i32
  %c12_i32 = arith.constant 12 : i32
  %c11_i32 = arith.constant 11 : i32
  %c10_i32 = arith.constant 10 : i32
  %c9_i32 = arith.constant 9 : i32
  %c8_i32 = arith.constant 8 : i32
  %c7_i32 = arith.constant 7 : i32
  %c6_i32 = arith.constant 6 : i32
  %c5_i32 = arith.constant 5 : i32
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %c3072 = arith.constant 3072 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c1088_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%6 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%7 : memref<128xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %expand_shape_1 = memref.expand_shape %5 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
  %8 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x16xi32, #hivm.address_space<ub>>) outs(%8 : memref<16x16xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %9 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %10 = hivm.hir.pointer_cast(%c3648_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x16xi32, #hivm.address_space<ub>>) outs(%9 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%10 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %11 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %9 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %8 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %11 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %12 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %6 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %13 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
  %14 = arith.index_cast %arg5 : i32 to index
  %15 = hivm.hir.pointer_cast(%c4160_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %15 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %16 = hivm.hir.pointer_cast(%c5184_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%16 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  %17 = hivm.hir.get_sub_block_idx -> i64
  %18 = arith.index_cast %17 : i64 to index
  %19 = arith.cmpi eq, %18, %c0 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
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
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %35 = arith.addi %31, %c16_i32 : i32
    %36 = arith.maxsi %35, %c0_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%30, %37]
    %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%30, %37]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %40 = hivm.hir.pointer_cast(%c5280_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %41 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%14, %33]
    %42 = affine.min affine_map<()[s0] -> (16, s0)>()[%41]
    %43 = arith.subi %c0_i32, %31 : i32
    %44 = arith.maxsi %43, %c0_i32 : i32
    %45 = arith.index_cast %44 : i32 to index
    %46 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%45, %41]
    %47 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%42, %46]
    %48 = arith.cmpi slt, %47, %c16 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%47, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %40[%46, 0] [%47, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %48 {
      %collapse_shape_369 = memref.collapse_shape %40 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_369 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %49 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %50 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%14, %37]
    %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
    %52 = arith.subi %c-16_i32, %31 : i32
    %53 = arith.maxsi %52, %c0_i32 : i32
    %54 = arith.index_cast %53 : i32 to index
    %55 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%54, %50]
    %56 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%51, %55]
    %57 = arith.cmpi slt, %56, %c16 : index
    %subview_10 = memref.subview %reinterpret_cast_8[0, 0] [%56, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %49[%55, 0] [%56, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %57 {
      %collapse_shape_369 = memref.collapse_shape %49 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_369 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_10 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %58 = hivm.hir.pointer_cast(%c7328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %40 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %58 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c8352_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_12, %cst_0 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c7328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %60 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_13, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_14 : memref<256xf32, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c7328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %61 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_14, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c8416_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %49 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %62 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c9440_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_16, %cst_0 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%63 : memref<16xf32, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c8416_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %64 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_17, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c8416_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %65 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_18, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_19 : memref<256xf32, #hivm.address_space<ub>>)
    %66 = arith.subi %arg5, %31 : i32
    %67 = arith.cmpi sgt, %66, %c2_i32 : i32
    %68 = arith.addi %31, %c2_i32 : i32
    %69 = arith.muli %68, %c128_i32 : i32
    %70 = arith.index_cast %69 : i32 to index
    %71 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %70]
    %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%71], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %72 = hivm.hir.pointer_cast(%c9504_i64) : memref<16xf32, #hivm.address_space<ub>>
    %73 = arith.index_castui %67 : i1 to index
    %74 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%73]
    %75 = arith.cmpi slt, %74, %c16 : index
    %subview_21 = memref.subview %reinterpret_cast_20[0] [%74] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_22 = memref.subview %72[0] [%74] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %75 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%72 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_21 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_22 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %76 = hivm.hir.pointer_cast(%c9504_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%72, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%76 : memref<16xf32, #hivm.address_space<ub>>)
    %77 = hivm.hir.pointer_cast(%c9504_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%76, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%77 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_23 = memref.expand_shape %77 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %78 = hivm.hir.pointer_cast(%c9568_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %79 = hivm.hir.pointer_cast(%c10592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_23, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%78 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%79 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %80 = hivm.hir.pointer_cast(%c11104_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %81 = hivm.hir.pointer_cast(%c11168_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%78 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%80 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%81 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_24 = memref.collapse_shape %80 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %82 = hivm.hir.pointer_cast(%c11104_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%77, %collapse_shape_24 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%82 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_25 = memref.expand_shape %82 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %83 = hivm.hir.pointer_cast(%c12192_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_25 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%83 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_26 = memref.subview %61[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_27 = memref.subview %83[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %subview_26 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %subview_27 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_28 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_30 = memref.subview %83[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_31 = memref.subview %61[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %subview_30 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_33 = memref.collapse_shape %subview_31 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_32 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>)
    %84 = arith.cmpi sgt, %66, %c3_i32 : i32
    %85 = arith.addi %31, %c3_i32 : i32
    %86 = arith.muli %85, %c128_i32 : i32
    %87 = arith.index_cast %86 : i32 to index
    %88 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %87]
    %reinterpret_cast_34 = memref.reinterpret_cast %arg3 to offset: [%88], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %89 = hivm.hir.pointer_cast(%c13216_i64) : memref<16xf32, #hivm.address_space<ub>>
    %90 = arith.index_castui %84 : i1 to index
    %91 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%90]
    %92 = arith.cmpi slt, %91, %c16 : index
    %subview_35 = memref.subview %reinterpret_cast_34[0] [%91] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_36 = memref.subview %89[0] [%91] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %92 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%89 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_35 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_36 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %93 = hivm.hir.pointer_cast(%c13216_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%89, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%93 : memref<16xf32, #hivm.address_space<ub>>)
    %94 = hivm.hir.pointer_cast(%c13216_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%93, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%94 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_37 = memref.expand_shape %94 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %95 = hivm.hir.pointer_cast(%c13280_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %96 = hivm.hir.pointer_cast(%c14304_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_37, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%95 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%96 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %97 = hivm.hir.pointer_cast(%c14816_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %98 = hivm.hir.pointer_cast(%c14880_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%95 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%97 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%98 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_38 = memref.collapse_shape %97 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %99 = hivm.hir.pointer_cast(%c14816_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%94, %collapse_shape_38 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_39 = memref.expand_shape %99 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %100 = hivm.hir.pointer_cast(%c15904_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_39 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%100 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_40 = memref.subview %100[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_41 = memref.collapse_shape %subview_40 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_32 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_41 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_42 = memref.subview %100[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_43 = memref.subview %61[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_44 = memref.collapse_shape %subview_42 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %subview_43 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_44 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_45 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>)
    %101 = arith.cmpi sgt, %66, %c4_i32 : i32
    %102 = arith.addi %31, %c4_i32 : i32
    %103 = arith.muli %102, %c128_i32 : i32
    %104 = arith.index_cast %103 : i32 to index
    %105 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %104]
    %reinterpret_cast_46 = memref.reinterpret_cast %arg3 to offset: [%105], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %106 = hivm.hir.pointer_cast(%c16928_i64) : memref<16xf32, #hivm.address_space<ub>>
    %107 = arith.index_castui %101 : i1 to index
    %108 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%107]
    %109 = arith.cmpi slt, %108, %c16 : index
    %subview_47 = memref.subview %reinterpret_cast_46[0] [%108] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_48 = memref.subview %106[0] [%108] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %109 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%106 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_47 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_48 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %110 = hivm.hir.pointer_cast(%c16928_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%106, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%110 : memref<16xf32, #hivm.address_space<ub>>)
    %111 = hivm.hir.pointer_cast(%c16928_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%110, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%111 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_49 = memref.expand_shape %111 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %112 = hivm.hir.pointer_cast(%c16992_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %113 = hivm.hir.pointer_cast(%c18016_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_49, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%112 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%113 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %114 = hivm.hir.pointer_cast(%c18528_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %115 = hivm.hir.pointer_cast(%c18592_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%112 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%114 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%115 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_50 = memref.collapse_shape %114 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %116 = hivm.hir.pointer_cast(%c18528_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%111, %collapse_shape_50 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%116 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_51 = memref.expand_shape %116 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %117 = hivm.hir.pointer_cast(%c19616_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_51 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%117 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_52 = memref.subview %117[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %subview_52 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_44 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_54 = memref.subview %117[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_55 = memref.subview %61[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_57 = memref.collapse_shape %subview_55 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_56 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_57 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>)
    %118 = arith.cmpi sgt, %66, %c5_i32 : i32
    %119 = arith.addi %31, %c5_i32 : i32
    %120 = arith.muli %119, %c128_i32 : i32
    %121 = arith.index_cast %120 : i32 to index
    %122 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %121]
    %reinterpret_cast_58 = memref.reinterpret_cast %arg3 to offset: [%122], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %123 = hivm.hir.pointer_cast(%c20640_i64) : memref<16xf32, #hivm.address_space<ub>>
    %124 = arith.index_castui %118 : i1 to index
    %125 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%124]
    %126 = arith.cmpi slt, %125, %c16 : index
    %subview_59 = memref.subview %reinterpret_cast_58[0] [%125] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_60 = memref.subview %123[0] [%125] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %126 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%123 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_59 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_60 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %127 = hivm.hir.pointer_cast(%c20640_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%123, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%127 : memref<16xf32, #hivm.address_space<ub>>)
    %128 = hivm.hir.pointer_cast(%c20640_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%127, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%128 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_61 = memref.expand_shape %128 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %129 = hivm.hir.pointer_cast(%c20704_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %130 = hivm.hir.pointer_cast(%c21728_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_61, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%129 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%130 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %131 = hivm.hir.pointer_cast(%c22240_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %132 = hivm.hir.pointer_cast(%c22304_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%129 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%131 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%132 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_62 = memref.collapse_shape %131 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %133 = hivm.hir.pointer_cast(%c22240_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%128, %collapse_shape_62 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%133 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_63 = memref.expand_shape %133 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %134 = hivm.hir.pointer_cast(%c23328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_63 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%134 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_64 = memref.subview %134[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_65 = memref.collapse_shape %subview_64 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_56 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_65 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_66 = memref.subview %134[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_67 = memref.subview %61[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_68 = memref.collapse_shape %subview_66 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_69 = memref.collapse_shape %subview_67 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_68 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_69 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>)
    %135 = arith.cmpi sgt, %66, %c6_i32 : i32
    %136 = arith.addi %31, %c6_i32 : i32
    %137 = arith.muli %136, %c128_i32 : i32
    %138 = arith.index_cast %137 : i32 to index
    %139 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %138]
    %reinterpret_cast_70 = memref.reinterpret_cast %arg3 to offset: [%139], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %140 = hivm.hir.pointer_cast(%c24352_i64) : memref<16xf32, #hivm.address_space<ub>>
    %141 = arith.index_castui %135 : i1 to index
    %142 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%141]
    %143 = arith.cmpi slt, %142, %c16 : index
    %subview_71 = memref.subview %reinterpret_cast_70[0] [%142] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_72 = memref.subview %140[0] [%142] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %143 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%140 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_71 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_72 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %144 = hivm.hir.pointer_cast(%c24352_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%140, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%144 : memref<16xf32, #hivm.address_space<ub>>)
    %145 = hivm.hir.pointer_cast(%c24352_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%144, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%145 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_73 = memref.expand_shape %145 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %146 = hivm.hir.pointer_cast(%c24416_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %147 = hivm.hir.pointer_cast(%c25440_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_73, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%146 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%147 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %148 = hivm.hir.pointer_cast(%c25952_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %149 = hivm.hir.pointer_cast(%c26016_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%146 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%148 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%149 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_74 = memref.collapse_shape %148 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %150 = hivm.hir.pointer_cast(%c24352_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%145, %collapse_shape_74 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%150 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_75 = memref.expand_shape %150 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %151 = hivm.hir.pointer_cast(%c27040_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_75 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%151 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_76 = memref.subview %151[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_77 = memref.collapse_shape %subview_76 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_68 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_77 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_78 = memref.subview %151[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_79 = memref.subview %61[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_80 = memref.collapse_shape %subview_78 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_81 = memref.collapse_shape %subview_79 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_80 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_81 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>)
    %152 = arith.cmpi sgt, %66, %c7_i32 : i32
    %153 = arith.addi %31, %c7_i32 : i32
    %154 = arith.muli %153, %c128_i32 : i32
    %155 = arith.index_cast %154 : i32 to index
    %156 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %155]
    %reinterpret_cast_82 = memref.reinterpret_cast %arg3 to offset: [%156], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %157 = hivm.hir.pointer_cast(%c28064_i64) : memref<16xf32, #hivm.address_space<ub>>
    %158 = arith.index_castui %152 : i1 to index
    %159 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%158]
    %160 = arith.cmpi slt, %159, %c16 : index
    %subview_83 = memref.subview %reinterpret_cast_82[0] [%159] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_84 = memref.subview %157[0] [%159] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %160 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%157 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_83 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_84 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %161 = hivm.hir.pointer_cast(%c28064_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%157, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%161 : memref<16xf32, #hivm.address_space<ub>>)
    %162 = hivm.hir.pointer_cast(%c28064_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%161, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%162 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_85 = memref.expand_shape %162 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %163 = hivm.hir.pointer_cast(%c28128_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %164 = hivm.hir.pointer_cast(%c29152_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_85, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%163 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%164 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %165 = hivm.hir.pointer_cast(%c29664_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %166 = hivm.hir.pointer_cast(%c29728_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%163 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%165 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%166 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_86 = memref.collapse_shape %165 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %167 = hivm.hir.pointer_cast(%c28064_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%162, %collapse_shape_86 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%167 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_87 = memref.expand_shape %167 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %168 = hivm.hir.pointer_cast(%c30752_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_87 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%168 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_88 = memref.subview %168[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_89 = memref.collapse_shape %subview_88 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_80 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_89 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_90 = memref.subview %168[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_91 = memref.subview %61[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_92 = memref.collapse_shape %subview_90 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_93 = memref.collapse_shape %subview_91 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_92 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_93 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>)
    %169 = arith.cmpi sgt, %66, %c8_i32 : i32
    %170 = arith.addi %31, %c8_i32 : i32
    %171 = arith.muli %170, %c128_i32 : i32
    %172 = arith.index_cast %171 : i32 to index
    %173 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %172]
    %reinterpret_cast_94 = memref.reinterpret_cast %arg3 to offset: [%173], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %174 = hivm.hir.pointer_cast(%c31776_i64) : memref<16xf32, #hivm.address_space<ub>>
    %175 = arith.index_castui %169 : i1 to index
    %176 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%175]
    %177 = arith.cmpi slt, %176, %c16 : index
    %subview_95 = memref.subview %reinterpret_cast_94[0] [%176] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_96 = memref.subview %174[0] [%176] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %177 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%174 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_95 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_96 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %178 = hivm.hir.pointer_cast(%c31776_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%174, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%178 : memref<16xf32, #hivm.address_space<ub>>)
    %179 = hivm.hir.pointer_cast(%c31776_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%178, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%179 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_97 = memref.expand_shape %179 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %180 = hivm.hir.pointer_cast(%c31840_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %181 = hivm.hir.pointer_cast(%c32864_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_97, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%180 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%181 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %182 = hivm.hir.pointer_cast(%c33376_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %183 = hivm.hir.pointer_cast(%c33440_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%180 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%182 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%183 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_98 = memref.collapse_shape %182 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %184 = hivm.hir.pointer_cast(%c33376_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%179, %collapse_shape_98 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%184 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_99 = memref.expand_shape %184 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %185 = hivm.hir.pointer_cast(%c34464_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_99 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%185 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_100 = memref.subview %185[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_101 = memref.collapse_shape %subview_100 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_92 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_101 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_102 = memref.subview %185[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_103 = memref.subview %61[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_104 = memref.collapse_shape %subview_102 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_105 = memref.collapse_shape %subview_103 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_104 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_105 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>)
    %186 = arith.cmpi sgt, %66, %c9_i32 : i32
    %187 = arith.addi %31, %c9_i32 : i32
    %188 = arith.muli %187, %c128_i32 : i32
    %189 = arith.index_cast %188 : i32 to index
    %190 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %189]
    %reinterpret_cast_106 = memref.reinterpret_cast %arg3 to offset: [%190], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %191 = hivm.hir.pointer_cast(%c35488_i64) : memref<16xf32, #hivm.address_space<ub>>
    %192 = arith.index_castui %186 : i1 to index
    %193 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%192]
    %194 = arith.cmpi slt, %193, %c16 : index
    %subview_107 = memref.subview %reinterpret_cast_106[0] [%193] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_108 = memref.subview %191[0] [%193] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %194 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%191 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_107 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_108 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %195 = hivm.hir.pointer_cast(%c35488_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%191, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%195 : memref<16xf32, #hivm.address_space<ub>>)
    %196 = hivm.hir.pointer_cast(%c35488_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%195, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%196 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_109 = memref.expand_shape %196 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %197 = hivm.hir.pointer_cast(%c35552_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %198 = hivm.hir.pointer_cast(%c36576_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_109, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%197 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%198 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %199 = hivm.hir.pointer_cast(%c37088_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %200 = hivm.hir.pointer_cast(%c37152_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%197 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%199 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%200 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_110 = memref.collapse_shape %199 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %201 = hivm.hir.pointer_cast(%c35488_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%196, %collapse_shape_110 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%201 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_111 = memref.expand_shape %201 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %202 = hivm.hir.pointer_cast(%c38176_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_111 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%202 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_112 = memref.subview %202[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_113 = memref.collapse_shape %subview_112 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_104 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_113 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_114 = memref.subview %202[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_115 = memref.subview %61[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_116 = memref.collapse_shape %subview_114 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_117 = memref.collapse_shape %subview_115 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_116 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_117 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>)
    %203 = arith.cmpi sgt, %66, %c10_i32 : i32
    %204 = arith.addi %31, %c10_i32 : i32
    %205 = arith.muli %204, %c128_i32 : i32
    %206 = arith.index_cast %205 : i32 to index
    %207 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %206]
    %reinterpret_cast_118 = memref.reinterpret_cast %arg3 to offset: [%207], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %208 = hivm.hir.pointer_cast(%c39200_i64) : memref<16xf32, #hivm.address_space<ub>>
    %209 = arith.index_castui %203 : i1 to index
    %210 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%209]
    %211 = arith.cmpi slt, %210, %c16 : index
    %subview_119 = memref.subview %reinterpret_cast_118[0] [%210] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_120 = memref.subview %208[0] [%210] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %211 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%208 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_119 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_120 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %212 = hivm.hir.pointer_cast(%c39200_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%208, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%212 : memref<16xf32, #hivm.address_space<ub>>)
    %213 = hivm.hir.pointer_cast(%c39200_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%212, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%213 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_121 = memref.expand_shape %213 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %214 = hivm.hir.pointer_cast(%c39264_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %215 = hivm.hir.pointer_cast(%c40288_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_121, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%214 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%215 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %216 = hivm.hir.pointer_cast(%c40800_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %217 = hivm.hir.pointer_cast(%c40864_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%214 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%216 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%217 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_122 = memref.collapse_shape %216 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %218 = hivm.hir.pointer_cast(%c39200_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%213, %collapse_shape_122 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%218 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_123 = memref.expand_shape %218 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %219 = hivm.hir.pointer_cast(%c41888_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_123 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%219 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_124 = memref.subview %219[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_125 = memref.collapse_shape %subview_124 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_116 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_125 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_126 = memref.subview %219[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_127 = memref.subview %61[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_128 = memref.collapse_shape %subview_126 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_129 = memref.collapse_shape %subview_127 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_128 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_129 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>)
    %220 = arith.cmpi sgt, %66, %c11_i32 : i32
    %221 = arith.addi %31, %c11_i32 : i32
    %222 = arith.muli %221, %c128_i32 : i32
    %223 = arith.index_cast %222 : i32 to index
    %224 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %223]
    %reinterpret_cast_130 = memref.reinterpret_cast %arg3 to offset: [%224], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %225 = hivm.hir.pointer_cast(%c42912_i64) : memref<16xf32, #hivm.address_space<ub>>
    %226 = arith.index_castui %220 : i1 to index
    %227 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%226]
    %228 = arith.cmpi slt, %227, %c16 : index
    %subview_131 = memref.subview %reinterpret_cast_130[0] [%227] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_132 = memref.subview %225[0] [%227] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %228 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%225 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_131 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_132 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %229 = hivm.hir.pointer_cast(%c42912_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%225, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%229 : memref<16xf32, #hivm.address_space<ub>>)
    %230 = hivm.hir.pointer_cast(%c42912_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%229, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%230 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_133 = memref.expand_shape %230 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %231 = hivm.hir.pointer_cast(%c42976_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %232 = hivm.hir.pointer_cast(%c44000_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_133, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%231 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%232 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %233 = hivm.hir.pointer_cast(%c44512_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %234 = hivm.hir.pointer_cast(%c44576_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%231 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%233 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%234 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_134 = memref.collapse_shape %233 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %235 = hivm.hir.pointer_cast(%c44512_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%230, %collapse_shape_134 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%235 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_135 = memref.expand_shape %235 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %236 = hivm.hir.pointer_cast(%c45600_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_135 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%236 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_136 = memref.subview %236[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_137 = memref.collapse_shape %subview_136 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_128 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_137 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_138 = memref.subview %236[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_139 = memref.subview %61[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_140 = memref.collapse_shape %subview_138 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_141 = memref.collapse_shape %subview_139 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_140 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_141 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>)
    %237 = arith.cmpi sgt, %66, %c12_i32 : i32
    %238 = arith.addi %31, %c12_i32 : i32
    %239 = arith.muli %238, %c128_i32 : i32
    %240 = arith.index_cast %239 : i32 to index
    %241 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %240]
    %reinterpret_cast_142 = memref.reinterpret_cast %arg3 to offset: [%241], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %242 = hivm.hir.pointer_cast(%c46624_i64) : memref<16xf32, #hivm.address_space<ub>>
    %243 = arith.index_castui %237 : i1 to index
    %244 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%243]
    %245 = arith.cmpi slt, %244, %c16 : index
    %subview_143 = memref.subview %reinterpret_cast_142[0] [%244] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_144 = memref.subview %242[0] [%244] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %245 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%242 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_143 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_144 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %246 = hivm.hir.pointer_cast(%c46624_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%242, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%246 : memref<16xf32, #hivm.address_space<ub>>)
    %247 = hivm.hir.pointer_cast(%c46624_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%246, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%247 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_145 = memref.expand_shape %247 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %248 = hivm.hir.pointer_cast(%c46688_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %249 = hivm.hir.pointer_cast(%c47712_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_145, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%248 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%249 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %250 = hivm.hir.pointer_cast(%c48224_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %251 = hivm.hir.pointer_cast(%c48288_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%248 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%250 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%251 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_146 = memref.collapse_shape %250 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %252 = hivm.hir.pointer_cast(%c46624_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%247, %collapse_shape_146 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%252 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_147 = memref.expand_shape %252 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %253 = hivm.hir.pointer_cast(%c49312_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_147 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%253 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_148 = memref.subview %253[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_149 = memref.collapse_shape %subview_148 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_140 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_149 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_150 = memref.subview %253[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_151 = memref.subview %61[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_152 = memref.collapse_shape %subview_150 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_153 = memref.collapse_shape %subview_151 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_152 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_153 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>)
    %254 = arith.cmpi sgt, %66, %c13_i32 : i32
    %255 = arith.addi %31, %c13_i32 : i32
    %256 = arith.muli %255, %c128_i32 : i32
    %257 = arith.index_cast %256 : i32 to index
    %258 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %257]
    %reinterpret_cast_154 = memref.reinterpret_cast %arg3 to offset: [%258], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %259 = hivm.hir.pointer_cast(%c50336_i64) : memref<16xf32, #hivm.address_space<ub>>
    %260 = arith.index_castui %254 : i1 to index
    %261 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%260]
    %262 = arith.cmpi slt, %261, %c16 : index
    %subview_155 = memref.subview %reinterpret_cast_154[0] [%261] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_156 = memref.subview %259[0] [%261] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %262 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%259 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_155 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_156 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %263 = hivm.hir.pointer_cast(%c50336_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%259, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%263 : memref<16xf32, #hivm.address_space<ub>>)
    %264 = hivm.hir.pointer_cast(%c50336_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%263, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%264 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_157 = memref.expand_shape %264 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %265 = hivm.hir.pointer_cast(%c50400_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %266 = hivm.hir.pointer_cast(%c51424_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_157, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%265 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%266 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %267 = hivm.hir.pointer_cast(%c51936_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %268 = hivm.hir.pointer_cast(%c52000_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%265 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%267 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%268 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_158 = memref.collapse_shape %267 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %269 = hivm.hir.pointer_cast(%c50336_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%264, %collapse_shape_158 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%269 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_159 = memref.expand_shape %269 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %270 = hivm.hir.pointer_cast(%c53024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_159 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%270 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_160 = memref.subview %270[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_161 = memref.collapse_shape %subview_160 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_152 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_161 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_162 = memref.subview %270[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_163 = memref.subview %61[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_164 = memref.collapse_shape %subview_162 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_165 = memref.collapse_shape %subview_163 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_164 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_165 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>)
    %271 = arith.cmpi sgt, %66, %c14_i32 : i32
    %272 = arith.addi %31, %c14_i32 : i32
    %273 = arith.muli %272, %c128_i32 : i32
    %274 = arith.index_cast %273 : i32 to index
    %275 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %274]
    %reinterpret_cast_166 = memref.reinterpret_cast %arg3 to offset: [%275], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %276 = hivm.hir.pointer_cast(%c54048_i64) : memref<16xf32, #hivm.address_space<ub>>
    %277 = arith.index_castui %271 : i1 to index
    %278 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%277]
    %279 = arith.cmpi slt, %278, %c16 : index
    %subview_167 = memref.subview %reinterpret_cast_166[0] [%278] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_168 = memref.subview %276[0] [%278] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %279 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%276 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_167 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_168 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %280 = hivm.hir.pointer_cast(%c54048_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%276, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%280 : memref<16xf32, #hivm.address_space<ub>>)
    %281 = hivm.hir.pointer_cast(%c54048_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%280, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%281 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_169 = memref.expand_shape %281 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %282 = hivm.hir.pointer_cast(%c54112_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %283 = hivm.hir.pointer_cast(%c55136_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_169, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%282 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%283 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %284 = hivm.hir.pointer_cast(%c55648_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %285 = hivm.hir.pointer_cast(%c55712_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%282 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%284 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%285 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_170 = memref.collapse_shape %284 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %286 = hivm.hir.pointer_cast(%c54048_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%281, %collapse_shape_170 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%286 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_171 = memref.expand_shape %286 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %287 = hivm.hir.pointer_cast(%c56736_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_171 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%287 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_172 = memref.subview %287[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_173 = memref.collapse_shape %subview_172 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_164 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_173 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_174 = memref.subview %287[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_175 = memref.subview %61[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_176 = memref.collapse_shape %subview_174 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_177 = memref.collapse_shape %subview_175 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_176 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_177 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>)
    %288 = arith.cmpi sgt, %66, %c15_i32 : i32
    %289 = arith.addi %31, %c15_i32 : i32
    %290 = arith.muli %289, %c128_i32 : i32
    %291 = arith.index_cast %290 : i32 to index
    %292 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %291]
    %reinterpret_cast_178 = memref.reinterpret_cast %arg3 to offset: [%292], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %293 = hivm.hir.pointer_cast(%c57760_i64) : memref<16xf32, #hivm.address_space<ub>>
    %294 = arith.index_castui %288 : i1 to index
    %295 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%294]
    %296 = arith.cmpi slt, %295, %c16 : index
    %subview_179 = memref.subview %reinterpret_cast_178[0] [%295] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_180 = memref.subview %293[0] [%295] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %296 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%293 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_179 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_180 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %297 = hivm.hir.pointer_cast(%c57760_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%293, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%297 : memref<16xf32, #hivm.address_space<ub>>)
    %298 = hivm.hir.pointer_cast(%c57760_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%297, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%298 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_181 = memref.expand_shape %298 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %299 = hivm.hir.pointer_cast(%c7328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %300 = hivm.hir.pointer_cast(%c57824_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_181, %61 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%299 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%300 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %301 = hivm.hir.pointer_cast(%c58336_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %302 = hivm.hir.pointer_cast(%c58400_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%299 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%301 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%302 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_182 = memref.collapse_shape %301 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %303 = hivm.hir.pointer_cast(%c57760_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%298, %collapse_shape_182 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%303 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_183 = memref.expand_shape %303 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %304 = hivm.hir.pointer_cast(%c59424_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_183 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%304 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_184 = memref.subview %304[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_185 = memref.collapse_shape %subview_184 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_176 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_185 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>)
    %305 = arith.cmpi sgt, %66, %c18_i32 : i32
    %306 = arith.addi %31, %c18_i32 : i32
    %307 = arith.muli %306, %c128_i32 : i32
    %308 = arith.index_cast %307 : i32 to index
    %309 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %308]
    %reinterpret_cast_186 = memref.reinterpret_cast %arg3 to offset: [%309], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %310 = hivm.hir.pointer_cast(%c60448_i64) : memref<16xf32, #hivm.address_space<ub>>
    %311 = arith.index_castui %305 : i1 to index
    %312 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%311]
    %313 = arith.cmpi slt, %312, %c16 : index
    %subview_187 = memref.subview %reinterpret_cast_186[0] [%312] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_188 = memref.subview %310[0] [%312] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %313 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%310 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_187 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_188 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %314 = hivm.hir.pointer_cast(%c60448_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%310, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%314 : memref<16xf32, #hivm.address_space<ub>>)
    %315 = hivm.hir.pointer_cast(%c60448_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%314, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%315 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_189 = memref.expand_shape %315 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %316 = hivm.hir.pointer_cast(%c60512_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %317 = hivm.hir.pointer_cast(%c61536_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_189, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%316 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%317 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %318 = hivm.hir.pointer_cast(%c62048_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %319 = hivm.hir.pointer_cast(%c62112_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%316 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%318 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%319 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_190 = memref.collapse_shape %318 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %320 = hivm.hir.pointer_cast(%c60448_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%315, %collapse_shape_190 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%320 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_191 = memref.expand_shape %320 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %321 = hivm.hir.pointer_cast(%c63136_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_191 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%321 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_192 = memref.subview %65[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_193 = memref.subview %321[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_194 = memref.collapse_shape %subview_192 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_195 = memref.collapse_shape %subview_193 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_194 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_195 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_196 = memref.subview %321[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_197 = memref.subview %65[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_198 = memref.collapse_shape %subview_196 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_199 = memref.collapse_shape %subview_197 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_198 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_199 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>)
    %322 = arith.cmpi sgt, %66, %c19_i32 : i32
    %323 = arith.addi %31, %c19_i32 : i32
    %324 = arith.muli %323, %c128_i32 : i32
    %325 = arith.index_cast %324 : i32 to index
    %326 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %325]
    %reinterpret_cast_200 = memref.reinterpret_cast %arg3 to offset: [%326], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %327 = hivm.hir.pointer_cast(%c64160_i64) : memref<16xf32, #hivm.address_space<ub>>
    %328 = arith.index_castui %322 : i1 to index
    %329 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%328]
    %330 = arith.cmpi slt, %329, %c16 : index
    %subview_201 = memref.subview %reinterpret_cast_200[0] [%329] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_202 = memref.subview %327[0] [%329] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %330 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%327 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_201 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_202 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %331 = hivm.hir.pointer_cast(%c64160_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%327, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%331 : memref<16xf32, #hivm.address_space<ub>>)
    %332 = hivm.hir.pointer_cast(%c64160_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%331, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%332 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_203 = memref.expand_shape %332 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %333 = hivm.hir.pointer_cast(%c64224_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %334 = hivm.hir.pointer_cast(%c65248_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_203, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%333 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%334 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %335 = hivm.hir.pointer_cast(%c65760_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %336 = hivm.hir.pointer_cast(%c65824_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%333 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%335 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%336 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_204 = memref.collapse_shape %335 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %337 = hivm.hir.pointer_cast(%c64160_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%332, %collapse_shape_204 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%337 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_205 = memref.expand_shape %337 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %338 = hivm.hir.pointer_cast(%c66848_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_205 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%338 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_206 = memref.subview %338[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_207 = memref.collapse_shape %subview_206 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_198 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_207 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_208 = memref.subview %338[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_209 = memref.subview %65[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_210 = memref.collapse_shape %subview_208 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_211 = memref.collapse_shape %subview_209 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_210 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_211 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>)
    %339 = arith.cmpi sgt, %66, %c20_i32 : i32
    %340 = arith.addi %31, %c20_i32 : i32
    %341 = arith.muli %340, %c128_i32 : i32
    %342 = arith.index_cast %341 : i32 to index
    %343 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %342]
    %reinterpret_cast_212 = memref.reinterpret_cast %arg3 to offset: [%343], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %344 = hivm.hir.pointer_cast(%c67872_i64) : memref<16xf32, #hivm.address_space<ub>>
    %345 = arith.index_castui %339 : i1 to index
    %346 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%345]
    %347 = arith.cmpi slt, %346, %c16 : index
    %subview_213 = memref.subview %reinterpret_cast_212[0] [%346] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_214 = memref.subview %344[0] [%346] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %347 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%344 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_213 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_214 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %348 = hivm.hir.pointer_cast(%c67872_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%344, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%348 : memref<16xf32, #hivm.address_space<ub>>)
    %349 = hivm.hir.pointer_cast(%c67872_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%348, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%349 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_215 = memref.expand_shape %349 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %350 = hivm.hir.pointer_cast(%c67936_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %351 = hivm.hir.pointer_cast(%c68960_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_215, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%350 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%351 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %352 = hivm.hir.pointer_cast(%c69472_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %353 = hivm.hir.pointer_cast(%c69536_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%350 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%352 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%353 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_216 = memref.collapse_shape %352 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %354 = hivm.hir.pointer_cast(%c69472_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%349, %collapse_shape_216 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%354 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_217 = memref.expand_shape %354 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %355 = hivm.hir.pointer_cast(%c70560_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_217 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%355 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_218 = memref.subview %355[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_219 = memref.collapse_shape %subview_218 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_210 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_219 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_220 = memref.subview %355[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_221 = memref.subview %65[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_222 = memref.collapse_shape %subview_220 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_223 = memref.collapse_shape %subview_221 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_222 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_223 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>)
    %356 = arith.cmpi sgt, %66, %c21_i32 : i32
    %357 = arith.addi %31, %c21_i32 : i32
    %358 = arith.muli %357, %c128_i32 : i32
    %359 = arith.index_cast %358 : i32 to index
    %360 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %359]
    %reinterpret_cast_224 = memref.reinterpret_cast %arg3 to offset: [%360], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %361 = hivm.hir.pointer_cast(%c71584_i64) : memref<16xf32, #hivm.address_space<ub>>
    %362 = arith.index_castui %356 : i1 to index
    %363 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%362]
    %364 = arith.cmpi slt, %363, %c16 : index
    %subview_225 = memref.subview %reinterpret_cast_224[0] [%363] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_226 = memref.subview %361[0] [%363] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %364 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%361 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_225 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_226 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %365 = hivm.hir.pointer_cast(%c71584_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%361, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%365 : memref<16xf32, #hivm.address_space<ub>>)
    %366 = hivm.hir.pointer_cast(%c71584_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%365, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%366 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_227 = memref.expand_shape %366 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %367 = hivm.hir.pointer_cast(%c71648_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %368 = hivm.hir.pointer_cast(%c72672_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_227, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%367 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%368 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %369 = hivm.hir.pointer_cast(%c73184_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %370 = hivm.hir.pointer_cast(%c73248_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%367 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%369 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%370 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_228 = memref.collapse_shape %369 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %371 = hivm.hir.pointer_cast(%c73184_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%366, %collapse_shape_228 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%371 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_229 = memref.expand_shape %371 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %372 = hivm.hir.pointer_cast(%c74272_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_229 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%372 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_230 = memref.subview %372[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_231 = memref.collapse_shape %subview_230 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_222 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_231 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_232 = memref.subview %372[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_233 = memref.subview %65[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_234 = memref.collapse_shape %subview_232 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_235 = memref.collapse_shape %subview_233 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_234 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_235 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>)
    %373 = arith.cmpi sgt, %66, %c22_i32 : i32
    %374 = arith.addi %31, %c22_i32 : i32
    %375 = arith.muli %374, %c128_i32 : i32
    %376 = arith.index_cast %375 : i32 to index
    %377 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %376]
    %reinterpret_cast_236 = memref.reinterpret_cast %arg3 to offset: [%377], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %378 = hivm.hir.pointer_cast(%c75296_i64) : memref<16xf32, #hivm.address_space<ub>>
    %379 = arith.index_castui %373 : i1 to index
    %380 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%379]
    %381 = arith.cmpi slt, %380, %c16 : index
    %subview_237 = memref.subview %reinterpret_cast_236[0] [%380] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_238 = memref.subview %378[0] [%380] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %381 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%378 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_237 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_238 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %382 = hivm.hir.pointer_cast(%c75296_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%378, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%382 : memref<16xf32, #hivm.address_space<ub>>)
    %383 = hivm.hir.pointer_cast(%c75296_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%382, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%383 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_239 = memref.expand_shape %383 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %384 = hivm.hir.pointer_cast(%c75360_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %385 = hivm.hir.pointer_cast(%c76384_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_239, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%384 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%385 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %386 = hivm.hir.pointer_cast(%c76896_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %387 = hivm.hir.pointer_cast(%c76960_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%384 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%386 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%387 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_240 = memref.collapse_shape %386 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %388 = hivm.hir.pointer_cast(%c75296_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%383, %collapse_shape_240 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%388 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_241 = memref.expand_shape %388 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %389 = hivm.hir.pointer_cast(%c77984_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_241 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%389 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_242 = memref.subview %389[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_243 = memref.collapse_shape %subview_242 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_234 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_243 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_244 = memref.subview %389[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_245 = memref.subview %65[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_246 = memref.collapse_shape %subview_244 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_247 = memref.collapse_shape %subview_245 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_246 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_247 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>)
    %390 = arith.cmpi sgt, %66, %c23_i32 : i32
    %391 = arith.addi %31, %c23_i32 : i32
    %392 = arith.muli %391, %c128_i32 : i32
    %393 = arith.index_cast %392 : i32 to index
    %394 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %393]
    %reinterpret_cast_248 = memref.reinterpret_cast %arg3 to offset: [%394], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %395 = hivm.hir.pointer_cast(%c79008_i64) : memref<16xf32, #hivm.address_space<ub>>
    %396 = arith.index_castui %390 : i1 to index
    %397 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%396]
    %398 = arith.cmpi slt, %397, %c16 : index
    %subview_249 = memref.subview %reinterpret_cast_248[0] [%397] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_250 = memref.subview %395[0] [%397] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %398 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%395 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_249 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_250 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %399 = hivm.hir.pointer_cast(%c79008_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%395, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%399 : memref<16xf32, #hivm.address_space<ub>>)
    %400 = hivm.hir.pointer_cast(%c79008_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%399, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%400 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_251 = memref.expand_shape %400 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %401 = hivm.hir.pointer_cast(%c79072_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %402 = hivm.hir.pointer_cast(%c80096_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_251, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%401 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%402 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %403 = hivm.hir.pointer_cast(%c80608_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %404 = hivm.hir.pointer_cast(%c80672_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%401 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%403 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%404 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_252 = memref.collapse_shape %403 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %405 = hivm.hir.pointer_cast(%c80608_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%400, %collapse_shape_252 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%405 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_253 = memref.expand_shape %405 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %406 = hivm.hir.pointer_cast(%c81696_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_253 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%406 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_254 = memref.subview %406[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_255 = memref.collapse_shape %subview_254 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_246 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_255 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_256 = memref.subview %406[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_257 = memref.subview %65[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_258 = memref.collapse_shape %subview_256 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_259 = memref.collapse_shape %subview_257 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_258 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_259 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>)
    %407 = arith.cmpi sgt, %66, %c24_i32 : i32
    %408 = arith.addi %31, %c24_i32 : i32
    %409 = arith.muli %408, %c128_i32 : i32
    %410 = arith.index_cast %409 : i32 to index
    %411 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %410]
    %reinterpret_cast_260 = memref.reinterpret_cast %arg3 to offset: [%411], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %412 = hivm.hir.pointer_cast(%c82720_i64) : memref<16xf32, #hivm.address_space<ub>>
    %413 = arith.index_castui %407 : i1 to index
    %414 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%413]
    %415 = arith.cmpi slt, %414, %c16 : index
    %subview_261 = memref.subview %reinterpret_cast_260[0] [%414] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_262 = memref.subview %412[0] [%414] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %415 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%412 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_261 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_262 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %416 = hivm.hir.pointer_cast(%c82720_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%412, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%416 : memref<16xf32, #hivm.address_space<ub>>)
    %417 = hivm.hir.pointer_cast(%c82720_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%416, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%417 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_263 = memref.expand_shape %417 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %418 = hivm.hir.pointer_cast(%c82784_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %419 = hivm.hir.pointer_cast(%c83808_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_263, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%418 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%419 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %420 = hivm.hir.pointer_cast(%c84320_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %421 = hivm.hir.pointer_cast(%c84384_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%418 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%420 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%421 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_264 = memref.collapse_shape %420 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %422 = hivm.hir.pointer_cast(%c84320_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%417, %collapse_shape_264 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%422 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_265 = memref.expand_shape %422 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %423 = hivm.hir.pointer_cast(%c85408_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_265 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%423 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_266 = memref.subview %423[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_267 = memref.collapse_shape %subview_266 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_258 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_267 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_268 = memref.subview %423[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_269 = memref.subview %65[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_270 = memref.collapse_shape %subview_268 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_271 = memref.collapse_shape %subview_269 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_270 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_271 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>)
    %424 = arith.cmpi sgt, %66, %c25_i32 : i32
    %425 = arith.addi %31, %c25_i32 : i32
    %426 = arith.muli %425, %c128_i32 : i32
    %427 = arith.index_cast %426 : i32 to index
    %428 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %427]
    %reinterpret_cast_272 = memref.reinterpret_cast %arg3 to offset: [%428], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %429 = hivm.hir.pointer_cast(%c86432_i64) : memref<16xf32, #hivm.address_space<ub>>
    %430 = arith.index_castui %424 : i1 to index
    %431 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%430]
    %432 = arith.cmpi slt, %431, %c16 : index
    %subview_273 = memref.subview %reinterpret_cast_272[0] [%431] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_274 = memref.subview %429[0] [%431] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %432 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%429 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_273 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_274 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %433 = hivm.hir.pointer_cast(%c86432_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%429, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%433 : memref<16xf32, #hivm.address_space<ub>>)
    %434 = hivm.hir.pointer_cast(%c86432_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%433, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%434 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_275 = memref.expand_shape %434 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %435 = hivm.hir.pointer_cast(%c86496_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %436 = hivm.hir.pointer_cast(%c87520_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_275, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%435 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%436 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %437 = hivm.hir.pointer_cast(%c88032_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %438 = hivm.hir.pointer_cast(%c88096_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%435 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%437 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%438 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_276 = memref.collapse_shape %437 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %439 = hivm.hir.pointer_cast(%c86432_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%434, %collapse_shape_276 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%439 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_277 = memref.expand_shape %439 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %440 = hivm.hir.pointer_cast(%c89120_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_277 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%440 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_278 = memref.subview %440[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_279 = memref.collapse_shape %subview_278 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_270 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_279 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_280 = memref.subview %440[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_281 = memref.subview %65[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_282 = memref.collapse_shape %subview_280 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_283 = memref.collapse_shape %subview_281 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_282 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_283 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>)
    %441 = arith.cmpi sgt, %66, %c26_i32 : i32
    %442 = arith.addi %31, %c26_i32 : i32
    %443 = arith.muli %442, %c128_i32 : i32
    %444 = arith.index_cast %443 : i32 to index
    %445 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %444]
    %reinterpret_cast_284 = memref.reinterpret_cast %arg3 to offset: [%445], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %446 = hivm.hir.pointer_cast(%c90144_i64) : memref<16xf32, #hivm.address_space<ub>>
    %447 = arith.index_castui %441 : i1 to index
    %448 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%447]
    %449 = arith.cmpi slt, %448, %c16 : index
    %subview_285 = memref.subview %reinterpret_cast_284[0] [%448] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_286 = memref.subview %446[0] [%448] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %449 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%446 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_285 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_286 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %450 = hivm.hir.pointer_cast(%c90144_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%446, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%450 : memref<16xf32, #hivm.address_space<ub>>)
    %451 = hivm.hir.pointer_cast(%c90144_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%450, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%451 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_287 = memref.expand_shape %451 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %452 = hivm.hir.pointer_cast(%c90208_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %453 = hivm.hir.pointer_cast(%c91232_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_287, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%452 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%453 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %454 = hivm.hir.pointer_cast(%c91744_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %455 = hivm.hir.pointer_cast(%c91808_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%452 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%454 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%455 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_288 = memref.collapse_shape %454 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %456 = hivm.hir.pointer_cast(%c90144_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%451, %collapse_shape_288 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%456 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_289 = memref.expand_shape %456 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %457 = hivm.hir.pointer_cast(%c92832_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_289 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%457 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_290 = memref.subview %457[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_291 = memref.collapse_shape %subview_290 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_282 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_291 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_292 = memref.subview %457[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_293 = memref.subview %65[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_294 = memref.collapse_shape %subview_292 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_295 = memref.collapse_shape %subview_293 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_294 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_295 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>)
    %458 = arith.cmpi sgt, %66, %c27_i32 : i32
    %459 = arith.addi %31, %c27_i32 : i32
    %460 = arith.muli %459, %c128_i32 : i32
    %461 = arith.index_cast %460 : i32 to index
    %462 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %461]
    %reinterpret_cast_296 = memref.reinterpret_cast %arg3 to offset: [%462], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %463 = hivm.hir.pointer_cast(%c93856_i64) : memref<16xf32, #hivm.address_space<ub>>
    %464 = arith.index_castui %458 : i1 to index
    %465 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%464]
    %466 = arith.cmpi slt, %465, %c16 : index
    %subview_297 = memref.subview %reinterpret_cast_296[0] [%465] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_298 = memref.subview %463[0] [%465] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %466 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%463 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_297 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_298 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %467 = hivm.hir.pointer_cast(%c93856_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%463, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%467 : memref<16xf32, #hivm.address_space<ub>>)
    %468 = hivm.hir.pointer_cast(%c93856_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%467, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%468 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_299 = memref.expand_shape %468 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %469 = hivm.hir.pointer_cast(%c93920_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %470 = hivm.hir.pointer_cast(%c94944_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_299, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%469 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%470 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %471 = hivm.hir.pointer_cast(%c95456_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %472 = hivm.hir.pointer_cast(%c95520_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%469 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%471 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%472 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_300 = memref.collapse_shape %471 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %473 = hivm.hir.pointer_cast(%c95456_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%468, %collapse_shape_300 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%473 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_301 = memref.expand_shape %473 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %474 = hivm.hir.pointer_cast(%c96544_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_301 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%474 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_302 = memref.subview %474[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_303 = memref.collapse_shape %subview_302 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_294 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_303 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_304 = memref.subview %474[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_305 = memref.subview %65[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_306 = memref.collapse_shape %subview_304 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_307 = memref.collapse_shape %subview_305 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_306 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_307 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>)
    %475 = arith.cmpi sgt, %66, %c28_i32 : i32
    %476 = arith.addi %31, %c28_i32 : i32
    %477 = arith.muli %476, %c128_i32 : i32
    %478 = arith.index_cast %477 : i32 to index
    %479 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %478]
    %reinterpret_cast_308 = memref.reinterpret_cast %arg3 to offset: [%479], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %480 = hivm.hir.pointer_cast(%c97568_i64) : memref<16xf32, #hivm.address_space<ub>>
    %481 = arith.index_castui %475 : i1 to index
    %482 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%481]
    %483 = arith.cmpi slt, %482, %c16 : index
    %subview_309 = memref.subview %reinterpret_cast_308[0] [%482] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_310 = memref.subview %480[0] [%482] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %483 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%480 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_309 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_310 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %484 = hivm.hir.pointer_cast(%c97568_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%480, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%484 : memref<16xf32, #hivm.address_space<ub>>)
    %485 = hivm.hir.pointer_cast(%c97568_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%484, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%485 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_311 = memref.expand_shape %485 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %486 = hivm.hir.pointer_cast(%c97632_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %487 = hivm.hir.pointer_cast(%c98656_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_311, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%486 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%487 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %488 = hivm.hir.pointer_cast(%c99168_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %489 = hivm.hir.pointer_cast(%c99232_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%486 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%488 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%489 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_312 = memref.collapse_shape %488 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %490 = hivm.hir.pointer_cast(%c97568_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%485, %collapse_shape_312 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%490 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_313 = memref.expand_shape %490 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %491 = hivm.hir.pointer_cast(%c100256_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_313 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%491 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_314 = memref.subview %491[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_315 = memref.collapse_shape %subview_314 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_306 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_315 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_316 = memref.subview %491[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_317 = memref.subview %65[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_318 = memref.collapse_shape %subview_316 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_319 = memref.collapse_shape %subview_317 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_318 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_319 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>)
    %492 = arith.cmpi sgt, %66, %c29_i32 : i32
    %493 = arith.addi %31, %c29_i32 : i32
    %494 = arith.muli %493, %c128_i32 : i32
    %495 = arith.index_cast %494 : i32 to index
    %496 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %495]
    %reinterpret_cast_320 = memref.reinterpret_cast %arg3 to offset: [%496], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %497 = hivm.hir.pointer_cast(%c101280_i64) : memref<16xf32, #hivm.address_space<ub>>
    %498 = arith.index_castui %492 : i1 to index
    %499 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%498]
    %500 = arith.cmpi slt, %499, %c16 : index
    %subview_321 = memref.subview %reinterpret_cast_320[0] [%499] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_322 = memref.subview %497[0] [%499] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %500 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%497 : memref<16xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.load ins(%subview_321 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_322 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %501 = hivm.hir.pointer_cast(%c101280_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%497, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%501 : memref<16xf32, #hivm.address_space<ub>>)
    %502 = hivm.hir.pointer_cast(%c101280_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%501, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%502 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_323 = memref.expand_shape %502 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %503 = hivm.hir.pointer_cast(%c101344_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %504 = hivm.hir.pointer_cast(%c102368_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_323, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%503 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%504 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %505 = hivm.hir.pointer_cast(%c102880_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %506 = hivm.hir.pointer_cast(%c102944_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%503 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%505 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%506 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_324 = memref.collapse_shape %505 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %507 = hivm.hir.pointer_cast(%c102880_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%502, %collapse_shape_324 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%507 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_325 = memref.expand_shape %507 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %508 = hivm.hir.pointer_cast(%c103968_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_325 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%508 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_326 = memref.subview %508[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_327 = memref.collapse_shape %subview_326 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_318 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_327 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_328 = memref.subview %508[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_329 = memref.subview %65[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_330 = memref.collapse_shape %subview_328 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_331 = memref.collapse_shape %subview_329 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_330 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_331 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>)
    %509 = arith.cmpi sgt, %66, %c30_i32 : i32
    %510 = arith.addi %31, %c30_i32 : i32
    %511 = arith.muli %510, %c128_i32 : i32
    %512 = arith.index_cast %511 : i32 to index
    %513 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %512]
    %reinterpret_cast_332 = memref.reinterpret_cast %arg3 to offset: [%513], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %514 = hivm.hir.pointer_cast(%c104992_i64) : memref<16xf32, #hivm.address_space<ub>>
    %515 = arith.index_castui %509 : i1 to index
    %516 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%515]
    %517 = arith.cmpi slt, %516, %c16 : index
    %subview_333 = memref.subview %reinterpret_cast_332[0] [%516] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_334 = memref.subview %514[0] [%516] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %517 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%514 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_333 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_334 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %518 = hivm.hir.pointer_cast(%c104992_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%514, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%518 : memref<16xf32, #hivm.address_space<ub>>)
    %519 = hivm.hir.pointer_cast(%c104992_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%518, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%519 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_335 = memref.expand_shape %519 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %520 = hivm.hir.pointer_cast(%c105056_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %521 = hivm.hir.pointer_cast(%c106080_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_335, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%520 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%521 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %522 = hivm.hir.pointer_cast(%c106592_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %523 = hivm.hir.pointer_cast(%c106656_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%520 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%522 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%523 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_336 = memref.collapse_shape %522 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %524 = hivm.hir.pointer_cast(%c104992_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%519, %collapse_shape_336 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%524 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_337 = memref.expand_shape %524 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %525 = hivm.hir.pointer_cast(%c107680_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_337 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%525 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_338 = memref.subview %525[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_339 = memref.collapse_shape %subview_338 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_330 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_339 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_340 = memref.subview %525[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_341 = memref.subview %65[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_342 = memref.collapse_shape %subview_340 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_343 = memref.collapse_shape %subview_341 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_342 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_343 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>)
    %526 = arith.cmpi sgt, %66, %c31_i32 : i32
    %527 = arith.addi %31, %c31_i32 : i32
    %528 = arith.muli %527, %c128_i32 : i32
    %529 = arith.index_cast %528 : i32 to index
    %530 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %529]
    %reinterpret_cast_344 = memref.reinterpret_cast %arg3 to offset: [%530], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %531 = hivm.hir.pointer_cast(%c108704_i64) : memref<16xf32, #hivm.address_space<ub>>
    %532 = arith.index_castui %526 : i1 to index
    %533 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%532]
    %534 = arith.cmpi slt, %533, %c16 : index
    %subview_345 = memref.subview %reinterpret_cast_344[0] [%533] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_346 = memref.subview %531[0] [%533] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %534 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%531 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_345 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_346 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %535 = hivm.hir.pointer_cast(%c108704_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%531, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%535 : memref<16xf32, #hivm.address_space<ub>>)
    %536 = hivm.hir.pointer_cast(%c108704_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%535, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%536 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_347 = memref.expand_shape %536 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %537 = hivm.hir.pointer_cast(%c8416_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %538 = hivm.hir.pointer_cast(%c108768_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_347, %65 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%537 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%538 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %539 = hivm.hir.pointer_cast(%c109280_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %540 = hivm.hir.pointer_cast(%c109344_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%537 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%539 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%540 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_348 = memref.collapse_shape %539 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %541 = hivm.hir.pointer_cast(%c109280_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%536, %collapse_shape_348 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%541 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_349 = memref.expand_shape %541 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %542 = hivm.hir.pointer_cast(%c110368_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_349 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%542 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_350 = memref.subview %542[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_351 = memref.collapse_shape %subview_350 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_342 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_351 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>)
    %543 = hivm.hir.pointer_cast(%c59424_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_352 = memref.collapse_shape %304 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_353 = memref.collapse_shape %543 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_352, %collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_353 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %544 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %19 {
      %collapse_shape_369 = memref.collapse_shape %544 [[0, 1]] : memref<16x16xf32, #hivm.address_space<gm>> into memref<256xf32, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_353 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_369 : memref<256xf32, #hivm.address_space<gm>>) {"inserted-store"}
    }
    annotation.mark %544 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %545 = hivm.hir.pointer_cast(%c110368_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_354 = memref.collapse_shape %542 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_355 = memref.collapse_shape %545 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_354, %collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_355 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %546 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %19 {
      %collapse_shape_369 = memref.collapse_shape %546 [[0, 1]] : memref<16x16xf32, #hivm.address_space<gm>> into memref<256xf32, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_355 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_369 : memref<256xf32, #hivm.address_space<gm>>) {"inserted-store"}
    }
    annotation.mark %546 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %547 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %548 = hivm.hir.pointer_cast(%c111392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_356 = memref.collapse_shape %547 [[0, 1]] : memref<16x16xf32, #hivm.address_space<gm>> into memref<256xf32, #hivm.address_space<gm>>
    %collapse_shape_357 = memref.collapse_shape %548 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_356 : memref<256xf32, #hivm.address_space<gm>>) outs(%collapse_shape_357 : memref<256xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    %549 = hivm.hir.pointer_cast(%c111392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_358 = memref.collapse_shape %549 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_357, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_358 : memref<256xf32, #hivm.address_space<ub>>)
    %550 = hivm.hir.pointer_cast(%c111392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_359 = memref.collapse_shape %550 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_358, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_359 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_360 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_361 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_362 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_363 = memref.subview %543[%46, 0] [%47, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_364 = memref.subview %reinterpret_cast_360[0, 0] [%47, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    scf.if %19 {
      hivm.hir.store ins(%subview_363 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_364 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_365 = memref.subview %545[%55, 0] [%56, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_366 = memref.subview %reinterpret_cast_362[0, 0] [%56, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    scf.if %19 {
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_365 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_366 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
    %subview_367 = memref.subview %550[%55, 0] [%56, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_368 = memref.subview %reinterpret_cast_361[0, 0] [%56, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %19 {
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_367 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_368 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
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
