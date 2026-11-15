// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_state_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<35xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c24_i32 = arith.constant 24 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c15_i32 = arith.constant 15 : i32
  %c0_i64_0 = arith.constant 0 : i64
  %c16_i32 = arith.constant 16 : i32
  %c1 = arith.constant 1 : index
  %c64_i64 = arith.constant 64 : i64
  %c4096 = arith.constant 4096 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg32, %arg33 : i32
  %1 = arith.muli %0, %arg34 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  %4 = arith.muli %arg34, %arg33 : i32
  %5 = arith.extsi %arg11 : i32 to i64
  %6 = arith.addi %arg9, %c15_i32 : i32
  %7 = arith.divsi %6, %c16_i32 : i32
  %8 = arith.extsi %arg10 : i32 to i64
  %9 = arith.extsi %arg12 : i32 to i64
  %10 = arith.extsi %arg20 : i32 to i64
  %11 = arith.extsi %arg21 : i32 to i64
  %12 = arith.index_cast %arg23 : i32 to index
  %13 = arith.index_cast %arg8 : i32 to index
  %14 = arith.index_cast %arg9 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg35 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %15 = arith.remsi %arg35, %arg34 : i32
    %16 = arith.divsi %arg35, %arg34 : i32
    %17 = arith.remsi %16, %arg33 : i32
    %18 = arith.divsi %arg35, %4 : i32
    %19 = arith.remsi %18, %arg32 : i32
    %20 = arith.extsi %17 : i32 to i64
    %21 = arith.divsi %20, %5 : i64
    %22 = arith.muli %21, %5 : i64
    %23 = arith.subi %20, %22 : i64
    %24 = arith.divsi %19, %7 : i32
    %25 = arith.remsi %19, %7 : i32
    %26 = arith.muli %21, %8 : i64
    %27 = arith.muli %24, %c16_i32 : i32
    %28 = arith.muli %25, %c16_i32 : i32
    %29 = arith.subi %9, %26 : i64
    %30 = arith.minsi %8, %29 : i64
    %31 = arith.index_cast %27 : i32 to index
    %32 = arith.index_cast %28 : i32 to index
    %33 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
    %34 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
    %35 = arith.index_cast %30 : i64 to index
    %36 = hivm.hir.pointer_cast(%c0_i64_0) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %36 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg36 = %c0_i64_0 to %30 step %c64_i64  : i64 {
      %55 = arith.index_cast %arg36 : i64 to index
      %56 = affine.min affine_map<(d0, d1) -> ((d0 - d1) ceildiv 16, 4)>(%35, %55)
      scf.for %arg37 = %c0 to %56 step %c1 {
        %57 = arith.index_cast %arg36 : i64 to index
        %58 = arith.index_cast %c0_i64_0 : i64 to index
        %59 = arith.index_cast %30 : i64 to index
        %60 = arith.index_cast %c64_i64 : i64 to index
        %61 = arith.index_cast %arg35 : i32 to index
        %62 = arith.index_cast %3 : i32 to index
        %63 = arith.index_cast %1 : i32 to index
        %64 = arith.index_cast %c24_i32 : i32 to index
        %65 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8] -> (s0 + ((s2 - s3) floordiv s5) * s1 + ((s6 - s7) floordiv s8) * (s1 * ((-s3 + s4 + s5 - 1) floordiv s5)))>()[%arg37, %56, %57, %58, %59, %60, %61, %62, %64]
        %66 = arith.index_cast %65 : index to i64
        %67 = arith.index_cast %arg36 : i64 to index
        %68 = arith.index_cast %c0_i64_0 : i64 to index
        %69 = arith.index_cast %30 : i64 to index
        %70 = arith.index_cast %c64_i64 : i64 to index
        %71 = arith.index_cast %arg35 : i32 to index
        %72 = arith.index_cast %3 : i32 to index
        %73 = arith.index_cast %1 : i32 to index
        %74 = arith.index_cast %c24_i32 : i32 to index
        %75 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8] -> ((s0 + ((s2 - s3) floordiv s5) * s1 + ((s6 - s7) floordiv s8) * (s1 * ((-s3 + s4 + s5 - 1) floordiv s5))) mod 2)>()[%arg37, %56, %67, %68, %69, %70, %71, %72, %74]
        %76 = arith.index_cast %75 : index to i1
        %c2_i64 = arith.constant 2 : i64
        %c3_i64 = arith.constant 3 : i64
        %77 = arith.select %76, %c2_i64, %c3_i64 : i64
        %78 = arith.index_cast %75 : index to i1
        %c0_i64_3 = arith.constant 0 : i64
        %c1_i64_4 = arith.constant 1 : i64
        %79 = arith.select %78, %c0_i64_3, %c1_i64_4 : i64
        %80 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %80 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
        %81 = hivm.hir.pointer_cast(%c0_i64_0, %c2048_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
        annotation.mark %81 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
        %82 = arith.index_cast %arg37 : index to i64
        %83 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg37]
        %84 = arith.index_cast %83 : index to i64
        %85 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%arg37)[%55]
        %86 = arith.index_cast %85 : index to i64
        %subview_5 = memref.subview %33[%arg37, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
        %cast_6 = memref.cast %81 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %79]
        hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %84
        %subview_7 = memref.subview %34[%arg37, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
        %cast_8 = memref.cast %80 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %77]
        hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %82
        %87 = arith.cmpi eq, %86, %c0_i64_0 : i64
        %c-1_i64 = arith.constant -1 : i64
        hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_6, %cast_8, %87, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %79, %77, %66, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
    }
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %37 = arith.muli %23, %10 : i64
    %38 = arith.muli %21, %11 : i64
    %39 = arith.addi %37, %38 : i64
    %40 = arith.muli %15, %arg22 : i32
    %41 = arith.extsi %40 : i32 to i64
    %42 = arith.addi %39, %41 : i64
    %43 = arith.index_cast %42 : i64 to index
    %44 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%32, %43, %31, %12]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [16, 16], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %45 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %13]
    %46 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%45, %31]
    %47 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%32, %14]
    %48 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%47, %32]
    %49 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%46, %31]
    %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
    %51 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%48, %32]
    %52 = affine.max affine_map<()[s0] -> (0, s0)>()[%51]
    %53 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%50]
    %54 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%52]
    %subview = memref.subview %36[0, 0, 0, 0] [%54, %53, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>>
    %cast_1 = memref.cast %subview : memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_2 = memref.subview %reinterpret_cast[0, 0] [%50, %52] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_1 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_state_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<35xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c2176_i64 = arith.constant 2176 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c24_i32 = arith.constant 24 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant -1.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c15_i32 = arith.constant 15 : i32
  %c0_i64 = arith.constant 0 : i64
  %c16_i32 = arith.constant 16 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c64_i64 = arith.constant 64 : i64
  %c4096 = arith.constant 4096 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg32, %arg33 : i32
  %1 = arith.muli %0, %arg34 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg34, %arg33 : i32
  %5 = arith.extsi %arg11 : i32 to i64
  %6 = arith.addi %arg9, %c15_i32 : i32
  %7 = arith.divsi %6, %c16_i32 : i32
  %8 = arith.extsi %arg17 : i32 to i64
  %9 = arith.extsi %arg10 : i32 to i64
  %10 = arith.extsi %arg18 : i32 to i64
  %11 = arith.extsi %arg14 : i32 to i64
  %12 = arith.extsi %arg15 : i32 to i64
  %13 = arith.extsi %arg24 : i32 to i64
  %14 = arith.extsi %arg25 : i32 to i64
  %15 = arith.extsi %arg27 : i32 to i64
  %16 = arith.extsi %arg28 : i32 to i64
  %17 = arith.subi %arg10, %c1_i32 : i32
  %18 = arith.index_cast %17 : i32 to index
  %19 = arith.extsi %arg12 : i32 to i64
  %20 = arith.muli %arg15, %c16_i32 : i32
  %21 = arith.muli %arg18, %c16_i32 : i32
  %22 = arith.index_cast %arg15 : i32 to index
  %23 = arith.index_cast %arg18 : i32 to index
  %24 = arith.index_cast %arg8 : i32 to index
  %25 = hivm.hir.get_sub_block_idx -> i64
  %26 = arith.index_cast %25 : i64 to index
  %27 = arith.cmpi eq, %26, %c0 : index
  %28 = arith.index_cast %arg9 : i32 to index
  %29 = arith.index_cast %20 : i32 to index
  %30 = arith.index_cast %21 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg35 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %31 = arith.remsi %arg35, %arg34 : i32
    %32 = arith.divsi %arg35, %arg34 : i32
    %33 = arith.remsi %32, %arg33 : i32
    %34 = arith.divsi %arg35, %4 : i32
    %35 = arith.remsi %34, %arg32 : i32
    %36 = arith.extsi %33 : i32 to i64
    %37 = arith.divsi %36, %5 : i64
    %38 = arith.muli %37, %5 : i64
    %39 = arith.subi %36, %38 : i64
    %40 = arith.divsi %35, %7 : i32
    %41 = arith.remsi %35, %7 : i32
    %42 = arith.muli %39, %8 : i64
    %43 = arith.muli %37, %9 : i64
    %44 = arith.muli %43, %10 : i64
    %45 = arith.addi %42, %44 : i64
    %46 = arith.divsi %31, %arg13 : i32
    %47 = arith.muli %46, %arg19 : i32
    %48 = arith.extsi %47 : i32 to i64
    %49 = arith.addi %45, %48 : i64
    %50 = arith.muli %39, %11 : i64
    %51 = arith.muli %43, %12 : i64
    %52 = arith.addi %50, %51 : i64
    %53 = arith.muli %31, %arg16 : i32
    %54 = arith.extsi %53 : i32 to i64
    %55 = arith.addi %52, %54 : i64
    %56 = arith.muli %39, %13 : i64
    %57 = arith.muli %37, %14 : i64
    %58 = arith.addi %56, %57 : i64
    %59 = arith.muli %31, %arg26 : i32
    %60 = arith.extsi %59 : i32 to i64
    %61 = arith.addi %58, %60 : i64
    %62 = arith.muli %39, %15 : i64
    %63 = arith.muli %37, %16 : i64
    %64 = arith.addi %62, %63 : i64
    %65 = arith.muli %31, %arg29 : i32
    %66 = arith.extsi %65 : i32 to i64
    %67 = arith.addi %64, %66 : i64
    %68 = arith.muli %40, %c16_i32 : i32
    %69 = arith.muli %41, %c16_i32 : i32
    %70 = arith.index_cast %67 : i64 to index
    %71 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%70, %18]
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%71], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %72 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %73 = arith.subi %19, %43 : i64
    %74 = arith.minsi %9, %73 : i64
    %75 = arith.index_cast %68 : i32 to index
    %76 = arith.index_cast %69 : i32 to index
    %77 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
    %78 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
    %79 = arith.index_cast %74 : i64 to index
    %80 = arith.index_cast %55 : i64 to index
    %81 = arith.index_cast %49 : i64 to index
    %82 = arith.index_cast %61 : i64 to index
    %83 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%75, %24]
    %84 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%83, %75]
    %85 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%84, %75]
    %86 = affine.max affine_map<()[s0] -> (0, s0)>()[%85]
    %87 = arith.cmpi slt, %86, %c16 : index
    %88 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%76, %28]
    %89 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%88, %76]
    %90 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%89, %76]
    %91 = affine.max affine_map<()[s0] -> (0, s0)>()[%90]
    %92 = arith.cmpi slt, %91, %c16 : index
    %93:4 = scf.for %arg36 = %c0_i64 to %74 step %c64_i64 iter_args(%arg37 = %75, %arg38 = %c0, %arg39 = %c0, %arg40 = %c0) -> (index, index, index, index)  : i64 {
      %94 = arith.index_cast %arg36 : i64 to index
      %95 = affine.min affine_map<(d0, d1) -> ((d0 - d1) ceildiv 16, 4)>(%79, %94)
      %96:4 = scf.for %arg41 = %c0 to %95 step %c1 iter_args(%arg42 = %arg37, %arg43 = %arg40, %arg44 = %arg39, %arg45 = %arg38) -> (index, index, index, index) {
        %97 = arith.index_cast %arg41 : index to i64
        %98 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg41]
        %99 = arith.index_cast %98 : index to i64
        %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%arg41)[%94]
        %101 = arith.index_cast %100 : index to i64
        %102 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%80, %arg42]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%102], sizes: [16, 16], strides: [1, %22] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
        %103 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%76, %81, %arg45]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%103], sizes: [16, 16], strides: [%23, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %104 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%82, %arg44]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%104], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %105 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%70, %arg43]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%105], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %106 = arith.subi %74, %101 : i64
        %107 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %108 = arith.index_cast %106 : i64 to index
        %109 = affine.max affine_map<()[s0] -> (0, s0)>()[%108]
        %110 = affine.min affine_map<()[s0] -> (16, s0)>()[%109]
        %111 = affine.max affine_map<()[s0] -> (0, s0)>()[%110]
        %112 = arith.cmpi slt, %111, %c16 : index
        %113 = arith.ori %87, %112 : i1
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%86, %111] [1, 1] : memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %107[0, 0] [%86, %111] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        scf.if %113 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape_15 = memref.collapse_shape %107 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%subview : memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = true
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        annotation.mark %107 {MayImplicitTransposeWithLastAxis} : memref<16x16xf32, #hivm.address_space<ub>>
        %subview_6 = memref.subview %77[%arg41, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %99
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        scf.if %27 {
          %collapse_shape_15 = memref.collapse_shape %107 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %collapse_shape_16 = memref.collapse_shape %collapse_shape [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        } {limit_sub_block_id0}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
        annotation.mark %107 {MayImplicitTransposeWithLastAxis} : memref<16x16xf32, #hivm.address_space<ub>>
        %114 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %115 = arith.ori %112, %92 : i1
        %subview_7 = memref.subview %reinterpret_cast_2[0, 0] [%111, %91] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_8 = memref.subview %114[0, 0] [%111, %91] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        scf.if %115 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape_15 = memref.collapse_shape %114 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %116 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
        %117 = arith.cmpi slt, %110, %c16 : index
        %subview_9 = memref.subview %reinterpret_cast_4[0] [%110] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %116[0] [%110] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %117 {
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%116 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
        %118 = hivm.hir.pointer_cast(%c2112_i64) : memref<16xf32, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_3[0] [%110] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_12 = memref.subview %118[0] [%110] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %117 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%118 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %119 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
        hivm.hir.vmul ins(%116, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%119 : memref<16xf32, #hivm.address_space<ub>>)
        %120 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%119, %72 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%120 : memref<16xf32, #hivm.address_space<ub>>)
        %121 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vexp ins(%120 : memref<16xf32, #hivm.address_space<ub>>) outs(%121 : memref<16xf32, #hivm.address_space<ub>>)
        %122 = hivm.hir.pointer_cast(%c2112_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%121, %118 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%122 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape = memref.expand_shape %122 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %123 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %124 = hivm.hir.pointer_cast(%c2176_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%114, %expand_shape : memref<16x16xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%123 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%124 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_13 = memref.subview %78[%arg41, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %97
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        scf.if %27 {
          %collapse_shape_15 = memref.collapse_shape %123 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %collapse_shape_16 = memref.collapse_shape %collapse_shape_14 [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        } {limit_sub_block_id0}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
        %125 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg42, %29]
        %126 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg45, %30]
        %127 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%arg44]
        %128 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%arg43]
        scf.yield %125, %128, %127, %126 : index, index, index, index
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      scf.yield %96#0, %96#3, %96#2, %96#1 : index, index, index, index
    }
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
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
