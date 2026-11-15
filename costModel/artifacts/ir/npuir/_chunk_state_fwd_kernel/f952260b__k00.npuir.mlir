// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_state_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<35xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c24_i32 = arith.constant 24 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c15_i32 = arith.constant 15 : i32
  %c0_i32 = arith.constant 0 : i32
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
    %24 = arith.muli %21, %8 : i64
    %25 = arith.muli %19, %c16_i32 : i32
    %26 = arith.subi %9, %24 : i64
    %27 = arith.minsi %8, %26 : i64
    %28 = arith.muli %23, %10 : i64
    %29 = arith.muli %21, %11 : i64
    %30 = arith.addi %28, %29 : i64
    %31 = arith.muli %15, %arg22 : i32
    %32 = arith.extsi %31 : i32 to i64
    %33 = arith.addi %30, %32 : i64
    %34 = arith.index_cast %25 : i32 to index
    %35 = arith.index_cast %27 : i64 to index
    %36 = arith.index_cast %33 : i64 to index
    %37 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%34, %13]
    %38 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%37, %34]
    %39 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%38, %34]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    scf.for %arg36 = %c0_i32 to %7 step %c1_i32  : i32 {
      %41 = arith.muli %arg36, %c16_i32 : i32
      %42 = arith.index_cast %41 : i32 to index
      %43 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %44 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %45 = hivm.hir.pointer_cast(%c0_i64_0) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast = memref.cast %45 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      scf.for %arg37 = %c0_i64_0 to %27 step %c64_i64  : i64 {
        %53 = arith.index_cast %arg37 : i64 to index
        %54 = affine.min affine_map<(d0, d1) -> ((d0 - d1) ceildiv 16, 4)>(%35, %53)
        scf.for %arg38 = %c0 to %54 step %c1 {
          %55 = arith.index_cast %arg37 : i64 to index
          %56 = arith.index_cast %c0_i64_0 : i64 to index
          %57 = arith.index_cast %27 : i64 to index
          %58 = arith.index_cast %c64_i64 : i64 to index
          %59 = arith.index_cast %arg36 : i32 to index
          %60 = arith.index_cast %c0_i32 : i32 to index
          %61 = arith.index_cast %7 : i32 to index
          %62 = arith.index_cast %c1_i32 : i32 to index
          %63 = arith.index_cast %arg35 : i32 to index
          %64 = arith.index_cast %3 : i32 to index
          %65 = arith.index_cast %1 : i32 to index
          %66 = arith.index_cast %c24_i32 : i32 to index
          %67 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12] -> (s0 + ((s2 - s3) floordiv s5) * s1 + ((s6 - s7) floordiv s9) * (s1 * ((-s3 + s4 + s5 - 1) floordiv s5)) + ((s10 - s11) floordiv s12) * ((s1 * ((-s3 + s4 + s5 - 1) floordiv s5)) * ((-s7 + s8 + s9 - 1) floordiv s9)))>()[%arg38, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %66]
          %68 = arith.index_cast %67 : index to i64
          %69 = arith.index_cast %arg37 : i64 to index
          %70 = arith.index_cast %c0_i64_0 : i64 to index
          %71 = arith.index_cast %27 : i64 to index
          %72 = arith.index_cast %c64_i64 : i64 to index
          %73 = arith.index_cast %arg36 : i32 to index
          %74 = arith.index_cast %c0_i32 : i32 to index
          %75 = arith.index_cast %7 : i32 to index
          %76 = arith.index_cast %c1_i32 : i32 to index
          %77 = arith.index_cast %arg35 : i32 to index
          %78 = arith.index_cast %3 : i32 to index
          %79 = arith.index_cast %1 : i32 to index
          %80 = arith.index_cast %c24_i32 : i32 to index
          %81 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12] -> ((s0 + ((s2 - s3) floordiv s5) * s1 + ((s6 - s7) floordiv s9) * (s1 * ((-s3 + s4 + s5 - 1) floordiv s5)) + ((s10 - s11) floordiv s12) * ((s1 * ((-s3 + s4 + s5 - 1) floordiv s5)) * ((-s7 + s8 + s9 - 1) floordiv s9))) mod 2)>()[%arg38, %54, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %80]
          %82 = arith.index_cast %81 : index to i1
          %c2_i64 = arith.constant 2 : i64
          %c3_i64 = arith.constant 3 : i64
          %83 = arith.select %82, %c2_i64, %c3_i64 : i64
          %84 = arith.index_cast %81 : index to i1
          %c0_i64_3 = arith.constant 0 : i64
          %c1_i64_4 = arith.constant 1 : i64
          %85 = arith.select %84, %c0_i64_3, %c1_i64_4 : i64
          %86 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
          annotation.mark %86 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
          %87 = hivm.hir.pointer_cast(%c0_i64_0, %c2048_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
          annotation.mark %87 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
          %88 = arith.index_cast %arg38 : index to i64
          %89 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg38]
          %90 = arith.index_cast %89 : index to i64
          %91 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%arg38)[%53]
          %92 = arith.index_cast %91 : index to i64
          %subview_5 = memref.subview %43[%arg38, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
          %cast_6 = memref.cast %87 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %85]
          hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %90
          %subview_7 = memref.subview %44[%arg38, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
          %cast_8 = memref.cast %86 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %83]
          hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %88
          %93 = arith.cmpi eq, %92, %c0_i64_0 : i64
          %c-1_i64 = arith.constant -1 : i64
          hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_6, %cast_8, %93, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %85, %83, %68, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      }
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %46 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%42, %36, %34, %12]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [16, 16], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%42, %14]
      %48 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%47, %42]
      %49 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%48, %42]
      %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
      %51 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%40]
      %52 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%50]
      %subview = memref.subview %45[0, 0, 0, 0] [%52, %51, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>>
      %cast_1 = memref.cast %subview : memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_2 = memref.subview %reinterpret_cast[0, 0] [%40, %50] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_1 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    }
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
  %c0_i32 = arith.constant 0 : i32
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
    %40 = arith.muli %39, %8 : i64
    %41 = arith.muli %37, %9 : i64
    %42 = arith.muli %41, %10 : i64
    %43 = arith.addi %40, %42 : i64
    %44 = arith.divsi %31, %arg13 : i32
    %45 = arith.muli %44, %arg19 : i32
    %46 = arith.extsi %45 : i32 to i64
    %47 = arith.addi %43, %46 : i64
    %48 = arith.muli %39, %11 : i64
    %49 = arith.muli %41, %12 : i64
    %50 = arith.addi %48, %49 : i64
    %51 = arith.muli %31, %arg16 : i32
    %52 = arith.extsi %51 : i32 to i64
    %53 = arith.addi %50, %52 : i64
    %54 = arith.muli %39, %13 : i64
    %55 = arith.muli %37, %14 : i64
    %56 = arith.addi %54, %55 : i64
    %57 = arith.muli %31, %arg26 : i32
    %58 = arith.extsi %57 : i32 to i64
    %59 = arith.addi %56, %58 : i64
    %60 = arith.muli %39, %15 : i64
    %61 = arith.muli %37, %16 : i64
    %62 = arith.addi %60, %61 : i64
    %63 = arith.muli %31, %arg29 : i32
    %64 = arith.extsi %63 : i32 to i64
    %65 = arith.addi %62, %64 : i64
    %66 = arith.muli %35, %c16_i32 : i32
    %67 = arith.index_cast %65 : i64 to index
    %68 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%67, %18]
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%68], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %69 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %70 = arith.subi %19, %41 : i64
    %71 = arith.minsi %9, %70 : i64
    %72 = arith.index_cast %66 : i32 to index
    %73 = arith.index_cast %71 : i64 to index
    %74 = arith.index_cast %53 : i64 to index
    %75 = arith.index_cast %47 : i64 to index
    %76 = arith.index_cast %59 : i64 to index
    %77 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%72, %24]
    %78 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%77, %72]
    %79 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%78, %72]
    %80 = affine.max affine_map<()[s0] -> (0, s0)>()[%79]
    %81 = arith.cmpi slt, %80, %c16 : index
    scf.for %arg36 = %c0_i32 to %7 step %c1_i32  : i32 {
      %82 = arith.muli %arg36, %c16_i32 : i32
      %83 = arith.index_cast %82 : i32 to index
      %84 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %85 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %86 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%83, %28]
      %87 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%86, %83]
      %88 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%87, %83]
      %89 = affine.max affine_map<()[s0] -> (0, s0)>()[%88]
      %90 = arith.cmpi slt, %89, %c16 : index
      %91:4 = scf.for %arg37 = %c0_i64 to %71 step %c64_i64 iter_args(%arg38 = %72, %arg39 = %c0, %arg40 = %c0, %arg41 = %c0) -> (index, index, index, index)  : i64 {
        %92 = arith.index_cast %arg37 : i64 to index
        %93 = affine.min affine_map<(d0, d1) -> ((d0 - d1) ceildiv 16, 4)>(%73, %92)
        %94:4 = scf.for %arg42 = %c0 to %93 step %c1 iter_args(%arg43 = %arg40, %arg44 = %arg39, %arg45 = %arg38, %arg46 = %arg41) -> (index, index, index, index) {
          %95 = arith.index_cast %arg42 : index to i64
          %96 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg42]
          %97 = arith.index_cast %96 : index to i64
          %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%arg42)[%92]
          %99 = arith.index_cast %98 : index to i64
          %100 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%74, %arg45]
          %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%100], sizes: [16, 16], strides: [1, %22] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
          %101 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%83, %75, %arg44]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%101], sizes: [16, 16], strides: [%23, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %102 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%76, %arg43]
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%102], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %103 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%67, %arg46]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %104 = arith.subi %71, %99 : i64
          %105 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<ub>>
          %106 = arith.index_cast %104 : i64 to index
          %107 = affine.max affine_map<()[s0] -> (0, s0)>()[%106]
          %108 = affine.min affine_map<()[s0] -> (16, s0)>()[%107]
          %109 = affine.max affine_map<()[s0] -> (0, s0)>()[%108]
          %110 = arith.cmpi slt, %109, %c16 : index
          %111 = arith.ori %81, %110 : i1
          %subview = memref.subview %reinterpret_cast_1[0, 0] [%80, %109] [1, 1] : memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
          %subview_5 = memref.subview %105[0, 0] [%80, %109] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          scf.if %111 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_15 = memref.collapse_shape %105 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.load ins(%subview : memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = true
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          annotation.mark %105 {MayImplicitTransposeWithLastAxis} : memref<16x16xf32, #hivm.address_space<ub>>
          %subview_6 = memref.subview %84[%arg42, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %97
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          scf.if %27 {
            %collapse_shape_15 = memref.collapse_shape %105 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            %collapse_shape_16 = memref.collapse_shape %collapse_shape [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.store ins(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          } {limit_sub_block_id0}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
          annotation.mark %105 {MayImplicitTransposeWithLastAxis} : memref<16x16xf32, #hivm.address_space<ub>>
          %112 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
          %113 = arith.ori %110, %90 : i1
          %subview_7 = memref.subview %reinterpret_cast_2[0, 0] [%109, %89] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_8 = memref.subview %112[0, 0] [%109, %89] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          scf.if %113 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_15 = memref.collapse_shape %112 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.load ins(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          %114 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          %115 = arith.cmpi slt, %108, %c16 : index
          %subview_9 = memref.subview %reinterpret_cast_4[0] [%108] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %114[0] [%108] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %115 {
            hivm.hir.pipe_barrier[<PIPE_V>]
            hivm.hir.vbrc ins(%cst_0 : f32) outs(%114 : memref<16xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.load ins(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          %116 = hivm.hir.pointer_cast(%c2112_i64) : memref<16xf32, #hivm.address_space<ub>>
          %subview_11 = memref.subview %reinterpret_cast_3[0] [%108] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_12 = memref.subview %116[0] [%108] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %115 {
            hivm.hir.vbrc ins(%cst_0 : f32) outs(%116 : memref<16xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.load ins(%subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %117 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.vmul ins(%114, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%117 : memref<16xf32, #hivm.address_space<ub>>)
          %118 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vadd ins(%117, %69 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%118 : memref<16xf32, #hivm.address_space<ub>>)
          %119 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vexp ins(%118 : memref<16xf32, #hivm.address_space<ub>>) outs(%119 : memref<16xf32, #hivm.address_space<ub>>)
          %120 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vmul ins(%119, %116 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%120 : memref<16xf32, #hivm.address_space<ub>>)
          %expand_shape = memref.expand_shape %120 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
          %121 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
          %122 = hivm.hir.pointer_cast(%c2176_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vmul ins(%112, %expand_shape : memref<16x16xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%121 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%122 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %subview_13 = memref.subview %85[%arg42, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %95
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          scf.if %27 {
            %collapse_shape_15 = memref.collapse_shape %121 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            %collapse_shape_16 = memref.collapse_shape %collapse_shape_14 [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.store ins(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          } {limit_sub_block_id0}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
          %123 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg45, %29]
          %124 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg44, %30]
          %125 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%arg43]
          %126 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%arg46]
          scf.yield %125, %124, %123, %126 : index, index, index, index
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
        scf.yield %94#2, %94#1, %94#0, %94#3 : index, index, index, index
      }
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
