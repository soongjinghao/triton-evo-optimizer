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
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
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
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  %6 = arith.index_cast %1 : i32 to index
  %7 = arith.muli %arg13, %arg12 : i32
  %8 = arith.index_cast %arg10 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c96_i32  : i32 {
    %9 = arith.index_cast %arg14 : i32 to index
    %10 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%9)[%6]
    scf.for %arg15 = %c0 to %10 step %c1 {
      %11 = arith.index_cast %arg14 : i32 to index
      %12 = arith.index_cast %3 : i32 to index
      %13 = arith.index_cast %1 : i32 to index
      %14 = arith.index_cast %c96_i32 : i32 to index
      %15 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> (s0 + ((s2 - s3) floordiv s4) * s1)>()[%arg15, %10, %11, %12, %14]
      %16 = arith.index_cast %15 : index to i64
      %17 = arith.index_cast %arg14 : i32 to index
      %18 = arith.index_cast %3 : i32 to index
      %19 = arith.index_cast %1 : i32 to index
      %20 = arith.index_cast %c96_i32 : i32 to index
      %21 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> ((s0 + ((s2 - s3) floordiv s4) * s1) mod 2)>()[%arg15, %10, %17, %18, %20]
      %22 = arith.index_cast %21 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %23 = arith.select %22, %c3_i64, %c4_i64 : i64
      %24 = arith.index_cast %21 : index to i1
      %c1_i64_2 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %25 = arith.select %24, %c1_i64_2, %c2_i64 : i64
      %26 = hivm.hir.pointer_cast(%c5120_i64, %c14336_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      %27 = hivm.hir.pointer_cast(%c1024_i64, %c10240_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
      %28 = hivm.hir.pointer_cast(%c0_i64_1, %c9216_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %29 = arith.index_cast %arg15 : index to i64
      %30 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg15]
      %31 = arith.index_cast %30 : index to i64
      %32 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg15)[%9]
      %33 = arith.index_cast %32 : index to i32
      %34 = arith.divsi %33, %arg13 : i32
      %35 = arith.remsi %34, %arg12 : i32
      %36 = arith.divsi %33, %7 : i32
      %37 = arith.remsi %36, %arg11 : i32
      %38 = arith.divsi %35, %c4_i32 : i32
      %39 = arith.remsi %35, %c4_i32 : i32
      %40 = arith.muli %38, %arg10 : i32
      %41 = arith.muli %40, %c4_i32 : i32
      %42 = arith.muli %37, %c16_i32 : i32
      %43 = arith.maxsi %42, %c0_i32 : i32
      %44 = arith.index_cast %43 : i32 to index
      %45 = arith.addi %41, %39 : i32
      %46 = arith.muli %45, %c16_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%47, %44]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%48], sizes: [16, 16], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %49 = arith.subi %c0_i32, %42 : i32
      %50 = arith.maxsi %49, %c0_i32 : i32
      %51 = arith.index_cast %50 : i32 to index
      %cast = memref.cast %28 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %52 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %44]
      %53 = affine.min affine_map<()[s0] -> (16, s0)>()[%52]
      %54 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%51, %52]
      %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%53, %54]
      %56 = arith.cmpi slt, %55, %c16 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%55, 16] [1, 1] : memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %57 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%53, %54]
      %58 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%54]
      %59 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%54]
      %subview_3 = memref.subview %28[0, %58, %59, 0] [2, %57, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_4 = memref.cast %subview_3 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %56 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %28 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %25]
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %60 = arith.muli %45, %c32_i32 : i32
      %61 = arith.index_cast %60 : i32 to index
      %62 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%61, %44]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%62], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %5[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
      %cast_7 = memref.cast %27 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %31
      %63 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
      %cast_8 = memref.cast %63 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_7, %true, %55, %c16, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c1_i64, %c-1_i64, %c-1_i64, %16, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %subview_9 = memref.subview %63[0, %58, %59, 0] [2, %57, 16, 16] [1, 1, 1, 1] : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<2x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_10 = memref.cast %subview_9 : memref<2x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_11 = memref.subview %reinterpret_cast_5[0, 0] [%55, 32] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_12 = memref.cast %subview_11 : memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_10 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%cast_12 : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %64 = arith.muli %45, %c64_i32 : i32
      %65 = arith.index_cast %64 : i32 to index
      %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%65, %44]
      %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%66], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_14 = memref.subview %4[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
      %cast_15 = memref.cast %26 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %23]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_14 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_15 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %29
      %67 = hivm.hir.pointer_cast(%c4096_i64) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
      %cast_16 = memref.cast %67 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %c-1_i64_17 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_15, %true, %55, %c16, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_17, %c0_i64, %25, %23, %16, %c-1_i64_17, %c-1_i64_17 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %subview_18 = memref.subview %67[0, %58, %59, 0] [4, %57, 16, 16] [1, 1, 1, 1] : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<4x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_19 = memref.cast %subview_18 : memref<4x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_20 = memref.subview %reinterpret_cast_13[0, 0] [%55, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_21 = memref.cast %subview_20 : memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_19 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%cast_21 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @recompute_w_u_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c5120_i64 = arith.constant 5120 : i64
  %c4864_i64 = arith.constant 4864 : i64
  %c2816_i64 = arith.constant 2816 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c512_i64 = arith.constant 512 : i64
  %c256_i64 = arith.constant 256 : i64
  %c0_i64 = arith.constant 0 : i64
  %c16384 = arith.constant 16384 : index
  %c96_i32 = arith.constant 96 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.index_cast %1 : i32 to index
  %5 = arith.muli %arg13, %arg12 : i32
  %6 = arith.index_cast %arg10 : i32 to index
  %7 = hivm.hir.get_sub_block_idx -> i64
  %8 = arith.index_cast %7 : i64 to index
  %9 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%8]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  %10 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
  %11 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg14 = %3 to %1 step %c96_i32  : i32 {
    %12 = arith.index_cast %arg14 : i32 to index
    %13 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%12)[%4]
    scf.for %arg15 = %c0 to %13 step %c1 {
      %14 = arith.index_cast %arg15 : index to i64
      %15 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg15]
      %16 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg15)[%12]
      %17 = arith.index_cast %15 : index to i64
      %18 = arith.index_cast %16 : index to i32
      %19 = arith.divsi %18, %arg13 : i32
      %20 = arith.remsi %19, %arg12 : i32
      %21 = arith.divsi %18, %5 : i32
      %22 = arith.remsi %21, %arg11 : i32
      %23 = arith.divsi %20, %c4_i32 : i32
      %24 = arith.remsi %20, %c4_i32 : i32
      %25 = arith.muli %23, %arg10 : i32
      %26 = arith.muli %25, %c4_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %24 : i32 to index
      %29 = arith.muli %22, %c16_i32 : i32
      %30 = arith.maxsi %29, %c0_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 + s2)>()[%31, %27, %28]
      %33 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%6, %31]
      %34 = affine.min affine_map<()[s0] -> (16, s0)>()[%33]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%32], sizes: [16, 1], strides: [4, 4] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %35 = arith.addi %26, %24 : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4)>()[%36, %31]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%37], sizes: [16], strides: [4] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %38 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x1xf32, #hivm.address_space<ub>>
      %subview = memref.subview %38[0, 0, 0] [8, 1, 1] [1, 1, 1] : memref<8x8x1xf32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %39 = arith.subi %c0_i32, %29 : i32
      %40 = arith.maxsi %39, %c0_i32 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%41, %33]
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%34, %42]
      %44 = arith.cmpi slt, %43, %c16 : index
      %45 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%8, %34, %42]
      %46 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%45, %34, %42]
      %subview_1 = memref.subview %reinterpret_cast[%9, 0] [8, 1] [1, 1] : memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %subview_1[0, 0] [%46, 1] [1, 1] : memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %47 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%42, %8]
      %48 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%33, %8]
      %49 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%48, %42, %8]
      %50 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%49, %47]
      %51 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%47, %8]
      %subview_3 = memref.subview %subview[%51, 0] [%50, 1] [1, 1] : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %44 {
        %collapse_shape_35 = memref.collapse_shape %subview [[0, 1]] : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_36, %offset_37, %sizes_38, %strides_39 = memref.extract_strided_metadata %collapse_shape_35 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_40 = memref.reinterpret_cast %base_buffer_36 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_40 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1]] : memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %42 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %52 = hivm.hir.pointer_cast(%c256_i64) : memref<8x8xf32, #hivm.address_space<ub>>
      %subview_5 = memref.subview %52[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_0[%9] [8] [1] : memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %subview_6[0] [%46] [1] : memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %subview_5[%51] [%50] [1] : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %44 {
        %base_buffer_35, %offset_36, %sizes_37, %strides_38 = memref.extract_strided_metadata %subview_5 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_39 = memref.reinterpret_cast %base_buffer_35 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_39 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %42 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c256_i64) : memref<8x8xf32, #hivm.address_space<ub>>
      %subview_9 = memref.subview %53[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_5 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_9 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp {vector_producer_to_fuse_0} ins(%reinterpret_cast_10 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      %54 = arith.muli %35, %c32_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%55, %31]
      %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%56], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %57 = hivm.hir.pointer_cast(%c512_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %subview_17 = memref.subview %reinterpret_cast_16[%9, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_18 = memref.subview %subview_17[0, 0] [%46, 32] [1, 1] : memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_19 = memref.subview %57[%51, 0] [%50, 32] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %57 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_20 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview_18 : memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_19 : memref<?x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %58 = hivm.hir.pointer_cast(%c512_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %59 = hivm.hir.pointer_cast(%c2560_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%57, %subview : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%58 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_21 = memref.subview %11[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_22 = memref.collapse_shape %subview_21 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_23 = memref.subview %collapse_shape_22[%9, 0] [8, 64] [1, 1] {to_be_bubbled_slice} : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %17
      %collapse_shape_24 = memref.collapse_shape %58 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %subview_23 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_24 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_0, tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
      %60 = arith.muli %35, %c64_i32 : i32
      %61 = arith.index_cast %60 : i32 to index
      %62 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%61, %31]
      %reinterpret_cast_26 = memref.reinterpret_cast %arg3 to offset: [%62], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %63 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %subview_27 = memref.subview %reinterpret_cast_26[%9, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %subview_27[0, 0] [%46, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_29 = memref.subview %63[%51, 0] [%50, 64] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %44 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_35 = memref.collapse_shape %63 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_35 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview_28 : memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_29 : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %64 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %65 = hivm.hir.pointer_cast(%c4864_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%63, %subview : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%64 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%65 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %expand_shape = memref.expand_shape %subview_9 [[0, 1]] output_shape [8, 1] : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> into memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
      %66 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c5120_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%64, %expand_shape : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%66 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %subview_30 = memref.subview %10[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_32 = memref.subview %collapse_shape_31[%9, 0] [8, 64] [1, 1] {to_be_bubbled_slice} : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %14
      %collapse_shape_33 = memref.collapse_shape %66 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %subview_32 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_33 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_1, tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
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
