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
      %34 = arith.remsi %33, %arg13 : i32
      %35 = arith.divsi %33, %arg13 : i32
      %36 = arith.remsi %35, %arg12 : i32
      %37 = arith.divsi %33, %7 : i32
      %38 = arith.remsi %37, %arg11 : i32
      %39 = arith.divsi %36, %c4_i32 : i32
      %40 = arith.remsi %36, %c4_i32 : i32
      %41 = arith.muli %39, %arg10 : i32
      %42 = arith.muli %41, %c4_i32 : i32
      %43 = arith.muli %38, %c16_i32 : i32
      %44 = arith.maxsi %43, %c0_i32 : i32
      %45 = arith.index_cast %44 : i32 to index
      %46 = arith.addi %42, %40 : i32
      %47 = arith.muli %46, %c16_i32 : i32
      %48 = arith.index_cast %47 : i32 to index
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%48, %45]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%49], sizes: [16, 16], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %50 = arith.subi %c0_i32, %43 : i32
      %51 = arith.maxsi %50, %c0_i32 : i32
      %52 = arith.index_cast %51 : i32 to index
      %cast = memref.cast %28 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %53 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %45]
      %54 = affine.min affine_map<()[s0] -> (16, s0)>()[%53]
      %55 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%52, %53]
      %56 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%54, %55]
      %57 = arith.cmpi slt, %56, %c16 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%56, 16] [1, 1] : memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %58 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%54, %55]
      %59 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%55]
      %60 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%55]
      %subview_3 = memref.subview %28[0, %59, %60, 0] [2, %58, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_4 = memref.cast %subview_3 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %57 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %28 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %25]
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_4 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %61 = arith.muli %46, %c32_i32 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = arith.muli %34, %c64_i32 : i32
      %64 = arith.maxsi %63, %c0_i32 : i32
      %65 = arith.index_cast %64 : i32 to index
      %66 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%65, %62, %45]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%66], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %67 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>()[%8, %65, %45]
      %68 = affine.min affine_map<()[s0] -> (16, s0)>()[%67]
      %69 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 32, 0)>()[%65]
      %70 = affine.min affine_map<()[s0] -> (64, s0)>()[%69]
      %71 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%52, %67]
      %72 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%68, %71]
      %73 = arith.subi %c0_i32, %63 : i32
      %74 = arith.maxsi %73, %c0_i32 : i32
      %75 = arith.index_cast %74 : i32 to index
      %76 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%75, %69]
      %77 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%70, %76]
      %subview_6 = memref.subview %4[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
      %cast_7 = memref.cast %27 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %31
      %78 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
      %cast_8 = memref.cast %78 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_7, %true, %56, %c16, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c1_i64, %c-1_i64, %c-1_i64, %16, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %79 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%68, %71]
      %80 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%70, %76]
      %81 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%71]
      %82 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%71]
      %83 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%76]
      %84 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%76]
      %subview_9 = memref.subview %78[%83, %81, %82, %84] [%80, %79, 16, 16] [1, 1, 1, 1] : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_10 = memref.cast %subview_9 : memref<?x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_11 = memref.subview %reinterpret_cast_5[0, 0] [%72, %77] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_10 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_11 : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %85 = arith.muli %46, %c64_i32 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%86, %45]
      %reinterpret_cast_12 = memref.reinterpret_cast %arg6 to offset: [%87], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %5[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
      %cast_14 = memref.cast %26 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %23]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_13 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_14 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %29
      %88 = hivm.hir.pointer_cast(%c4096_i64) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
      %cast_15 = memref.cast %88 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %c-1_i64_16 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_14, %true, %56, %c16, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_15 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_16, %c0_i64, %25, %23, %16, %c-1_i64_16, %c-1_i64_16 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %subview_17 = memref.subview %88[0, %59, %60, 0] [4, %58, 16, 16] [1, 1, 1, 1] : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<4x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_18 = memref.cast %subview_17 : memref<4x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_19 = memref.subview %reinterpret_cast_12[0, 0] [%56, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_20 = memref.cast %subview_19 : memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_18 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%cast_20 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
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
  %c64 = arith.constant 64 : index
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
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c96_i32  : i32 {
    %12 = arith.index_cast %arg14 : i32 to index
    %13 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%12)[%4]
    scf.for %arg15 = %c0 to %13 step %c1 {
      %14 = arith.index_cast %arg15 : index to i64
      %15 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg15]
      %16 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg15)[%12]
      %17 = arith.index_cast %15 : index to i64
      %18 = arith.index_cast %16 : index to i32
      %19 = arith.remsi %18, %arg13 : i32
      %20 = arith.divsi %18, %arg13 : i32
      %21 = arith.remsi %20, %arg12 : i32
      %22 = arith.divsi %18, %5 : i32
      %23 = arith.remsi %22, %arg11 : i32
      %24 = arith.divsi %21, %c4_i32 : i32
      %25 = arith.remsi %21, %c4_i32 : i32
      %26 = arith.muli %24, %arg10 : i32
      %27 = arith.muli %26, %c4_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %25 : i32 to index
      %30 = arith.muli %23, %c16_i32 : i32
      %31 = arith.maxsi %30, %c0_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 + s2)>()[%32, %28, %29]
      %34 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%6, %32]
      %35 = affine.min affine_map<()[s0] -> (16, s0)>()[%34]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [16, 1], strides: [4, 4] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %36 = arith.addi %27, %25 : i32
      %37 = arith.index_cast %36 : i32 to index
      %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4)>()[%37, %32]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%38], sizes: [16], strides: [4] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %39 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x1xf32, #hivm.address_space<ub>>
      %subview = memref.subview %39[0, 0, 0] [8, 1, 1] [1, 1, 1] : memref<8x8x1xf32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %40 = arith.subi %c0_i32, %30 : i32
      %41 = arith.maxsi %40, %c0_i32 : i32
      %42 = arith.index_cast %41 : i32 to index
      %43 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%42, %34]
      %44 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %43]
      %45 = arith.cmpi slt, %44, %c16 : index
      %46 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%8, %35, %43]
      %47 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%46, %35, %43]
      %subview_1 = memref.subview %reinterpret_cast[%9, 0] [8, 1] [1, 1] : memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %subview_1[0, 0] [%47, 1] [1, 1] : memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
      %48 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%43, %8]
      %49 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%34, %8]
      %50 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%49, %43, %8]
      %51 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %48]
      %52 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%48, %8]
      %subview_3 = memref.subview %subview[%52, 0] [%51, 1] [1, 1] : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %45 {
        %collapse_shape_34 = memref.collapse_shape %subview [[0, 1]] : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_35, %offset_36, %sizes_37, %strides_38 = memref.extract_strided_metadata %collapse_shape_34 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_39 = memref.reinterpret_cast %base_buffer_35 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_39 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1]] : memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %43 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %53 = hivm.hir.pointer_cast(%c256_i64) : memref<8x8xf32, #hivm.address_space<ub>>
      %subview_5 = memref.subview %53[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_0[%9] [8] [1] : memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %subview_6[0] [%47] [1] : memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %subview_5[%52] [%51] [1] : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %45 {
        %base_buffer_34, %offset_35, %sizes_36, %strides_37 = memref.extract_strided_metadata %subview_5 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_38 = memref.reinterpret_cast %base_buffer_34 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_38 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %43 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %54 = hivm.hir.pointer_cast(%c256_i64) : memref<8x8xf32, #hivm.address_space<ub>>
      %subview_9 = memref.subview %54[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_5 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_9 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp {vector_producer_to_fuse_0} ins(%reinterpret_cast_10 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      %55 = arith.muli %36, %c32_i32 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = arith.muli %19, %c64_i32 : i32
      %58 = arith.maxsi %57, %c0_i32 : i32
      %59 = arith.index_cast %58 : i32 to index
      %60 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%59, %56, %32]
      %61 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 32, 0)>()[%59]
      %62 = affine.min affine_map<()[s0] -> (64, s0)>()[%61]
      %63 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>()[%6, %59, %32]
      %64 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%42, %63]
      %65 = affine.min affine_map<()[s0] -> (16, s0)>()[%63]
      %66 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%65, %64]
      %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%60], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %67 = hivm.hir.pointer_cast(%c512_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %68 = arith.subi %c0_i32, %57 : i32
      %69 = arith.maxsi %68, %c0_i32 : i32
      %70 = arith.index_cast %69 : i32 to index
      %71 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%70, %61]
      %72 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%62, %71]
      %73 = arith.cmpi slt, %66, %c16 : index
      %74 = arith.cmpi slt, %72, %c64 : index
      %75 = arith.ori %73, %74 : i1
      %76 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%8, %65, %64]
      %77 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%76, %65, %64]
      %subview_17 = memref.subview %reinterpret_cast_16[%9, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_18 = memref.subview %subview_17[0, 0] [%77, %72] [1, 1] : memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %78 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%64, %8]
      %79 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%63, %8]
      %80 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%79, %64, %8]
      %81 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%80, %78]
      %82 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%78, %8]
      %subview_19 = memref.subview %67[%82, %71] [%81, %72] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %75 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_34 = memref.collapse_shape %67 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_34 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview_18 : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_19 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %71 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %83 = hivm.hir.pointer_cast(%c512_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %84 = hivm.hir.pointer_cast(%c2560_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%67, %subview : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%83 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%84 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_20 = memref.subview %10[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_22 = memref.subview %collapse_shape_21[%9, 0] [8, 64] [1, 1] {to_be_bubbled_slice} : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %17
      %collapse_shape_23 = memref.collapse_shape %83 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_24 = memref.collapse_shape %subview_22 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_23 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_0, tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
      %85 = arith.muli %36, %c64_i32 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%86, %32]
      %reinterpret_cast_25 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %88 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %subview_26 = memref.subview %reinterpret_cast_25[%9, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_27 = memref.subview %subview_26[0, 0] [%47, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %88[%52, 0] [%51, 64] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %45 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_34 = memref.collapse_shape %88 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_34 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%subview_27 : memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_28 : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %89 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %90 = hivm.hir.pointer_cast(%c4864_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%88, %subview : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%89 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%90 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %expand_shape = memref.expand_shape %subview_9 [[0, 1]] output_shape [8, 1] : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> into memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
      %91 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
      %92 = hivm.hir.pointer_cast(%c5120_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%89, %expand_shape : memref<8x64xf32, #hivm.address_space<ub>>, memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%91 : memref<8x64xf32, #hivm.address_space<ub>>) temp_buffer(%92 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %subview_29 = memref.subview %11[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_30 = memref.collapse_shape %subview_29 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_31 = memref.subview %collapse_shape_30[%9, 0] [8, 64] [1, 1] {to_be_bubbled_slice} : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %14
      %collapse_shape_32 = memref.collapse_shape %91 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_33 = memref.collapse_shape %subview_31 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_32 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_1, tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
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
