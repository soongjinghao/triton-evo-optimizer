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
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
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
  %5 = arith.index_cast %arg10 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c24_i32  : i32 {
    %6 = arith.index_cast %arg14 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c24_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %arg14 : i32 to index
    %13 = arith.index_cast %3 : i32 to index
    %14 = arith.index_cast %1 : i32 to index
    %15 = arith.index_cast %c24_i32 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%12, %13, %15]
    %17 = arith.index_cast %16 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64_3 = arith.constant 1 : i64
    %18 = arith.select %17, %c0_i64_2, %c1_i64_3 : i64
    %19 = hivm.hir.pointer_cast(%c4096_i64, %c24576_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %20 = hivm.hir.pointer_cast(%c0_i64_1, %c20480_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_mask_norm
    %21 = arith.divsi %arg14, %arg13 : i32
    %22 = arith.remsi %21, %arg12 : i32
    %23 = arith.divsi %arg14, %4 : i32
    %24 = arith.remsi %23, %arg11 : i32
    %25 = arith.divsi %22, %c8_i32 : i32
    %26 = arith.remsi %22, %c8_i32 : i32
    %27 = arith.muli %25, %arg10 : i32
    %28 = arith.muli %27, %c8_i32 : i32
    %29 = arith.muli %24, %c32_i32 : i32
    %30 = arith.maxsi %29, %c0_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = arith.addi %28, %26 : i32
    %33 = arith.muli %32, %c32_i32 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%34, %31]
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%35], sizes: [32, 32], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %36 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %31]
    %37 = affine.min affine_map<()[s0] -> (32, s0)>()[%36]
    %38 = arith.subi %c0_i32, %29 : i32
    %39 = arith.maxsi %38, %c0_i32 : i32
    %40 = arith.index_cast %39 : i32 to index
    %41 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%40, %36]
    %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%37, %41]
    %43 = arith.cmpi slt, %42, %c32 : index
    %cast = memref.cast %20 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%42, 32] [1, 1] : memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %44 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%37, %41]
    %45 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%41]
    %46 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%41]
    %subview_4 = memref.subview %20[0, %45, %46, 0] [4, %44, 16, 8] [1, 1, 1, 1] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_5 = memref.cast %subview_4 : memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %43 {
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      %collapse_shape = memref.collapse_shape %20 [[0, 1, 2, 3]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> into memref<1024xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<1024xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
    hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %47 = arith.muli %32, %c64_i32 : i32
    %48 = arith.index_cast %47 : i32 to index
    %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%48, %31]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%49], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %50 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_7 = memref.cast %19 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%50 : memref<32x64xf32, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    %51 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    %cast_8 = memref.cast %51 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_7, %true, %42, %c32, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %11, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %subview_9 = memref.subview %51[0, %45, %46, 0] [4, %44, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<4x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
    %cast_10 = memref.cast %subview_9 : memref<4x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%42, 64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_12 = memref.cast %subview_11 : memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_10 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%cast_12 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %52 = arith.muli %32, %c128_i32 : i32
    %53 = arith.index_cast %52 : i32 to index
    %54 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
    scf.for %arg15 = %c0 to %c2 step %c1 {
      %55 = arith.index_cast %arg14 : i32 to index
      %56 = arith.index_cast %3 : i32 to index
      %57 = arith.index_cast %1 : i32 to index
      %58 = arith.index_cast %c24_i32 : i32 to index
      %59 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + ((s1 - s2) floordiv s3) * 2)>()[%arg15, %55, %56, %58]
      %60 = arith.index_cast %59 : index to i64
      %61 = arith.index_cast %arg14 : i32 to index
      %62 = arith.index_cast %3 : i32 to index
      %63 = arith.index_cast %1 : i32 to index
      %64 = arith.index_cast %c24_i32 : i32 to index
      %65 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 mod 2)>()[%arg15, %61, %62, %64]
      %66 = arith.index_cast %65 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %67 = arith.select %66, %c3_i64, %c4_i64 : i64
      %68 = hivm.hir.pointer_cast(%c12288_i64, %c32768_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %68 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      %69 = arith.index_cast %arg15 : index to i32
      %70 = arith.muli %69, %c64_i32 : i32
      %71 = arith.maxsi %70, %c0_i32 : i32
      %72 = arith.index_cast %71 : i32 to index
      %73 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%72, %53, %31]
      %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%73], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %74 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>()[%5, %72, %31]
      %75 = affine.min affine_map<()[s0] -> (32, s0)>()[%74]
      %76 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128, 0)>()[%72]
      %77 = affine.min affine_map<()[s0] -> (64, s0)>()[%76]
      %78 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%40, %74]
      %79 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%75, %78]
      %80 = arith.subi %c0_i32, %70 : i32
      %81 = arith.maxsi %80, %c0_i32 : i32
      %82 = arith.index_cast %81 : i32 to index
      %83 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%82, %76]
      %84 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%77, %83]
      %subview_14 = memref.subview %54[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_15 = memref.cast %68 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %67]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_14 : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_15 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %85 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
      %cast_16 = memref.cast %85 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64_17 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_15, %true, %42, %c32, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_17, %c0_i64_0, %c-1_i64_17, %67, %60, %c-1_i64_17, %c-1_i64_17 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %86 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%75, %78]
      %87 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%77, %83]
      %88 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%78]
      %89 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%78]
      %90 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%83]
      %91 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%83]
      %subview_18 = memref.subview %85[%90, %88, %89, %91] [%87, %86, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_19 = memref.cast %subview_18 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_20 = memref.subview %reinterpret_cast_13[0, 0] [%79, %84] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_19 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_20 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @recompute_w_u_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c10240_i64 = arith.constant 10240 : i64
  %c9728_i64 = arith.constant 9728 : i64
  %c5632_i64 = arith.constant 5632 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c8192 = arith.constant 8192 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c24_i32 = arith.constant 24 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  %5 = arith.index_cast %arg10 : i32 to index
  %6 = hivm.hir.get_sub_block_idx -> i64
  %7 = arith.index_cast %6 : i64 to index
  %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%7]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %9 = arith.divsi %arg14, %arg13 : i32
    %10 = arith.remsi %9, %arg12 : i32
    %11 = arith.divsi %arg14, %4 : i32
    %12 = arith.remsi %11, %arg11 : i32
    %13 = arith.divsi %10, %c8_i32 : i32
    %14 = arith.remsi %10, %c8_i32 : i32
    %15 = arith.muli %13, %arg10 : i32
    %16 = arith.muli %15, %c8_i32 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.index_cast %14 : i32 to index
    %19 = arith.muli %12, %c32_i32 : i32
    %20 = arith.maxsi %19, %c0_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 8 + s1 + s2)>()[%21, %17, %18]
    %23 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %21]
    %24 = affine.min affine_map<()[s0] -> (32, s0)>()[%23]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [32, 1], strides: [8, 8] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %25 = arith.addi %16, %14 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8)>()[%26, %21]
    %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%27], sizes: [32], strides: [8] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %28 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %28[0, 0, 0] [16, 1, 1] [1, 1, 1] : memref<16x8x1xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %29 = arith.subi %c0_i32, %19 : i32
    %30 = arith.maxsi %29, %c0_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%31, %23]
    %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %32]
    %34 = arith.cmpi slt, %33, %c32 : index
    %35 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>()[%7, %24, %32]
    %36 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>()[%35, %24, %32]
    %subview_1 = memref.subview %reinterpret_cast[%8, 0] [16, 1] [1, 1] : memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> to memref<16x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %subview_1[0, 0] [%36, 1] [1, 1] : memref<16x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %37 = affine.max affine_map<()[s0, s1] -> (s1 * 16, s0)>()[%32, %7]
    %38 = affine.min affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>()[%23, %7]
    %39 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>()[%38, %32, %7]
    %40 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%39, %37]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 16)>()[%37, %7]
    %subview_3 = memref.subview %subview[%41, 0] [%40, 1] [1, 1] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %34 {
      %collapse_shape_23 = memref.collapse_shape %subview [[0, 1]] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer_24, %offset_25, %sizes_26, %strides_27 = memref.extract_strided_metadata %collapse_shape_23 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_28 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    } {hivm.unlikely_condition}
    %collapse_shape = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1]] : memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %32 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %42 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %subview_5 = memref.subview %42[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast_0[%8] [16] [1] : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %subview_6[0] [%36] [1] : memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %subview_5[%41] [%40] [1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
    scf.if %34 {
      %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %subview_5 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_27 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %32 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %43 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %subview_9 = memref.subview %43[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_5 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_9 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%reinterpret_cast_10 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %44 = arith.muli %25, %c64_i32 : i32
    %45 = arith.index_cast %44 : i32 to index
    %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%45, %21]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %47 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %subview_17 = memref.subview %reinterpret_cast_16[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %subview_17[0, 0] [%36, 64] [1, 1] : memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_19 = memref.subview %47[%41, 0] [%40, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %34 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_23 = memref.collapse_shape %47 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_23 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%subview_18 : memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_19 : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %48 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c5120_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%47, %subview : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%48 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%49 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %50 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
    %subview_20 = memref.subview %50[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    %collapse_shape_21 = memref.collapse_shape %48 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %subview_20 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_21 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    annotation.mark %subview_20 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %51 = arith.muli %25, %c128_i32 : i32
    %52 = arith.index_cast %51 : i32 to index
    %53 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
    %expand_shape = memref.expand_shape %subview_9 [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
    scf.for %arg15 = %c0 to %c2 step %c1 {
      %54 = arith.index_cast %arg15 : index to i32
      %55 = arith.muli %54, %c64_i32 : i32
      %56 = arith.maxsi %55, %c0_i32 : i32
      %57 = arith.index_cast %56 : i32 to index
      %58 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%57, %52, %21]
      %59 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128, 0)>()[%57]
      %60 = affine.min affine_map<()[s0] -> (64, s0)>()[%59]
      %61 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>()[%5, %57, %21]
      %62 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%31, %61]
      %63 = affine.min affine_map<()[s0] -> (32, s0)>()[%61]
      %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%63, %62]
      %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%58], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %65 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %66 = arith.subi %c0_i32, %55 : i32
      %67 = arith.maxsi %66, %c0_i32 : i32
      %68 = arith.index_cast %67 : i32 to index
      %69 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%68, %59]
      %70 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%60, %69]
      %71 = arith.cmpi slt, %64, %c32 : index
      %72 = arith.cmpi slt, %70, %c64 : index
      %73 = arith.ori %71, %72 : i1
      %74 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>()[%7, %63, %62]
      %75 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>()[%74, %63, %62]
      %subview_24 = memref.subview %reinterpret_cast_23[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_25 = memref.subview %subview_24[0, 0] [%75, %70] [1, 1] : memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %76 = affine.max affine_map<()[s0, s1] -> (s1 * 16, s0)>()[%62, %7]
      %77 = affine.min affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>()[%61, %7]
      %78 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>()[%77, %62, %7]
      %79 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%78, %76]
      %80 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 16)>()[%76, %7]
      %subview_26 = memref.subview %65[%80, %69] [%79, %70] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %73 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_32 = memref.collapse_shape %65 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_32 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview_25 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_26 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %69 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %81 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %82 = hivm.hir.pointer_cast(%c9728_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%65, %subview : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%81 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%82 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %83 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %84 = hivm.hir.pointer_cast(%c10240_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%81, %expand_shape : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%83 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%84 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_27 = memref.subview %53[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1], [2]] : memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_29 = memref.subview %collapse_shape_28[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %collapse_shape_30 = memref.collapse_shape %83 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_31 = memref.collapse_shape %subview_29 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_30 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_0, tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
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
