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
    %21 = arith.remsi %arg14, %arg13 : i32
    %22 = arith.divsi %arg14, %arg13 : i32
    %23 = arith.remsi %22, %arg12 : i32
    %24 = arith.divsi %arg14, %4 : i32
    %25 = arith.remsi %24, %arg11 : i32
    %26 = arith.divsi %23, %c8_i32 : i32
    %27 = arith.remsi %23, %c8_i32 : i32
    %28 = arith.muli %26, %arg10 : i32
    %29 = arith.muli %28, %c8_i32 : i32
    %30 = arith.muli %25, %c32_i32 : i32
    %31 = arith.maxsi %30, %c0_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33 = arith.addi %29, %27 : i32
    %34 = arith.muli %33, %c32_i32 : i32
    %35 = arith.index_cast %34 : i32 to index
    %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%35, %32]
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%36], sizes: [32, 32], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %37 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %32]
    %38 = affine.min affine_map<()[s0] -> (32, s0)>()[%37]
    %39 = arith.subi %c0_i32, %30 : i32
    %40 = arith.maxsi %39, %c0_i32 : i32
    %41 = arith.index_cast %40 : i32 to index
    %42 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%41, %37]
    %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %42]
    %44 = arith.cmpi slt, %43, %c32 : index
    %cast = memref.cast %20 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%43, 32] [1, 1] : memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %45 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%38, %42]
    %46 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%42]
    %47 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%42]
    %subview_4 = memref.subview %20[0, %46, %47, 0] [4, %45, 16, 8] [1, 1, 1, 1] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_5 = memref.cast %subview_4 : memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %44 {
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      %collapse_shape = memref.collapse_shape %20 [[0, 1, 2, 3]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> into memref<1024xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<1024xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
    hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %48 = arith.muli %33, %c64_i32 : i32
    %49 = arith.index_cast %48 : i32 to index
    %50 = arith.muli %21, %c64_i32 : i32
    %51 = arith.maxsi %50, %c0_i32 : i32
    %52 = arith.index_cast %51 : i32 to index
    %53 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 512)>()[%52, %49, %32]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%53], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %54 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 512, 0)>()[%5, %52, %32]
    %55 = affine.min affine_map<()[s0] -> (32, s0)>()[%54]
    %56 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 512) * 512 + 64, 0)>()[%52]
    %57 = affine.min affine_map<()[s0] -> (64, s0)>()[%56]
    %58 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%41, %54]
    %59 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%55, %58]
    %60 = arith.subi %c0_i32, %50 : i32
    %61 = arith.maxsi %60, %c0_i32 : i32
    %62 = arith.index_cast %61 : i32 to index
    %63 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%62, %56]
    %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%57, %63]
    %65 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_7 = memref.cast %19 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%65 : memref<32x64xf32, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    %66 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    %cast_8 = memref.cast %66 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_7, %true, %43, %c32, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_8 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %11, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %67 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%55, %58]
    %68 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%57, %63]
    %69 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%58]
    %70 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%58]
    %71 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%63]
    %72 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%63]
    %subview_9 = memref.subview %66[%71, %69, %70, %72] [%68, %67, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
    %cast_10 = memref.cast %subview_9 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_11 = memref.subview %reinterpret_cast_6[0, 0] [%59, %64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_10 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_11 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %73 = arith.muli %33, %c128_i32 : i32
    %74 = arith.index_cast %73 : i32 to index
    %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
    scf.for %arg15 = %c0 to %c2 step %c1 {
      %76 = arith.index_cast %arg14 : i32 to index
      %77 = arith.index_cast %3 : i32 to index
      %78 = arith.index_cast %1 : i32 to index
      %79 = arith.index_cast %c24_i32 : i32 to index
      %80 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + ((s1 - s2) floordiv s3) * 2)>()[%arg15, %76, %77, %79]
      %81 = arith.index_cast %80 : index to i64
      %82 = arith.index_cast %arg14 : i32 to index
      %83 = arith.index_cast %3 : i32 to index
      %84 = arith.index_cast %1 : i32 to index
      %85 = arith.index_cast %c24_i32 : i32 to index
      %86 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 mod 2)>()[%arg15, %82, %83, %85]
      %87 = arith.index_cast %86 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %88 = arith.select %87, %c3_i64, %c4_i64 : i64
      %89 = hivm.hir.pointer_cast(%c12288_i64, %c32768_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %89 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      %90 = arith.index_cast %arg15 : index to i32
      %91 = arith.muli %90, %c64_i32 : i32
      %92 = arith.maxsi %91, %c0_i32 : i32
      %93 = arith.index_cast %92 : i32 to index
      %94 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%93, %74, %32]
      %reinterpret_cast_12 = memref.reinterpret_cast %arg6 to offset: [%94], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %95 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>()[%5, %93, %32]
      %96 = affine.min affine_map<()[s0] -> (32, s0)>()[%95]
      %97 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128, 0)>()[%93]
      %98 = affine.min affine_map<()[s0] -> (64, s0)>()[%97]
      %99 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%41, %95]
      %100 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%96, %99]
      %101 = arith.subi %c0_i32, %91 : i32
      %102 = arith.maxsi %101, %c0_i32 : i32
      %103 = arith.index_cast %102 : i32 to index
      %104 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%103, %97]
      %105 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%98, %104]
      %subview_13 = memref.subview %75[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_14 = memref.cast %89 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %88]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_13 : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_14 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %106 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
      %cast_15 = memref.cast %106 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64_16 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_14, %true, %43, %c32, %c64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_15 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_16, %c0_i64_0, %c-1_i64_16, %88, %81, %c-1_i64_16, %c-1_i64_16 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %107 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%96, %99]
      %108 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%98, %104]
      %109 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%99]
      %110 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%99]
      %111 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%104]
      %112 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%104]
      %subview_17 = memref.subview %106[%111, %109, %110, %112] [%108, %107, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_18 = memref.cast %subview_17 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_19 = memref.subview %reinterpret_cast_12[0, 0] [%100, %105] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_18 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_19 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>)
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
    %9 = arith.remsi %arg14, %arg13 : i32
    %10 = arith.divsi %arg14, %arg13 : i32
    %11 = arith.remsi %10, %arg12 : i32
    %12 = arith.divsi %arg14, %4 : i32
    %13 = arith.remsi %12, %arg11 : i32
    %14 = arith.divsi %11, %c8_i32 : i32
    %15 = arith.remsi %11, %c8_i32 : i32
    %16 = arith.muli %14, %arg10 : i32
    %17 = arith.muli %16, %c8_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %15 : i32 to index
    %20 = arith.muli %13, %c32_i32 : i32
    %21 = arith.maxsi %20, %c0_i32 : i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 8 + s1 + s2)>()[%22, %18, %19]
    %24 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %22]
    %25 = affine.min affine_map<()[s0] -> (32, s0)>()[%24]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%23], sizes: [32, 1], strides: [8, 8] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %26 = arith.addi %17, %15 : i32
    %27 = arith.index_cast %26 : i32 to index
    %28 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8)>()[%27, %22]
    %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%28], sizes: [32], strides: [8] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %29 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %29[0, 0, 0] [16, 1, 1] [1, 1, 1] : memref<16x8x1xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %30 = arith.subi %c0_i32, %20 : i32
    %31 = arith.maxsi %30, %c0_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%32, %24]
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%25, %33]
    %35 = arith.cmpi slt, %34, %c32 : index
    %36 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>()[%7, %25, %33]
    %37 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>()[%36, %25, %33]
    %subview_1 = memref.subview %reinterpret_cast[%8, 0] [16, 1] [1, 1] : memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> to memref<16x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %subview_1[0, 0] [%37, 1] [1, 1] : memref<16x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
    %38 = affine.max affine_map<()[s0, s1] -> (s1 * 16, s0)>()[%33, %7]
    %39 = affine.min affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>()[%24, %7]
    %40 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>()[%39, %33, %7]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %38]
    %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 16)>()[%38, %7]
    %subview_3 = memref.subview %subview[%42, 0] [%41, 1] [1, 1] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %35 {
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
    hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %33 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %43 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %subview_5 = memref.subview %43[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast_0[%8] [16] [1] : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %subview_6[0] [%37] [1] : memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %subview_5[%42] [%41] [1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
    scf.if %35 {
      %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %subview_5 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_27 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %33 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %44 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %subview_9 = memref.subview %44[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_5 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_9 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%reinterpret_cast_10 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %45 = arith.muli %26, %c64_i32 : i32
    %46 = arith.index_cast %45 : i32 to index
    %47 = arith.muli %9, %c64_i32 : i32
    %48 = arith.maxsi %47, %c0_i32 : i32
    %49 = arith.index_cast %48 : i32 to index
    %50 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 512)>()[%49, %46, %22]
    %51 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 512) * 512 + 64, 0)>()[%49]
    %52 = affine.min affine_map<()[s0] -> (64, s0)>()[%51]
    %53 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 512, 0)>()[%5, %49, %22]
    %54 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%32, %53]
    %55 = affine.min affine_map<()[s0] -> (32, s0)>()[%53]
    %56 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%55, %54]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %57 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %58 = arith.subi %c0_i32, %47 : i32
    %59 = arith.maxsi %58, %c0_i32 : i32
    %60 = arith.index_cast %59 : i32 to index
    %61 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%60, %51]
    %62 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%52, %61]
    %63 = arith.cmpi slt, %56, %c32 : index
    %64 = arith.cmpi slt, %62, %c64 : index
    %65 = arith.ori %63, %64 : i1
    %66 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>()[%7, %55, %54]
    %67 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>()[%66, %55, %54]
    %subview_17 = memref.subview %reinterpret_cast_16[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %subview_17[0, 0] [%67, %62] [1, 1] : memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %68 = affine.max affine_map<()[s0, s1] -> (s1 * 16, s0)>()[%54, %7]
    %69 = affine.min affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>()[%53, %7]
    %70 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>()[%69, %54, %7]
    %71 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%70, %68]
    %72 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 16)>()[%68, %7]
    %subview_19 = memref.subview %57[%72, %61] [%71, %62] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %65 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_23 = memref.collapse_shape %57 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_23 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%subview_18 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_19 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %61 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %73 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %74 = hivm.hir.pointer_cast(%c5120_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%57, %subview : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%73 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%74 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
    %subview_20 = memref.subview %75[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    %collapse_shape_21 = memref.collapse_shape %73 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %subview_20 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_21 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    annotation.mark %subview_20 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %76 = arith.muli %26, %c128_i32 : i32
    %77 = arith.index_cast %76 : i32 to index
    %78 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
    %expand_shape = memref.expand_shape %subview_9 [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
    scf.for %arg15 = %c0 to %c2 step %c1 {
      %79 = arith.index_cast %arg15 : index to i32
      %80 = arith.muli %79, %c64_i32 : i32
      %81 = arith.maxsi %80, %c0_i32 : i32
      %82 = arith.index_cast %81 : i32 to index
      %83 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%82, %77, %22]
      %84 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128, 0)>()[%82]
      %85 = affine.min affine_map<()[s0] -> (64, s0)>()[%84]
      %86 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>()[%5, %82, %22]
      %87 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%32, %86]
      %88 = affine.min affine_map<()[s0] -> (32, s0)>()[%86]
      %89 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%88, %87]
      %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %90 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %91 = arith.subi %c0_i32, %80 : i32
      %92 = arith.maxsi %91, %c0_i32 : i32
      %93 = arith.index_cast %92 : i32 to index
      %94 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%93, %84]
      %95 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%85, %94]
      %96 = arith.cmpi slt, %89, %c32 : index
      %97 = arith.cmpi slt, %95, %c64 : index
      %98 = arith.ori %96, %97 : i1
      %99 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>()[%7, %88, %87]
      %100 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>()[%99, %88, %87]
      %subview_24 = memref.subview %reinterpret_cast_23[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_25 = memref.subview %subview_24[0, 0] [%100, %95] [1, 1] : memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %101 = affine.max affine_map<()[s0, s1] -> (s1 * 16, s0)>()[%87, %7]
      %102 = affine.min affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>()[%86, %7]
      %103 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>()[%102, %87, %7]
      %104 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%103, %101]
      %105 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 16)>()[%101, %7]
      %subview_26 = memref.subview %90[%105, %94] [%104, %95] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %98 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_32 = memref.collapse_shape %90 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_32 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview_25 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_26 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %94 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %106 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %107 = hivm.hir.pointer_cast(%c9728_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%90, %subview : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%106 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%107 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %108 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %109 = hivm.hir.pointer_cast(%c10240_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul {vector_producer_to_fuse_0} ins(%106, %expand_shape : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%108 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%109 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_27 = memref.subview %78[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1], [2]] : memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_29 = memref.subview %collapse_shape_28[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %collapse_shape_30 = memref.collapse_shape %108 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
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
