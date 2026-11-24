// -----// IR Dump After EnableMultiBuffer (hivm-enable-multi-buffer) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c24_i32 = arith.constant 24 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c63_i32 = arith.constant 63 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %c3_i64 = arith.constant 3 : i64
  %c4_i64 = arith.constant 4 : i64
  %c2_i64 = arith.constant 2 : i64
  %c-1_i64 = arith.constant -1 : i64
  %c1 = arith.constant 1 : index
  %c24 = arith.constant 24 : index
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i64 = arith.constant 0 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %0 = hivm.hir.pointer_cast(%c16384_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  %1 = hivm.hir.pointer_cast(%c49152_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  annotation.mark %1 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  %2 = hivm.hir.pointer_cast(%c0_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  annotation.mark %2 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  %3 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  annotation.mark %3 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
  %4 = arith.muli %arg12, %arg13 : i32
  %5 = arith.muli %4, %arg14 : i32
  annotation.mark %5 {logical_block_num} : i32
  %6 = hivm.hir.get_block_idx -> i64
  %7 = arith.trunci %6 : i64 to i32
  %8 = arith.muli %arg14, %arg13 : i32
  %9 = arith.addi %arg6, %c63_i32 : i32
  %10 = arith.divsi %9, %c64_i32 : i32
  %11 = arith.addi %arg7, %c63_i32 : i32
  %12 = arith.divsi %11, %c64_i32 : i32
  %13 = arith.muli %12, %c8_i32 : i32
  %14 = arith.addi %arg8, %c63_i32 : i32
  %15 = arith.divsi %14, %c64_i32 : i32
  %16 = arith.muli %arg10, %c64_i32 : i32
  %17 = arith.index_cast %arg9 : i32 to index
  %18 = arith.index_cast %arg10 : i32 to index
  %19 = arith.index_cast %arg6 : i32 to index
  %20 = arith.index_cast %arg7 : i32 to index
  %21 = arith.index_cast %16 : i32 to index
  %22 = arith.index_cast %arg11 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg15 = %7 to %5 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %23 = arith.divsi %arg15, %8 : i32
    %24 = arith.remsi %23, %arg12 : i32
    %25 = arith.divsi %24, %13 : i32
    %26 = arith.muli %25, %c8_i32 : i32
    %27 = arith.subi %10, %26 : i32
    %28 = arith.minsi %27, %c8_i32 : i32
    %29 = arith.remsi %24, %28 : i32
    %30 = arith.addi %26, %29 : i32
    %31 = arith.remsi %24, %13 : i32
    %32 = arith.divsi %31, %28 : i32
    %33 = arith.muli %30, %c64_i32 : i32
    %34 = arith.muli %32, %c64_i32 : i32
    %35 = arith.index_cast %33 : i32 to index
    %36 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%35, %17]
    %37 = arith.index_cast %34 : i32 to index
    %38 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%35, %19]
    %39 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%38, %35]
    %40 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%39, %35]
    %41 = affine.max affine_map<()[s0] -> (0, s0)>()[%40]
    %42 = arith.cmpi slt, %41, %c64 : index
    %43 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%37, %20]
    %44 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%43, %37]
    %45 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%44, %37]
    %46 = affine.max affine_map<()[s0] -> (0, s0)>()[%45]
    %47 = arith.cmpi slt, %46, %c64 : index
    %48 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %48 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %49:2 = scf.for %arg16 = %c0_i32 to %15 step %c1_i32 iter_args(%arg17 = %36, %arg18 = %c0) -> (index, index)  : i32 {
      %53 = arith.index_cast %arg16 : i32 to index
      %54 = arith.index_cast %15 : i32 to index
      %55 = arith.index_cast %arg15 : i32 to index
      %56 = arith.index_cast %7 : i32 to index
      %57 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%53, %c0, %54, %c1, %55, %56, %c24]
      %58 = arith.index_cast %57 : index to i64
      %59 = arith.cmpi eq, %58, %c1_i64 : i64
      %60 = arith.select %59, %1, %0 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %61 = arith.index_cast %arg16 : i32 to index
      %62 = arith.index_cast %15 : i32 to index
      %63 = arith.index_cast %arg15 : i32 to index
      %64 = arith.index_cast %7 : i32 to index
      %65 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%61, %c0, %62, %c1, %63, %64, %c24]
      %66 = arith.index_cast %65 : index to i64
      %67 = arith.cmpi eq, %66, %c1_i64 : i64
      %68 = arith.select %67, %3, %2 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %69 = arith.index_cast %arg16 : i32 to index
      %70 = arith.index_cast %15 : i32 to index
      %71 = arith.index_cast %arg15 : i32 to index
      %72 = arith.index_cast %7 : i32 to index
      %73 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3))>()[%69, %c0, %70, %c1, %71, %72, %c24]
      %74 = arith.index_cast %73 : index to i64
      %75 = arith.index_cast %arg16 : i32 to index
      %76 = arith.index_cast %15 : i32 to index
      %77 = arith.index_cast %arg15 : i32 to index
      %78 = arith.index_cast %7 : i32 to index
      %79 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%75, %c0, %76, %c1, %77, %78, %c24]
      %80 = arith.index_cast %79 : index to i1
      %81 = arith.select %80, %c3_i64, %c4_i64 : i64
      %82 = arith.index_cast %79 : index to i1
      %83 = arith.select %82, %c1_i64, %c2_i64 : i64
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %84 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %37]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%84], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %85 = arith.muli %arg16, %c64_i32 : i32
      %86 = arith.subi %arg8, %85 : i32
      %cast_4 = memref.cast %68 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %87 = arith.index_cast %86 : i32 to index
      %88 = affine.max affine_map<()[s0] -> (0, s0)>()[%87]
      %89 = affine.min affine_map<()[s0] -> (64, s0)>()[%88]
      %90 = affine.max affine_map<()[s0] -> (0, s0)>()[%89]
      %91 = arith.cmpi slt, %90, %c64 : index
      %92 = arith.ori %42, %91 : i1
      %subview_5 = memref.subview %reinterpret_cast_2[0, 0] [%41, %90] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %93 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%41]
      %94 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%90]
      %subview_6 = memref.subview %68[0, 0, 0, 0] [%94, %93, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_7 = memref.cast %subview_6 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %92 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %68 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %83]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %cast_8 = memref.cast %60 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %95 = arith.ori %91, %47 : i1
      %subview_9 = memref.subview %reinterpret_cast_3[0, 0] [%90, %46] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %96 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%90]
      %97 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%46]
      %subview_10 = memref.subview %60[0, 0, 0, 0] [%97, %96, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_11 = memref.cast %subview_10 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %95 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %60 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %81]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_9 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %98 = arith.cmpi eq, %arg16, %c0_i32 : i32
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_4, %cast_8, %98, %41, %90, %46 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %83, %81, %74, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      %99 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%arg17]
      %100 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %21]
      scf.yield %99, %100 : index, index
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %50 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%37, %35, %22]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%50], sizes: [64, 64], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %51 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%41]
    %52 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%46]
    %subview = memref.subview %48[0, 0, 0, 0] [%52, %51, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
    %cast_0 = memref.cast %subview : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_1 = memref.subview %reinterpret_cast[0, 0] [%41, %46] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_0 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

