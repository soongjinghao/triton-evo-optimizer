// -----// IR Dump After AddFFTSToSyncBlockSetOp (hivm-add-ffts-to-syncblocksetop) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c24 = arith.constant 24 : index
  %c1 = arith.constant 1 : index
  %c-1_i64 = arith.constant -1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c4_i64 = arith.constant 4 : i64
  %c3_i64 = arith.constant 3 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c63_i32 = arith.constant 63 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.addi %arg6, %c63_i32 : i32
  %6 = arith.divsi %5, %c64_i32 : i32
  %7 = arith.addi %arg7, %c63_i32 : i32
  %8 = arith.divsi %7, %c64_i32 : i32
  %9 = arith.muli %8, %c8_i32 : i32
  %10 = arith.addi %arg8, %c63_i32 : i32
  %11 = arith.divsi %10, %c64_i32 : i32
  %12 = arith.muli %arg10, %c64_i32 : i32
  %13 = arith.index_cast %arg9 : i32 to index
  %14 = arith.index_cast %arg10 : i32 to index
  %15 = arith.index_cast %arg6 : i32 to index
  %16 = arith.index_cast %arg7 : i32 to index
  %17 = arith.index_cast %12 : i32 to index
  %18 = arith.index_cast %arg11 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg15, %4 : i32
    %20 = arith.remsi %19, %arg12 : i32
    %21 = arith.divsi %20, %9 : i32
    %22 = arith.muli %21, %c8_i32 : i32
    %23 = arith.subi %6, %22 : i32
    %24 = arith.minsi %23, %c8_i32 : i32
    %25 = arith.remsi %20, %24 : i32
    %26 = arith.addi %22, %25 : i32
    %27 = arith.remsi %20, %9 : i32
    %28 = arith.divsi %27, %24 : i32
    %29 = arith.muli %26, %c64_i32 : i32
    %30 = arith.muli %28, %c64_i32 : i32
    %31 = arith.index_cast %29 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%31, %13]
    %33 = arith.index_cast %30 : i32 to index
    %34 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %15]
    %35 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%34, %31]
    %36 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%35, %31]
    %37 = affine.max affine_map<()[s0] -> (0, s0)>()[%36]
    %38 = arith.cmpi slt, %37, %c64 : index
    %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%33, %16]
    %40 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%39, %33]
    %41 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%40, %33]
    %42 = affine.max affine_map<()[s0] -> (0, s0)>()[%41]
    %43 = arith.cmpi slt, %42, %c64 : index
    %44 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %44 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %45:2 = scf.for %arg16 = %c0_i32 to %11 step %c1_i32 iter_args(%arg17 = %32, %arg18 = %c0) -> (index, index)  : i32 {
      %49 = arith.index_cast %arg16 : i32 to index
      %50 = arith.index_cast %11 : i32 to index
      %51 = arith.index_cast %arg15 : i32 to index
      %52 = arith.index_cast %3 : i32 to index
      %53 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3))>()[%49, %c0, %50, %c1, %51, %52, %c24]
      %54 = arith.index_cast %53 : index to i64
      %55 = arith.index_cast %arg16 : i32 to index
      %56 = arith.index_cast %11 : i32 to index
      %57 = arith.index_cast %arg15 : i32 to index
      %58 = arith.index_cast %3 : i32 to index
      %59 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%55, %c0, %56, %c1, %57, %58, %c24]
      %60 = arith.index_cast %59 : index to i1
      %61 = arith.select %60, %c3_i64, %c4_i64 : i64
      %62 = arith.index_cast %59 : index to i1
      %63 = arith.select %62, %c1_i64, %c2_i64 : i64
      %64 = hivm.hir.pointer_cast(%c16384_i64, %c49152_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %64 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %65 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %65 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %33]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%66], sizes: [64, 64], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %67 = arith.muli %arg16, %c64_i32 : i32
      %68 = arith.subi %arg8, %67 : i32
      %cast_4 = memref.cast %65 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %69 = arith.index_cast %68 : i32 to index
      %70 = affine.max affine_map<()[s0] -> (0, s0)>()[%69]
      %71 = affine.min affine_map<()[s0] -> (64, s0)>()[%70]
      %72 = affine.max affine_map<()[s0] -> (0, s0)>()[%71]
      %73 = arith.cmpi slt, %72, %c64 : index
      %74 = arith.ori %38, %73 : i1
      %subview_5 = memref.subview %reinterpret_cast_2[0, 0] [%37, %72] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %75 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%37]
      %76 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%72]
      %subview_6 = memref.subview %65[0, 0, 0, 0] [%76, %75, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_7 = memref.cast %subview_6 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %74 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %65 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %63]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %cast_8 = memref.cast %64 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %77 = arith.ori %73, %43 : i1
      %subview_9 = memref.subview %reinterpret_cast_3[0, 0] [%72, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %78 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%72]
      %79 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%42]
      %subview_10 = memref.subview %64[0, 0, 0, 0] [%79, %78, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_11 = memref.cast %subview_10 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %77 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %64 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %61]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_9 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %80 = arith.cmpi eq, %arg16, %c0_i32 : i32
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_4, %cast_8, %80, %37, %72, %42 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %63, %61, %54, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      %81 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%arg17]
      %82 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %17]
      scf.yield %81, %82 : index, index
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %46 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%33, %31, %18]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %47 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%37]
    %48 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%42]
    %subview = memref.subview %44[0, 0, 0, 0] [%48, %47, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
    %cast_0 = memref.cast %subview : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_1 = memref.subview %reinterpret_cast[0, 0] [%37, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
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

