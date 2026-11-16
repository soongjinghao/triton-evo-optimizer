// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c28672_i64 = arith.constant 28672 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c47_i32 = arith.constant 47 : i32
  %c31_i32 = arith.constant 31 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c48_i32 = arith.constant 48 : i32
  %c32_i32 = arith.constant 32 : i32
  %c63_i32 = arith.constant 63 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c48 = arith.constant 48 : index
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.addi %arg6, %c31_i32 : i32
  %6 = arith.divsi %5, %c32_i32 : i32
  %7 = arith.addi %arg7, %c47_i32 : i32
  %8 = arith.divsi %7, %c48_i32 : i32
  %9 = arith.addi %arg8, %c63_i32 : i32
  %10 = arith.divsi %9, %c64_i32 : i32
  %11 = arith.muli %arg10, %c64_i32 : i32
  %12 = arith.index_cast %arg9 : i32 to index
  %13 = arith.index_cast %arg10 : i32 to index
  %14 = arith.index_cast %arg6 : i32 to index
  %15 = arith.index_cast %arg7 : i32 to index
  %16 = arith.index_cast %11 : i32 to index
  %17 = arith.index_cast %arg11 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg15, %4 : i32
    %19 = arith.remsi %18, %arg12 : i32
    %20 = arith.divsi %19, %8 : i32
    %21 = arith.subi %6, %20 : i32
    %22 = arith.minsi %21, %c1_i32 : i32
    %23 = arith.remsi %19, %22 : i32
    %24 = arith.addi %20, %23 : i32
    %25 = arith.remsi %19, %8 : i32
    %26 = arith.divsi %25, %22 : i32
    %27 = arith.muli %24, %c32_i32 : i32
    %28 = arith.muli %26, %c48_i32 : i32
    %29 = arith.index_cast %27 : i32 to index
    %30 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%29, %12]
    %31 = arith.index_cast %28 : i32 to index
    %32 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %14]
    %33 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%32, %29]
    %34 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%33, %29]
    %35 = affine.max affine_map<()[s0] -> (0, s0)>()[%34]
    %36 = arith.cmpi slt, %35, %c32 : index
    %37 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %15]
    %38 = affine.min affine_map<()[s0, s1] -> (s1 + 48, s0)>()[%37, %31]
    %39 = affine.min affine_map<()[s0, s1] -> (48, s0 - s1)>()[%38, %31]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %41 = arith.cmpi slt, %40, %c48 : index
    %42 = hivm.hir.pointer_cast(%c0_i64_0) : memref<3x2x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %42 : memref<3x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %43:2 = scf.for %arg16 = %c0_i32 to %10 step %c1_i32 iter_args(%arg17 = %30, %arg18 = %c0) -> (index, index)  : i32 {
      %47 = arith.index_cast %arg16 : i32 to index
      %48 = arith.index_cast %c0_i32 : i32 to index
      %49 = arith.index_cast %10 : i32 to index
      %50 = arith.index_cast %c1_i32 : i32 to index
      %51 = arith.index_cast %arg15 : i32 to index
      %52 = arith.index_cast %3 : i32 to index
      %53 = arith.index_cast %1 : i32 to index
      %54 = arith.index_cast %c24_i32 : i32 to index
      %55 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3))>()[%47, %48, %49, %50, %51, %52, %54]
      %56 = arith.index_cast %55 : index to i64
      %57 = arith.index_cast %arg16 : i32 to index
      %58 = arith.index_cast %c0_i32 : i32 to index
      %59 = arith.index_cast %10 : i32 to index
      %60 = arith.index_cast %c1_i32 : i32 to index
      %61 = arith.index_cast %arg15 : i32 to index
      %62 = arith.index_cast %3 : i32 to index
      %63 = arith.index_cast %1 : i32 to index
      %64 = arith.index_cast %c24_i32 : i32 to index
      %65 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%57, %58, %59, %60, %61, %62, %64]
      %66 = arith.index_cast %65 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %67 = arith.select %66, %c3_i64, %c4_i64 : i64
      %68 = arith.index_cast %65 : index to i1
      %c1_i64_3 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %69 = arith.select %68, %c1_i64_3, %c2_i64 : i64
      %70 = hivm.hir.pointer_cast(%c8192_i64, %c28672_i64) : memref<6x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %70 {hivm.multi_buffer = 2 : i32} : memref<6x4x16x8xf32, #hivm.address_space<cbuf>>
      %71 = hivm.hir.pointer_cast(%c0_i64_0, %c20480_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %71 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [32, 64], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %72 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %31]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%72], sizes: [64, 48], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x48xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %73 = arith.muli %arg16, %c64_i32 : i32
      %74 = arith.subi %arg8, %73 : i32
      %cast_6 = memref.cast %71 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %75 = arith.index_cast %74 : i32 to index
      %76 = affine.max affine_map<()[s0] -> (0, s0)>()[%75]
      %77 = affine.min affine_map<()[s0] -> (64, s0)>()[%76]
      %78 = affine.max affine_map<()[s0] -> (0, s0)>()[%77]
      %79 = arith.cmpi slt, %78, %c64 : index
      %80 = arith.ori %36, %79 : i1
      %subview_7 = memref.subview %reinterpret_cast_4[0, 0] [%35, %78] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %81 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%35]
      %82 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%78]
      %subview_8 = memref.subview %71[0, 0, 0, 0] [%82, %81, 16, 8] [1, 1, 1, 1] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_9 = memref.cast %subview_8 : memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %80 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %71 [[0, 1, 2, 3]] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %69]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %cast_10 = memref.cast %70 : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %83 = arith.ori %79, %41 : i1
      %subview_11 = memref.subview %reinterpret_cast_5[0, 0] [%78, %40] [1, 1] : memref<64x48xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %84 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%78]
      %85 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%40]
      %subview_12 = memref.subview %70[0, 0, 0, 0] [%85, %84, 16, 8] [1, 1, 1, 1] : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_13 = memref.cast %subview_12 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %83 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %70 [[0, 1, 2, 3]] : memref<6x4x16x8xf32, #hivm.address_space<cbuf>> into memref<3072xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<3072xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %67]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %86 = arith.cmpi eq, %arg16, %c0_i32 : i32
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_6, %cast_10, %86, %35, %78, %40 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %69, %67, %56, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      %87 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%arg17]
      %88 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %16]
      scf.yield %87, %88 : index, index
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %44 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%31, %29, %17]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [32, 48], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x48xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %45 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%35]
    %46 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%40]
    %subview = memref.subview %42[0, 0, 0, 0] [%46, %45, 16, 16] [1, 1, 1, 1] : memref<3x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>>
    %cast_1 = memref.cast %subview : memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_2 = memref.subview %reinterpret_cast[0, 0] [%35, %40] [1, 1] : memref<32x48xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_1 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
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

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
