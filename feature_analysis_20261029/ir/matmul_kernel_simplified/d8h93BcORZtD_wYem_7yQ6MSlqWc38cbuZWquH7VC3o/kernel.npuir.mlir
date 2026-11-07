// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c63_i32 = arith.constant 63 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg12 : i32
  %7 = arith.addi %arg6, %c63_i32 : i32
  %8 = arith.divsi %7, %c64_i32 : i32
  %9 = arith.addi %arg7, %c63_i32 : i32
  %10 = arith.divsi %9, %c64_i32 : i32
  %11 = arith.muli %10, %c8_i32 : i32
  %12 = arith.divsi %6, %11 : i32
  %13 = arith.muli %12, %c8_i32 : i32
  %14 = arith.subi %8, %13 : i32
  %15 = arith.minsi %14, %c8_i32 : i32
  %16 = arith.remsi %6, %15 : i32
  %17 = arith.addi %13, %16 : i32
  %18 = arith.remsi %6, %11 : i32
  %19 = arith.divsi %18, %15 : i32
  %20 = arith.muli %17, %c64_i32 : i32
  %21 = arith.muli %19, %c64_i32 : i32
  %22 = arith.addi %arg8, %c63_i32 : i32
  %23 = arith.divsi %22, %c64_i32 : i32
  %24 = arith.muli %arg10, %c64_i32 : i32
  %25 = arith.index_cast %20 : i32 to index
  %26 = arith.index_cast %arg9 : i32 to index
  %27 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%25, %26]
  %28 = arith.index_cast %arg10 : i32 to index
  %29 = arith.index_cast %21 : i32 to index
  %30 = arith.index_cast %arg6 : i32 to index
  %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%25, %30]
  %32 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%31, %25]
  %33 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%32, %25]
  %34 = arith.cmpi slt, %33, %c64 : index
  %35 = arith.index_cast %arg7 : i32 to index
  %36 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %35]
  %37 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%36, %29]
  %38 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%37, %29]
  %39 = arith.cmpi slt, %38, %c64 : index
  %40 = arith.index_cast %24 : i32 to index
  %41 = hivm.hir.pointer_cast(%c0_i64_0) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
  %cast = memref.cast %41 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  %42:2 = scf.for %arg15 = %c0_i32 to %23 step %c1_i32 iter_args(%arg16 = %27, %arg17 = %c0) -> (index, index)  : i32 {
    %47 = arith.index_cast %arg15 : i32 to index
    %48 = arith.index_cast %c0_i32 : i32 to index
    %49 = arith.index_cast %23 : i32 to index
    %50 = arith.index_cast %c1_i32 : i32 to index
    %51 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%47, %48, %50]
    %52 = arith.index_cast %51 : index to i64
    %53 = arith.index_cast %arg15 : i32 to index
    %54 = arith.index_cast %c0_i32 : i32 to index
    %55 = arith.index_cast %23 : i32 to index
    %56 = arith.index_cast %c1_i32 : i32 to index
    %57 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%53, %54, %56]
    %58 = arith.index_cast %57 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %59 = arith.select %58, %c3_i64, %c4_i64 : i64
    %60 = arith.index_cast %57 : index to i1
    %c1_i64_3 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %61 = arith.select %60, %c1_i64_3, %c2_i64 : i64
    %62 = hivm.hir.pointer_cast(%c16384_i64, %c49152_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %62 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %63 = hivm.hir.pointer_cast(%c0_i64_0, %c32768_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %63 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %64 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg17, %29]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [64, 64], strides: [%28, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%arg16], sizes: [64, 64], strides: [%26, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %65 = arith.muli %arg15, %c64_i32 : i32
    %66 = arith.subi %arg8, %65 : i32
    %cast_6 = memref.cast %63 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %67 = arith.index_cast %66 : i32 to index
    %68 = affine.max affine_map<()[s0] -> (0, s0)>()[%67]
    %69 = affine.min affine_map<()[s0] -> (64, s0)>()[%68]
    %70 = arith.cmpi slt, %69, %c64 : index
    %71 = arith.ori %34, %70 : i1
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%33, %69] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %72 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%33]
    %73 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%69]
    %subview_8 = memref.subview %63[0, 0, 0, 0] [%73, %72, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
    %cast_9 = memref.cast %subview_8 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %71 {
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      %collapse_shape = memref.collapse_shape %63 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %61]
    hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %cast_10 = memref.cast %62 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %74 = arith.ori %70, %39 : i1
    %subview_11 = memref.subview %reinterpret_cast_4[0, 0] [%69, %38] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %75 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%69]
    %76 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%38]
    %subview_12 = memref.subview %62[0, 0, 0, 0] [%76, %75, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
    %cast_13 = memref.cast %subview_12 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %74 {
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      %collapse_shape = memref.collapse_shape %62 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %59]
    hivm.hir.nd2nz {dst_continuous} ins(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %77 = arith.cmpi eq, %arg15, %c0_i32 : i32
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_6, %cast_10, %77, %33, %69, %38 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %61, %59, %52, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    %78 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%arg16]
    %79 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg17, %40]
    scf.yield %78, %79 : index, index
  }
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  %43 = arith.index_cast %arg11 : i32 to index
  %44 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%29, %25, %43]
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [64, 64], strides: [%43, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %45 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%33]
  %46 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%38]
  %subview = memref.subview %41[0, 0, 0, 0] [%46, %45, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
  %cast_1 = memref.cast %subview : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
  %subview_2 = memref.subview %reinterpret_cast[0, 0] [%33, %38] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_1 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
