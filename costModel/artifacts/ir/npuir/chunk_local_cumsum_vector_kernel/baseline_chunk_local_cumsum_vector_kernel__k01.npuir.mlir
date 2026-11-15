// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @chunk_local_cumsum_vector_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c3712_i64 = arith.constant 3712 : i64
  %c2688_i64 = arith.constant 2688 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %c16 = arith.constant 16 : index
  %c96_i32 = arith.constant 96 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.index_cast %1 : i32 to index
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xi32, #hivm.address_space<ub>>
  hivm.hir.varange {vector_producer_to_fuse_0} offset[%c0] strides[%c1] outs(%5 : memref<32xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [1, 32] {vector_producer_to_fuse_0} : memref<32xi32, #hivm.address_space<ub>> into memref<1x32xi32, #hivm.address_space<ub>>
  %6 = hivm.hir.get_sub_block_idx -> i64
  %7 = arith.index_cast %6 : i64 to index
  %8 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%7]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  %9 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x32xf32, #hivm.address_space<gm>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c96_i32  : i32 {
    %10 = arith.index_cast %arg9 : i32 to index
    %11 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%10)[%4]
    scf.for %arg10 = %c0 to %11 step %c1 {
      %12 = arith.index_cast %arg10 : index to i64
      %subview = memref.subview %9[%arg10, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #hivm.address_space<gm>> to memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %12
      scf.for %arg11 = %c0 to %c32 step %c16 {
        %subview_1 = memref.subview %5[%arg11] [16] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<16xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %subview_1[%8] [8] [1] {to_be_bubbled_slice} : memref<16xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %expand_shape_3 = memref.expand_shape %subview_2 [[0, 1]] output_shape [8, 1] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<8x1xi32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        %13 = hivm.hir.pointer_cast(%c128_i64) : memref<8x32xi32, #hivm.address_space<ub>>
        %14 = hivm.hir.pointer_cast(%c1152_i64) : memref<64xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmax {vector_producer_to_fuse_0} ins(%expand_shape_3, %expand_shape : memref<8x1xi32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>, memref<1x32xi32, #hivm.address_space<ub>>) outs(%13 : memref<8x32xi32, #hivm.address_space<ub>>) temp_buffer(%14 : memref<64xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
        %15 = hivm.hir.pointer_cast(%c1408_i64) : memref<8x32xi32, #hivm.address_space<ub>>
        %16 = hivm.hir.pointer_cast(%c2432_i64) : memref<64xi32, #hivm.address_space<ub>>
        hivm.hir.vbrc {vector_producer_to_fuse_0} ins(%expand_shape_3 : memref<8x1xi32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%15 : memref<8x32xi32, #hivm.address_space<ub>>) temp_buffer(%16 : memref<64xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
        %17 = hivm.hir.pointer_cast(%c128_i64) : memref<8x32xi1, #hivm.address_space<ub>>
        %collapse_shape_4 = memref.collapse_shape %13 [[0, 1]] : memref<8x32xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
        %collapse_shape_5 = memref.collapse_shape %15 [[0, 1]] : memref<8x32xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
        %collapse_shape_6 = memref.collapse_shape %17 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vcmp {vector_producer_to_fuse_0} ins(%collapse_shape_4, %collapse_shape_5 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
        %18 = hivm.hir.pointer_cast(%c2688_i64) : memref<8x32xf32, #hivm.address_space<ub>>
        %collapse_shape_7 = memref.collapse_shape %18 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %19 = hivm.hir.pointer_cast(%c3712_i64) : memref<24xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vsel {vector_producer_to_fuse_0} ins(%collapse_shape_6, %cst, %cst_0 : memref<256xi1, #hivm.address_space<ub>>, f32, f32) outs(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%19 : memref<24xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_8 = memref.subview %collapse_shape[%arg11, 0] [16, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %subview_8[%8, 0] [8, 32] [1, 1] {to_be_bubbled_slice} : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_10 = memref.collapse_shape %subview_9 [[0, 1]] : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_10 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {op_to_tile_0_0, tiled_op}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      }
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 4
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @chunk_local_cumsum_vector_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c96_i32 = arith.constant 96 : i32
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x32xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  %5 = arith.index_cast %1 : i32 to index
  %6 = arith.muli %arg8, %arg7 : i32
  %7 = arith.index_cast %arg5 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c96_i32  : i32 {
    %8 = arith.index_cast %arg9 : i32 to index
    %9 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%8)[%5]
    scf.for %arg10 = %c0 to %9 step %c1 {
      %10 = arith.index_cast %arg9 : i32 to index
      %11 = arith.index_cast %3 : i32 to index
      %12 = arith.index_cast %1 : i32 to index
      %13 = arith.index_cast %c96_i32 : i32 to index
      %14 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> (s0 + ((s2 - s3) floordiv s4) * s1)>()[%arg10, %9, %10, %11, %13]
      %15 = arith.index_cast %14 : index to i64
      %16 = arith.index_cast %arg9 : i32 to index
      %17 = arith.index_cast %3 : i32 to index
      %18 = arith.index_cast %1 : i32 to index
      %19 = arith.index_cast %c96_i32 : i32 to index
      %20 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> ((s0 + ((s2 - s3) floordiv s4) * s1) mod 2)>()[%arg10, %9, %16, %17, %19]
      %21 = arith.index_cast %20 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %22 = arith.select %21, %c3_i64, %c4_i64 : i64
      %23 = arith.index_cast %20 : index to i1
      %c1_i64_1 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %24 = arith.select %23, %c1_i64_1, %c2_i64 : i64
      %25 = hivm.hir.pointer_cast(%c4096_i64, %c16384_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      %26 = hivm.hir.pointer_cast(%c0_i64_0, %c12288_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      %27 = arith.index_cast %arg10 : index to i64
      %28 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg10)[%8]
      %29 = arith.index_cast %28 : index to i32
      %30 = arith.remsi %29, %arg8 : i32
      %31 = arith.divsi %29, %arg8 : i32
      %32 = arith.remsi %31, %arg7 : i32
      %33 = arith.divsi %29, %6 : i32
      %34 = arith.remsi %33, %arg6 : i32
      %35 = arith.divsi %30, %c16_i32 : i32
      %36 = arith.remsi %30, %c16_i32 : i32
      %37 = arith.muli %35, %arg5 : i32
      %subview = memref.subview %4[%arg10, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
      %cast = memref.cast %26 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %24]
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %27
      %38 = arith.muli %37, %c16_i32 : i32
      %39 = arith.addi %38, %36 : i32
      %40 = arith.muli %39, %c64_i32 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = arith.muli %32, %c32_i32 : i32
      %43 = arith.muli %34, %c64_i32 : i32
      %44 = arith.maxsi %42, %c0_i32 : i32
      %45 = arith.index_cast %44 : i32 to index
      %46 = arith.maxsi %43, %c0_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%47, %41, %45]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_3 = memref.cast %25 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %49 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>()[%7, %47, %45]
      %50 = affine.min affine_map<()[s0] -> (32, s0)>()[%49]
      %51 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 64, 0)>()[%47]
      %52 = affine.min affine_map<()[s0] -> (64, s0)>()[%51]
      %53 = arith.subi %c0_i32, %42 : i32
      %54 = arith.maxsi %53, %c0_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%55, %49]
      %57 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %56]
      %58 = arith.subi %c0_i32, %43 : i32
      %59 = arith.maxsi %58, %c0_i32 : i32
      %60 = arith.index_cast %59 : i32 to index
      %61 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%60, %51]
      %62 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%52, %61]
      %63 = arith.cmpi slt, %57, %c32 : index
      %64 = arith.cmpi slt, %62, %c64 : index
      %65 = arith.ori %63, %64 : i1
      %subview_4 = memref.subview %reinterpret_cast[0, 0] [%57, %62] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %66 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%50, %56]
      %67 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 7) floordiv 8)>()[%52, %61]
      %68 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%56]
      %69 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%56]
      %70 = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%61]
      %71 = affine.apply affine_map<()[s0] -> (s0 mod 8)>()[%61]
      %subview_5 = memref.subview %25[%70, %68, %69, %71] [%67, %66, 16, 8] [1, 1, 1, 1] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_6 = memref.cast %subview_5 : memref<?x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %65 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %25 [[0, 1, 2, 3]] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %22]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_4 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %72 = hivm.hir.pointer_cast(%c0_i64_0) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
      %cast_7 = memref.cast %72 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_3, %true, %c32, %c32, %62 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %24, %22, %15, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %73 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%52, %61]
      %74 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%61]
      %75 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%61]
      %subview_8 = memref.subview %72[%74, %68, %69, %75] [%73, %66, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %cast_9 = memref.cast %subview_8 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_10 = memref.subview %reinterpret_cast_2[0, 0] [%57, %62] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_9 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_10 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
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
