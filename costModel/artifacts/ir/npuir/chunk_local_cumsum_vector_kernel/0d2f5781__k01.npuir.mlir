// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @chunk_local_cumsum_vector_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c3616_i64 = arith.constant 3616 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2336_i64 = arith.constant 2336 : i64
  %c1312_i64 = arith.constant 1312 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c256_i64 = arith.constant 256 : i64
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
  %6 = hivm.hir.pointer_cast(%c128_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast {vector_producer_to_fuse_0} ins(%5 : memref<32xi32, #hivm.address_space<ub>>) outs(%6 : memref<32xf32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %6 [[0, 1]] output_shape [1, 32] {vector_producer_to_fuse_0} : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c256_i64) : memref<8x32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc {vector_producer_to_fuse_0} ins(%expand_shape : memref<1x32xf32, #hivm.address_space<ub>>) outs(%7 : memref<8x32xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %8 = hivm.hir.get_sub_block_idx -> i64
  %9 = arith.index_cast %8 : i64 to index
  %10 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%9]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  %11 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x32x32xf32, #hivm.address_space<gm>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c96_i32  : i32 {
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%12)[%4]
    scf.for %arg10 = %c0 to %13 step %c1 {
      %14 = arith.index_cast %arg10 : index to i64
      %subview = memref.subview %11[%arg10, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #hivm.address_space<gm>> to memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %14
      scf.for %arg11 = %c0 to %c32 step %c16 {
        %subview_1 = memref.subview %5[%arg11] [16] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<16xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %subview_1[%10] [8] [1] {to_be_bubbled_slice} : memref<16xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %15 = hivm.hir.pointer_cast(%c1280_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.vcast {vector_producer_to_fuse_0} ins(%subview_2 : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%15 : memref<8xf32, #hivm.address_space<ub>>)
        %expand_shape_3 = memref.expand_shape %15 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
        %16 = hivm.hir.pointer_cast(%c1312_i64) : memref<8x32xf32, #hivm.address_space<ub>>
        %17 = hivm.hir.pointer_cast(%c2336_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc {vector_producer_to_fuse_0} ins(%expand_shape_3 : memref<8x1xf32, #hivm.address_space<ub>>) outs(%16 : memref<8x32xf32, #hivm.address_space<ub>>) temp_buffer(%17 : memref<64xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
        %18 = hivm.hir.pointer_cast(%c1312_i64) : memref<8x32xi1, #hivm.address_space<ub>>
        %collapse_shape_4 = memref.collapse_shape %16 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %collapse_shape_5 = memref.collapse_shape %7 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %collapse_shape_6 = memref.collapse_shape %18 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vcmp {vector_producer_to_fuse_0} ins(%collapse_shape_4, %collapse_shape_5 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) compare_mode = <ge>
        %19 = hivm.hir.pointer_cast(%c2592_i64) : memref<8x32xf32, #hivm.address_space<ub>>
        %collapse_shape_7 = memref.collapse_shape %19 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %20 = hivm.hir.pointer_cast(%c3616_i64) : memref<24xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vsel {vector_producer_to_fuse_0} ins(%collapse_shape_6, %cst, %cst_0 : memref<256xi1, #hivm.address_space<ub>>, f32, f32) outs(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%20 : memref<24xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_8 = memref.subview %collapse_shape[%arg11, 0] [16, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %subview_8[%10, 0] [8, 32] [1, 1] {to_be_bubbled_slice} : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
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
func.func @chunk_local_cumsum_vector_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c31_i32 = arith.constant 31 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c2_i32 = arith.constant 2 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
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
  %7 = arith.addi %arg5, %c31_i32 : i32
  %8 = arith.divsi %7, %c32_i32 : i32
  %9 = arith.addi %8, %c1_i32 : i32
  %10 = arith.divsi %9, %c2_i32 : i32
  %11 = arith.muli %10, %c32_i32 : i32
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c96_i32  : i32 {
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = affine.min affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>(%12)[%5]
    scf.for %arg10 = %c0 to %13 step %c1 {
      %14 = arith.index_cast %arg9 : i32 to index
      %15 = arith.index_cast %3 : i32 to index
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %c96_i32 : i32 to index
      %18 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> ((s0 + ((s2 - s3) floordiv s4) * s1) mod 2)>()[%arg10, %13, %14, %15, %17]
      %19 = arith.index_cast %18 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %20 = arith.select %19, %c0_i64_1, %c1_i64 : i64
      %21 = hivm.hir.pointer_cast(%c0_i64_0, %c12288_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      %22 = arith.index_cast %arg10 : index to i64
      %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 24 + s0)>(%arg10)[%12]
      %24 = arith.index_cast %23 : index to i32
      %25 = arith.divsi %24, %6 : i32
      %26 = arith.remsi %25, %arg6 : i32
      %subview = memref.subview %4[%arg10, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
      %cast = memref.cast %21 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %20]
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %22
      %27 = arith.cmpi slt, %26, %11 : i32
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      scf.if %27 {
        %28 = arith.remsi %26, %10 : i32
        %29 = arith.divsi %26, %10 : i32
        %30 = arith.muli %29, %c2_i32 : i32
        %31 = arith.addi %30, %c2_i32 : i32
        %32 = arith.minsi %31, %c64_i32 : i32
        %33 = arith.muli %28, %c2_i32 : i32
        %34 = arith.addi %33, %c2_i32 : i32
        %35 = arith.minsi %34, %8 : i32
        scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %36 = arith.addi %30, %arg11 : i32
          %37 = arith.cmpi slt, %36, %32 : i32
          scf.if %37 {
            %38 = arith.divsi %36, %c16_i32 : i32
            %39 = arith.remsi %36, %c16_i32 : i32
            scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
              %40 = arith.addi %33, %arg12 : i32
              %41 = arith.cmpi slt, %40, %35 : i32
              scf.if %41 {
                %42 = arith.muli %38, %arg5 : i32
                hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
                hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
                hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
                scf.for %arg13 = %c0_i32 to %c8_i32 step %c4_i32  : i32 {
                  %43 = arith.index_cast %arg13 : i32 to index
                  %44 = arith.index_cast %c0_i32 : i32 to index
                  %45 = arith.index_cast %c8_i32 : i32 to index
                  %46 = arith.index_cast %c4_i32 : i32 to index
                  %47 = arith.index_cast %arg12 : i32 to index
                  %48 = arith.index_cast %c0_i32 : i32 to index
                  %49 = arith.index_cast %c2_i32 : i32 to index
                  %50 = arith.index_cast %c1_i32 : i32 to index
                  %51 = arith.index_cast %arg11 : i32 to index
                  %52 = arith.index_cast %c0_i32 : i32 to index
                  %53 = arith.index_cast %c2_i32 : i32 to index
                  %54 = arith.index_cast %c1_i32 : i32 to index
                  %55 = arith.index_cast %arg9 : i32 to index
                  %56 = arith.index_cast %3 : i32 to index
                  %57 = arith.index_cast %1 : i32 to index
                  %58 = arith.index_cast %c96_i32 : i32 to index
                  %59 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s11) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) + s12 * ((((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) * ((-s9 + s10 + s11 - 1) floordiv s11)) + ((s14 - s15) floordiv s16) * (((((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) * ((-s9 + s10 + s11 - 1) floordiv s11)) * s13))>()[%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %arg10, %13, %55, %56, %58]
                  %60 = arith.index_cast %59 : index to i64
                  %61 = arith.index_cast %arg13 : i32 to index
                  %62 = arith.index_cast %c0_i32 : i32 to index
                  %63 = arith.index_cast %c8_i32 : i32 to index
                  %64 = arith.index_cast %c4_i32 : i32 to index
                  %65 = arith.index_cast %arg12 : i32 to index
                  %66 = arith.index_cast %c0_i32 : i32 to index
                  %67 = arith.index_cast %c2_i32 : i32 to index
                  %68 = arith.index_cast %c1_i32 : i32 to index
                  %69 = arith.index_cast %arg11 : i32 to index
                  %70 = arith.index_cast %c0_i32 : i32 to index
                  %71 = arith.index_cast %c2_i32 : i32 to index
                  %72 = arith.index_cast %c1_i32 : i32 to index
                  %73 = arith.index_cast %arg9 : i32 to index
                  %74 = arith.index_cast %3 : i32 to index
                  %75 = arith.index_cast %1 : i32 to index
                  %76 = arith.index_cast %c96_i32 : i32 to index
                  %77 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s11) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) + s12 * ((((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) * ((-s9 + s10 + s11 - 1) floordiv s11)) + ((s14 - s15) floordiv s16) * (((((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) * ((-s9 + s10 + s11 - 1) floordiv s11)) * s13)) mod 2)>()[%61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %arg10, %13, %73, %74, %76]
                  %78 = arith.index_cast %77 : index to i1
                  %c2_i64 = arith.constant 2 : i64
                  %c3_i64 = arith.constant 3 : i64
                  %79 = arith.select %78, %c2_i64, %c3_i64 : i64
                  %80 = hivm.hir.pointer_cast(%c4096_i64, %c16384_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
                  annotation.mark %80 {hivm.multi_buffer = 2 : i32} : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
                  %81 = arith.cmpi slt, %arg13, %c1_i32 : i32
                  scf.if %81 {
                    %82 = arith.muli %42, %c16_i32 : i32
                    %83 = arith.addi %82, %39 : i32
                    %84 = arith.muli %83, %c64_i32 : i32
                    %85 = arith.index_cast %84 : i32 to index
                    %86 = arith.muli %40, %c32_i32 : i32
                    %87 = arith.muli %arg13, %c64_i32 : i32
                    %88 = arith.maxsi %86, %c0_i32 : i32
                    %89 = arith.index_cast %88 : i32 to index
                    %90 = arith.maxsi %87, %c0_i32 : i32
                    %91 = arith.index_cast %90 : i32 to index
                    %92 = arith.index_cast %arg5 : i32 to index
                    %93 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>()[%91, %85, %89]
                    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%93], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%93], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                    %94 = arith.cmpi slt, %86, %arg5 : i32
                    %95 = arith.cmpi slt, %87, %c64_i32 : i32
                    %96 = arith.andi %94, %95 : i1
                    scf.if %96 {
                      %cast_3 = memref.cast %80 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
                      %97 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>()[%92, %91, %89]
                      %98 = affine.min affine_map<()[s0] -> (32, s0)>()[%97]
                      %99 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 64, 0)>()[%91]
                      %100 = affine.min affine_map<()[s0] -> (64, s0)>()[%99]
                      %101 = arith.subi %c0_i32, %86 : i32
                      %102 = arith.maxsi %101, %c0_i32 : i32
                      %103 = arith.index_cast %102 : i32 to index
                      %104 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%103, %97]
                      %105 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%98, %104]
                      %106 = arith.subi %c0_i32, %87 : i32
                      %107 = arith.maxsi %106, %c0_i32 : i32
                      %108 = arith.index_cast %107 : i32 to index
                      %109 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%108, %99]
                      %110 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%100, %109]
                      %111 = arith.cmpi slt, %105, %c32 : index
                      %112 = arith.cmpi slt, %110, %c64 : index
                      %113 = arith.ori %111, %112 : i1
                      %subview_4 = memref.subview %reinterpret_cast[0, 0] [%105, %110] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                      %114 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%98, %104]
                      %115 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 7) floordiv 8)>()[%100, %109]
                      %116 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%104]
                      %117 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%104]
                      %118 = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%109]
                      %119 = affine.apply affine_map<()[s0] -> (s0 mod 8)>()[%109]
                      %subview_5 = memref.subview %80[%118, %116, %117, %119] [%115, %114, 16, 8] [1, 1, 1, 1] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
                      %cast_6 = memref.cast %subview_5 : memref<?x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
                      scf.if %113 {
                        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
                        %collapse_shape = memref.collapse_shape %80 [[0, 1, 2, 3]] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
                        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
                        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>>)
                        hivm.hir.pipe_barrier[<PIPE_MTE2>]
                      } {hivm.unlikely_condition}
                      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %79]
                      hivm.hir.nd2nz {dst_continuous} ins(%subview_4 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
                      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
                      %120 = hivm.hir.pointer_cast(%c0_i64_0) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
                      %cast_7 = memref.cast %120 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
                      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
                      %c-1_i64 = arith.constant -1 : i64
                      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast, %cast_3, %true, %c32, %c32, %110 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_7 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64, %c0_i64, %c-1_i64, %79, %60, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
                      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
                      %121 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%100, %109]
                      %122 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%109]
                      %123 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%109]
                      %subview_8 = memref.subview %120[%122, %116, %117, %123] [%121, %114, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
                      %cast_9 = memref.cast %subview_8 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
                      %subview_10 = memref.subview %reinterpret_cast_2[0, 0] [%105, %110] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
                      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_9 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_10 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>)
                      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
                    }
                  }
                } {fixpipe_for_mmad_result_already_inserted = true}
                hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
                hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
                hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
              }
            }
          }
        }
      } {pipeline.cubeonly}
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %20]
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
  }
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
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
