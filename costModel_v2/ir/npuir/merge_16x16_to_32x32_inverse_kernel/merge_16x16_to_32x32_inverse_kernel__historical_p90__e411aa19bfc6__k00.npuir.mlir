// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_32x32_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c0_i64_2 = arith.constant 0 : i64
  %c1_i64_3 = arith.constant 1 : i64
  %c0_i64_4 = arith.constant 0 : i64
  %c0_i64_5 = arith.constant 0 : i64
  %c1_i64_6 = arith.constant 1 : i64
  %c0_i64_7 = arith.constant 0 : i64
  %c0_i64_8 = arith.constant 0 : i64
  %c1_i64_9 = arith.constant 1 : i64
  %c31744_i64 = arith.constant 31744 : i64
  %c15360_i64 = arith.constant 15360 : i64
  %c30720_i64 = arith.constant 30720 : i64
  %c14336_i64 = arith.constant 14336 : i64
  %c29696_i64 = arith.constant 29696 : i64
  %c13312_i64 = arith.constant 13312 : i64
  %c28672_i64 = arith.constant 28672 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c27648_i64 = arith.constant 27648 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c26624_i64 = arith.constant 26624 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c25600_i64 = arith.constant 25600 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c23552_i64 = arith.constant 23552 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c22528_i64 = arith.constant 22528 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c21504_i64 = arith.constant 21504 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c19456_i64 = arith.constant 19456 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64_10 = arith.constant 0 : i64
  %c24_i32 = arith.constant 24 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c-16_i32 = arith.constant -16 : i32
  %c128_i32 = arith.constant 128 : i32
  %c31_i32 = arith.constant 31 : i32
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  %c3072 = arith.constant 3072 : index
  %c4096 = arith.constant 4096 : index
  %c5120 = arith.constant 5120 : index
  %c6144 = arith.constant 6144 : index
  %c7168 = arith.constant 7168 : index
  %c8192 = arith.constant 8192 : index
  %c9216 = arith.constant 9216 : index
  %c10240 = arith.constant 10240 : index
  %c11264 = arith.constant 11264 : index
  %c12288 = arith.constant 12288 : index
  %c13312 = arith.constant 13312 : index
  %c14336 = arith.constant 14336 : index
  %c15360 = arith.constant 15360 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 10
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.addi %arg6, %c31_i32 : i32
  %6 = arith.divsi %5, %c32_i32 : i32
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  scf.for %arg10 = %3 to %1 step %c24_i32  : i32 {
    %7 = arith.index_cast %arg10 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c24_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i64
    %13 = arith.index_cast %arg10 : i32 to index
    %14 = arith.index_cast %3 : i32 to index
    %15 = arith.index_cast %1 : i32 to index
    %16 = arith.index_cast %c24_i32 : i32 to index
    %17 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%13, %14, %16]
    %18 = arith.index_cast %17 : index to i1
    %c0_i64_11 = arith.constant 0 : i64
    %c1_i64_12 = arith.constant 1 : i64
    %19 = arith.select %18, %c0_i64_11, %c1_i64_12 : i64
    %20 = arith.index_cast %17 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %21 = arith.select %20, %c2_i64, %c3_i64 : i64
    %22 = arith.index_cast %17 : index to i1
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %23 = arith.select %22, %c4_i64, %c5_i64 : i64
    %24 = arith.index_cast %17 : index to i1
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %25 = arith.select %24, %c6_i64, %c7_i64 : i64
    %26 = hivm.hir.pointer_cast(%c15360_i64, %c31744_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %27 = hivm.hir.pointer_cast(%c14336_i64, %c30720_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %28 = hivm.hir.pointer_cast(%c13312_i64, %c29696_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %29 = hivm.hir.pointer_cast(%c12288_i64, %c28672_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %30 = hivm.hir.pointer_cast(%c11264_i64, %c27648_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %31 = hivm.hir.pointer_cast(%c10240_i64, %c26624_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %32 = hivm.hir.pointer_cast(%c9216_i64, %c25600_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %33 = hivm.hir.pointer_cast(%c8192_i64, %c24576_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %34 = hivm.hir.pointer_cast(%c7168_i64, %c23552_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %35 = hivm.hir.pointer_cast(%c6144_i64, %c22528_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %36 = hivm.hir.pointer_cast(%c5120_i64, %c21504_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %37 = hivm.hir.pointer_cast(%c4096_i64, %c20480_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %38 = hivm.hir.pointer_cast(%c3072_i64, %c19456_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %39 = hivm.hir.pointer_cast(%c2048_i64, %c18432_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %40 = hivm.hir.pointer_cast(%c1024_i64, %c17408_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %41 = hivm.hir.pointer_cast(%c0_i64_10, %c16384_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_mask_norm
    %42 = arith.divsi %arg10, %arg9 : i32
    %43 = arith.remsi %42, %arg8 : i32
    %44 = arith.divsi %arg10, %4 : i32
    %45 = arith.remsi %44, %arg7 : i32
    %46 = arith.divsi %43, %c4_i32 : i32
    %47 = arith.remsi %43, %c4_i32 : i32
    %48 = arith.muli %45, %c4_i32 : i32
    %49 = arith.subi %6, %48 : i32
    %50 = arith.minsi %49, %c4_i32 : i32
    %51 = arith.muli %46, %arg6 : i32
    %52 = arith.muli %51, %c4_i32 : i32
    %53 = arith.addi %52, %47 : i32
    %54 = arith.muli %53, %c32_i32 : i32
    %55 = arith.index_cast %54 : i32 to index
    %56 = arith.cmpi sgt, %50, %c0_i32 : i32
    scf.if %56 {
      %60 = arith.muli %45, %c128_i32 : i32
      %61 = arith.index_cast %arg6 : i32 to index
      %62 = arith.addi %60, %c16_i32 : i32
      %63 = arith.maxsi %62, %c0_i32 : i32
      %64 = arith.index_cast %63 : i32 to index
      %65 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%55, %64]
      %66 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%61, %64]
      %67 = affine.min affine_map<()[s0] -> (16, s0)>()[%66]
      %68 = arith.subi %c-16_i32, %60 : i32
      %69 = arith.maxsi %68, %c0_i32 : i32
      %70 = arith.index_cast %69 : i32 to index
      %71 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%70, %66]
      %72 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%67, %71]
      %73 = arith.cmpi slt, %72, %c16 : index
      %74 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast = memref.cast %41 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %25]
      hivm.hir.nd2nz {dst_continuous} ins(%74 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11
      %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast_13 = memref.cast %40 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%75 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 10
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%65], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %39 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%72, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %76 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%67, %71]
      %77 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%71]
      %78 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%71]
      %subview_15 = memref.subview %39[0, %77, %78, 0] [2, %76, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_16 = memref.cast %subview_15 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %73 {
        %collapse_shape = memref.collapse_shape %39 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %79 = hivm.hir.pointer_cast(%c0_i64_10) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_17 = memref.cast %79 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_13, %cast_14, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_8, %c1_i64_9, %c-1_i64, %c-1_i64, %12, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %80 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%80 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_18 = memref.cast %38 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%80 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
      %81 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_19 = memref.cast %81 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      %c-1_i64_20 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_18, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_7, %c-1_i64_20, %c-1_i64_20, %25, %12, %c-1_i64_20, %c-1_i64_20 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %82 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%82 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
      annotation.mark %82 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
    }
    %57 = arith.cmpi sgt, %50, %c1_i32 : i32
    scf.if %57 {
      %60 = arith.addi %48, %c1_i32 : i32
      %61 = arith.muli %60, %c32_i32 : i32
      %62 = arith.index_cast %arg6 : i32 to index
      %63 = arith.addi %61, %c16_i32 : i32
      %64 = arith.maxsi %63, %c0_i32 : i32
      %65 = arith.index_cast %64 : i32 to index
      %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%55, %65]
      %67 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%62, %65]
      %68 = affine.min affine_map<()[s0] -> (16, s0)>()[%67]
      %69 = arith.subi %c-16_i32, %61 : i32
      %70 = arith.maxsi %69, %c0_i32 : i32
      %71 = arith.index_cast %70 : i32 to index
      %72 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%71, %67]
      %73 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%68, %72]
      %74 = arith.cmpi slt, %73, %c16 : index
      %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast = memref.cast %37 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %23]
      hivm.hir.nd2nz {dst_continuous} ins(%75 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
      %76 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast_13 = memref.cast %36 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%76 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%66], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %35 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%73, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %77 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%68, %72]
      %78 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%72]
      %79 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%72]
      %subview_15 = memref.subview %35[0, %78, %79, 0] [2, %77, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_16 = memref.cast %subview_15 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %74 {
        %collapse_shape = memref.collapse_shape %35 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %80 = hivm.hir.pointer_cast(%c2048_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_17 = memref.cast %80 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_13, %cast_14, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_5, %c1_i64_6, %c-1_i64, %c-1_i64, %12, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %81 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%81 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
      %cast_18 = memref.cast %34 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%81 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
      %82 = hivm.hir.pointer_cast(%c3072_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_19 = memref.cast %82 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
      %c-1_i64_20 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_18, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_4, %c-1_i64_20, %c-1_i64_20, %23, %12, %c-1_i64_20, %c-1_i64_20 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %83 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%83 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
      annotation.mark %83 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
    }
    %58 = arith.cmpi sgt, %50, %c2_i32 : i32
    scf.if %58 {
      %60 = arith.addi %48, %c2_i32 : i32
      %61 = arith.muli %60, %c32_i32 : i32
      %62 = arith.index_cast %arg6 : i32 to index
      %63 = arith.addi %61, %c16_i32 : i32
      %64 = arith.maxsi %63, %c0_i32 : i32
      %65 = arith.index_cast %64 : i32 to index
      %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%55, %65]
      %67 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%62, %65]
      %68 = affine.min affine_map<()[s0] -> (16, s0)>()[%67]
      %69 = arith.subi %c-16_i32, %61 : i32
      %70 = arith.maxsi %69, %c0_i32 : i32
      %71 = arith.index_cast %70 : i32 to index
      %72 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%71, %67]
      %73 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%68, %72]
      %74 = arith.cmpi slt, %73, %c16 : index
      %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast = memref.cast %33 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %21]
      hivm.hir.nd2nz {dst_continuous} ins(%75 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      %76 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast_13 = memref.cast %32 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%76 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%66], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %31 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%73, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %77 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%68, %72]
      %78 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%72]
      %79 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%72]
      %subview_15 = memref.subview %31[0, %78, %79, 0] [2, %77, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_16 = memref.cast %subview_15 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %74 {
        %collapse_shape = memref.collapse_shape %31 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %80 = hivm.hir.pointer_cast(%c4096_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_17 = memref.cast %80 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_13, %cast_14, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_2, %c1_i64_3, %c-1_i64, %c-1_i64, %12, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %81 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%81 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
      %cast_18 = memref.cast %30 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%81 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
      %82 = hivm.hir.pointer_cast(%c5120_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_19 = memref.cast %82 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
      %c-1_i64_20 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_18, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_1, %c-1_i64_20, %c-1_i64_20, %21, %12, %c-1_i64_20, %c-1_i64_20 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %83 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%83 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
      annotation.mark %83 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
    }
    %59 = arith.cmpi sgt, %50, %c3_i32 : i32
    scf.if %59 {
      %60 = arith.addi %48, %c3_i32 : i32
      %61 = arith.muli %60, %c32_i32 : i32
      %62 = arith.index_cast %arg6 : i32 to index
      %63 = arith.addi %61, %c16_i32 : i32
      %64 = arith.maxsi %63, %c0_i32 : i32
      %65 = arith.index_cast %64 : i32 to index
      %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%55, %65]
      %67 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%62, %65]
      %68 = affine.min affine_map<()[s0] -> (16, s0)>()[%67]
      %69 = arith.subi %c-16_i32, %61 : i32
      %70 = arith.maxsi %69, %c0_i32 : i32
      %71 = arith.index_cast %70 : i32 to index
      %72 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%71, %67]
      %73 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%68, %72]
      %74 = arith.cmpi slt, %73, %c16 : index
      %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c12288] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast = memref.cast %29 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %19]
      hivm.hir.nd2nz {dst_continuous} ins(%75 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      %76 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %cast_13 = memref.cast %28 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%76 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%66], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %27 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%73, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %77 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%68, %72]
      %78 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%72]
      %79 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%72]
      %subview_15 = memref.subview %27[0, %78, %79, 0] [2, %77, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_16 = memref.cast %subview_15 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %74 {
        %collapse_shape = memref.collapse_shape %27 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %80 = hivm.hir.pointer_cast(%c6144_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_17 = memref.cast %80 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_13, %cast_14, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c1_i64, %c-1_i64, %c-1_i64, %12, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %81 = memref_ext.alloc_workspace() from %arg2 offset = [%c14336] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%81 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %cast_18 = memref.cast %26 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%81 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
      %82 = hivm.hir.pointer_cast(%c7168_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_19 = memref.cast %82 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64_20 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_18, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64_20, %c-1_i64_20, %19, %12, %c-1_i64_20, %c-1_i64_20 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %83 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%83 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      annotation.mark %83 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
    }
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_32x32_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c32192_i64 = arith.constant 32192 : i64
  %c32096_i64 = arith.constant 32096 : i64
  %c31584_i64 = arith.constant 31584 : i64
  %c31072_i64 = arith.constant 31072 : i64
  %c31040_i64 = arith.constant 31040 : i64
  %c30528_i64 = arith.constant 30528 : i64
  %c30496_i64 = arith.constant 30496 : i64
  %c29984_i64 = arith.constant 29984 : i64
  %c29472_i64 = arith.constant 29472 : i64
  %c29408_i64 = arith.constant 29408 : i64
  %c28896_i64 = arith.constant 28896 : i64
  %c28864_i64 = arith.constant 28864 : i64
  %c28352_i64 = arith.constant 28352 : i64
  %c28320_i64 = arith.constant 28320 : i64
  %c27808_i64 = arith.constant 27808 : i64
  %c27296_i64 = arith.constant 27296 : i64
  %c27232_i64 = arith.constant 27232 : i64
  %c27168_i64 = arith.constant 27168 : i64
  %c26656_i64 = arith.constant 26656 : i64
  %c26592_i64 = arith.constant 26592 : i64
  %c26080_i64 = arith.constant 26080 : i64
  %c25568_i64 = arith.constant 25568 : i64
  %c25056_i64 = arith.constant 25056 : i64
  %c24544_i64 = arith.constant 24544 : i64
  %c24448_i64 = arith.constant 24448 : i64
  %c23936_i64 = arith.constant 23936 : i64
  %c23424_i64 = arith.constant 23424 : i64
  %c23392_i64 = arith.constant 23392 : i64
  %c22880_i64 = arith.constant 22880 : i64
  %c22848_i64 = arith.constant 22848 : i64
  %c22336_i64 = arith.constant 22336 : i64
  %c21824_i64 = arith.constant 21824 : i64
  %c21760_i64 = arith.constant 21760 : i64
  %c21248_i64 = arith.constant 21248 : i64
  %c21216_i64 = arith.constant 21216 : i64
  %c20704_i64 = arith.constant 20704 : i64
  %c20672_i64 = arith.constant 20672 : i64
  %c20160_i64 = arith.constant 20160 : i64
  %c19648_i64 = arith.constant 19648 : i64
  %c19584_i64 = arith.constant 19584 : i64
  %c19520_i64 = arith.constant 19520 : i64
  %c19008_i64 = arith.constant 19008 : i64
  %c18944_i64 = arith.constant 18944 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c17920_i64 = arith.constant 17920 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c16896_i64 = arith.constant 16896 : i64
  %c16800_i64 = arith.constant 16800 : i64
  %c16288_i64 = arith.constant 16288 : i64
  %c15776_i64 = arith.constant 15776 : i64
  %c15744_i64 = arith.constant 15744 : i64
  %c15232_i64 = arith.constant 15232 : i64
  %c15200_i64 = arith.constant 15200 : i64
  %c14688_i64 = arith.constant 14688 : i64
  %c14176_i64 = arith.constant 14176 : i64
  %c14112_i64 = arith.constant 14112 : i64
  %c13600_i64 = arith.constant 13600 : i64
  %c13568_i64 = arith.constant 13568 : i64
  %c13056_i64 = arith.constant 13056 : i64
  %c13024_i64 = arith.constant 13024 : i64
  %c12512_i64 = arith.constant 12512 : i64
  %c12000_i64 = arith.constant 12000 : i64
  %c11936_i64 = arith.constant 11936 : i64
  %c11872_i64 = arith.constant 11872 : i64
  %c11360_i64 = arith.constant 11360 : i64
  %c11296_i64 = arith.constant 11296 : i64
  %c10784_i64 = arith.constant 10784 : i64
  %c10272_i64 = arith.constant 10272 : i64
  %c9760_i64 = arith.constant 9760 : i64
  %c9248_i64 = arith.constant 9248 : i64
  %c9152_i64 = arith.constant 9152 : i64
  %c8640_i64 = arith.constant 8640 : i64
  %c8128_i64 = arith.constant 8128 : i64
  %c8096_i64 = arith.constant 8096 : i64
  %c7584_i64 = arith.constant 7584 : i64
  %c7552_i64 = arith.constant 7552 : i64
  %c7040_i64 = arith.constant 7040 : i64
  %c6528_i64 = arith.constant 6528 : i64
  %c6464_i64 = arith.constant 6464 : i64
  %c5952_i64 = arith.constant 5952 : i64
  %c5920_i64 = arith.constant 5920 : i64
  %c5408_i64 = arith.constant 5408 : i64
  %c5376_i64 = arith.constant 5376 : i64
  %c4864_i64 = arith.constant 4864 : i64
  %c4352_i64 = arith.constant 4352 : i64
  %c4288_i64 = arith.constant 4288 : i64
  %c4224_i64 = arith.constant 4224 : i64
  %c3712_i64 = arith.constant 3712 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c576_i64 = arith.constant 576 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c15360 = arith.constant 15360 : index
  %c13312 = arith.constant 13312 : index
  %c12288 = arith.constant 12288 : index
  %c11264 = arith.constant 11264 : index
  %c9216 = arith.constant 9216 : index
  %c8192 = arith.constant 8192 : index
  %c7168 = arith.constant 7168 : index
  %c5120 = arith.constant 5120 : index
  %c4096 = arith.constant 4096 : index
  %c3072 = arith.constant 3072 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c31_i32 = arith.constant 31 : i32
  %c128_i32 = arith.constant 128 : i32
  %c16384_i32 = arith.constant 16384 : i32
  %c18_i32 = arith.constant 18 : i32
  %c4096_i32 = arith.constant 4096 : i32
  %c-16_i32 = arith.constant -16 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.addi %arg6, %c31_i32 : i32
  %6 = arith.divsi %5, %c32_i32 : i32
  %7 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%7 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %7 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %8 = hivm.hir.pointer_cast(%c64_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %9 = hivm.hir.pointer_cast(%c576_i64) : memref<0xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%8 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%9 : memref<0xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %10 = hivm.hir.get_sub_block_idx -> i64
  %11 = arith.index_cast %10 : i64 to index
  %12 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%11]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
  %subview = memref.subview %7[%12] [8] [1] {to_be_bubbled_slice} : memref<16xi32, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %expand_shape_1 = memref.expand_shape %subview [[0, 1]] output_shape [1, 8] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
  %13 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%13 : memref<16x8xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %14 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c1600_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%14 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%15 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %16 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %14 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %13 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<128xi1, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %17 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %8 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %18 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  scf.for %arg10 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg10, %arg9 : i32
    %20 = arith.remsi %19, %arg8 : i32
    %21 = arith.divsi %arg10, %4 : i32
    %22 = arith.remsi %21, %arg7 : i32
    %23 = arith.divsi %20, %c4_i32 : i32
    %24 = arith.remsi %20, %c4_i32 : i32
    %25 = arith.muli %22, %c4_i32 : i32
    %26 = arith.subi %6, %25 : i32
    %27 = arith.minsi %26, %c4_i32 : i32
    %28 = arith.muli %23, %arg6 : i32
    %29 = arith.muli %28, %c4_i32 : i32
    %30 = arith.addi %29, %24 : i32
    %31 = arith.muli %30, %c32_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33 = arith.cmpi sgt, %27, %c0_i32 : i32
    scf.if %33 {
      %37 = arith.muli %22, %c128_i32 : i32
      %38 = arith.maxsi %37, %c0_i32 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %39]
      %41 = arith.index_cast %arg6 : i32 to index
      %42 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%41, %39]
      %43 = affine.min affine_map<()[s0] -> (16, s0)>()[%42]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %44 = arith.addi %37, %c16_i32 : i32
      %45 = arith.maxsi %44, %c0_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %46]
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%32, %46]
      %49 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%41, %46]
      %50 = affine.min affine_map<()[s0] -> (16, s0)>()[%49]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %51 = hivm.hir.pointer_cast(%c2112_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %52 = arith.subi %c0_i32, %37 : i32
      %53 = arith.maxsi %52, %c0_i32 : i32
      %54 = arith.index_cast %53 : i32 to index
      %55 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%54, %42]
      %56 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%43, %55]
      %57 = arith.cmpi slt, %56, %c16 : index
      %subview_8 = memref.subview %reinterpret_cast[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %subview_8[0, 0] [%56, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %51[%55, 0] [%56, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %57 {
        %collapse_shape_44 = memref.collapse_shape %51 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_9 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %58 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %59 = arith.subi %c-16_i32, %37 : i32
      %60 = arith.maxsi %59, %c0_i32 : i32
      %61 = arith.index_cast %60 : i32 to index
      %62 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%61, %49]
      %63 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %62]
      %64 = arith.cmpi slt, %63, %c16 : index
      %subview_11 = memref.subview %reinterpret_cast_7[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %subview_11[0, 0] [%63, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %58[%62, 0] [%63, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %64 {
        %collapse_shape_44 = memref.collapse_shape %58 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_12 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %65 = hivm.hir.pointer_cast(%c3136_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %51 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %65 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %66 = hivm.hir.pointer_cast(%c3648_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%66 : memref<16xf32, #hivm.address_space<ub>>)
      %67 = hivm.hir.pointer_cast(%c3136_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %67 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_15, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<128xf32, #hivm.address_space<ub>>)
      %68 = hivm.hir.pointer_cast(%c3136_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_16, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<128xf32, #hivm.address_space<ub>>)
      %69 = hivm.hir.pointer_cast(%c3712_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %58 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %70 = hivm.hir.pointer_cast(%c4224_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_18, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<16xf32, #hivm.address_space<ub>>)
      %71 = hivm.hir.pointer_cast(%c3712_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %71 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_19, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
      %72 = hivm.hir.pointer_cast(%c3712_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %72 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_20, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
      %73 = arith.subi %arg6, %37 : i32
      %74 = arith.minsi %73, %c16_i32 : i32
      %75 = arith.muli %22, %c16384_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c2_i32 to %74 step %c1_i32  : i32 {
        %95 = arith.muli %arg11, %c128_i32 : i32
        %96 = arith.index_cast %95 : i32 to index
        %97 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%96, %32, %76]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%97], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %98 = hivm.hir.pointer_cast(%c4288_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%98 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %99 = hivm.hir.pointer_cast(%c4288_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%98, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%99 : memref<16xf32, #hivm.address_space<ub>>)
        %100 = hivm.hir.pointer_cast(%c4288_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%99, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %100 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %101 = hivm.hir.pointer_cast(%c4352_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c4864_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %68 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%101 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%102 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %103 = hivm.hir.pointer_cast(%c5376_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %104 = hivm.hir.pointer_cast(%c5408_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%101 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%103 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%104 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %103 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %100[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c5920_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%105 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %expand_shape_48 = memref.expand_shape %105 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c5952_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%106 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %107 = arith.index_cast %arg11 : i32 to index
        %108 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%107]
        %109 = arith.cmpi slt, %107, %c0 : index
        %110 = arith.cmpi sge, %107, %c16 : index
        %111 = arith.ori %109, %110 : i1
        scf.if %111 {
        } else {
          %subview_49 = memref.subview %68[0, 0] [%107, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %106[0, 0] [%107, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %106[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %68[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %77 = arith.minsi %73, %c32_i32 : i32
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c18_i32 to %77 step %c1_i32  : i32 {
        %95 = arith.muli %arg11, %c128_i32 : i32
        %96 = arith.index_cast %95 : i32 to index
        %97 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 16)>()[%96, %32, %76]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%97], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %98 = hivm.hir.pointer_cast(%c6464_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%98 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %99 = hivm.hir.pointer_cast(%c6464_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%98, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%99 : memref<16xf32, #hivm.address_space<ub>>)
        %100 = hivm.hir.pointer_cast(%c6464_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%99, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %100 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %101 = hivm.hir.pointer_cast(%c6528_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c7040_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %72 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%101 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%102 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %103 = hivm.hir.pointer_cast(%c7552_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %104 = hivm.hir.pointer_cast(%c7584_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%101 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%103 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%104 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %103 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %100[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c8096_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%105 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %106 = arith.subi %arg11, %c16_i32 : i32
        %expand_shape_48 = memref.expand_shape %105 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %107 = hivm.hir.pointer_cast(%c8128_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%107 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %108 = arith.index_cast %106 : i32 to index
        %109 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%108]
        %110 = arith.cmpi slt, %108, %c0 : index
        %111 = arith.cmpi sge, %108, %c16 : index
        %112 = arith.ori %110, %111 : i1
        scf.if %112 {
        } else {
          %subview_49 = memref.subview %72[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %107[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %107[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %72[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %78 = hivm.hir.pointer_cast(%c8640_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %78 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %79 = hivm.hir.pointer_cast(%c9152_i64) : memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
      hivm.hir.vcast ins(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%79 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
      %80 = hivm.hir.pointer_cast(%c3136_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %80 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_17, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %81 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_24 = memref.subview %81[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%80 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %82 = hivm.hir.pointer_cast(%c8640_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %82 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_21, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %83 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_26 = memref.subview %83[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 10
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%82 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %84 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_27 = memref.subview %84[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %85 = hivm.hir.pointer_cast(%c9248_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_29 = memref.collapse_shape %85 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.load ins(%collapse_shape_28 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
      %86 = hivm.hir.pointer_cast(%c9248_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %86 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_29, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
      %87 = hivm.hir.pointer_cast(%c9248_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_31 = memref.collapse_shape %87 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_33 = memref.subview %reinterpret_cast_32[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_34 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_35 = memref.subview %reinterpret_cast_34[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_37 = memref.subview %reinterpret_cast_36[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_38 = memref.subview %80[%55, 0] [%56, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_39 = memref.subview %subview_33[0, 0] [%56, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_38 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
      %subview_40 = memref.subview %82[%62, 0] [%63, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_41 = memref.subview %subview_37[0, 0] [%63, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_40 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
      %88 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%62, %11]
      %89 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%49, %11]
      %90 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%89, %62, %11]
      %91 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%90, %88]
      %92 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%88, %11]
      %subview_42 = memref.subview %87[%92, 0] [%91, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %93 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%11, %50, %62]
      %94 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%93, %50, %62]
      %subview_43 = memref.subview %subview_35[0, 0] [%94, 16] [1, 1] : memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_42 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_43 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    }
    %34 = arith.cmpi sgt, %27, %c1_i32 : i32
    scf.if %34 {
      %37 = arith.addi %25, %c1_i32 : i32
      %38 = arith.muli %37, %c32_i32 : i32
      %39 = arith.maxsi %38, %c0_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %40]
      %42 = arith.index_cast %arg6 : i32 to index
      %43 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%42, %40]
      %44 = affine.min affine_map<()[s0] -> (16, s0)>()[%43]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.addi %38, %c16_i32 : i32
      %46 = arith.maxsi %45, %c0_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %47]
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%32, %47]
      %50 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%42, %47]
      %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %52 = hivm.hir.pointer_cast(%c9760_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %53 = arith.subi %c0_i32, %38 : i32
      %54 = arith.maxsi %53, %c0_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%55, %43]
      %57 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%44, %56]
      %58 = arith.cmpi slt, %57, %c16 : index
      %subview_8 = memref.subview %reinterpret_cast[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %subview_8[0, 0] [%57, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %52[%56, 0] [%57, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %58 {
        %collapse_shape_44 = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_9 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %59 = hivm.hir.pointer_cast(%c10272_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %60 = arith.subi %c-16_i32, %38 : i32
      %61 = arith.maxsi %60, %c0_i32 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%62, %50]
      %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%51, %63]
      %65 = arith.cmpi slt, %64, %c16 : index
      %subview_11 = memref.subview %reinterpret_cast_7[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %subview_11[0, 0] [%64, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %59[%63, 0] [%64, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %65 {
        %collapse_shape_44 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_12 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %66 = hivm.hir.pointer_cast(%c10784_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %66 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c11296_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
      %68 = hivm.hir.pointer_cast(%c10784_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_15, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<128xf32, #hivm.address_space<ub>>)
      %69 = hivm.hir.pointer_cast(%c10784_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_16, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<128xf32, #hivm.address_space<ub>>)
      %70 = hivm.hir.pointer_cast(%c11360_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %70 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %71 = hivm.hir.pointer_cast(%c11872_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_18, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<16xf32, #hivm.address_space<ub>>)
      %72 = hivm.hir.pointer_cast(%c11360_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %72 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_19, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
      %73 = hivm.hir.pointer_cast(%c11360_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %73 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_20, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
      %74 = arith.subi %arg6, %38 : i32
      %75 = arith.minsi %74, %c16_i32 : i32
      %76 = arith.muli %37, %c4096_i32 : i32
      %77 = arith.index_cast %76 : i32 to index
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c2_i32 to %75 step %c1_i32  : i32 {
        %96 = arith.muli %arg11, %c128_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%97, %32, %77]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %99 = hivm.hir.pointer_cast(%c11936_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %100 = hivm.hir.pointer_cast(%c11936_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%99, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %101 = hivm.hir.pointer_cast(%c11936_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%100, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%101 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c12000_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = hivm.hir.pointer_cast(%c12512_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %69 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%103 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %104 = hivm.hir.pointer_cast(%c13024_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c13056_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%102 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%104 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %104 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %101[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c13568_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%106 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %expand_shape_48 = memref.expand_shape %106 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %107 = hivm.hir.pointer_cast(%c13600_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%107 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %108 = arith.index_cast %arg11 : i32 to index
        %109 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%108]
        %110 = arith.cmpi slt, %108, %c0 : index
        %111 = arith.cmpi sge, %108, %c16 : index
        %112 = arith.ori %110, %111 : i1
        scf.if %112 {
        } else {
          %subview_49 = memref.subview %69[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %107[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %107[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %69[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %78 = arith.minsi %74, %c32_i32 : i32
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c18_i32 to %78 step %c1_i32  : i32 {
        %96 = arith.muli %arg11, %c128_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 16)>()[%97, %32, %77]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %99 = hivm.hir.pointer_cast(%c14112_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %100 = hivm.hir.pointer_cast(%c14112_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%99, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %101 = hivm.hir.pointer_cast(%c14112_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%100, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%101 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c14176_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = hivm.hir.pointer_cast(%c14688_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %73 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%103 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %104 = hivm.hir.pointer_cast(%c15200_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c15232_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%102 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%104 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %104 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %101[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c15744_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%106 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %107 = arith.subi %arg11, %c16_i32 : i32
        %expand_shape_48 = memref.expand_shape %106 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %108 = hivm.hir.pointer_cast(%c15776_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%108 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %109 = arith.index_cast %107 : i32 to index
        %110 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%109]
        %111 = arith.cmpi slt, %109, %c0 : index
        %112 = arith.cmpi sge, %109, %c16 : index
        %113 = arith.ori %111, %112 : i1
        scf.if %113 {
        } else {
          %subview_49 = memref.subview %73[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %108[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %108[0, 0] [%110, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %73[0, 0] [%110, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %79 = hivm.hir.pointer_cast(%c16288_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %79 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %80 = hivm.hir.pointer_cast(%c16800_i64) : memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%80 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
      %81 = hivm.hir.pointer_cast(%c10784_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %81 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_17, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %82 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_24 = memref.subview %82[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%81 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %83 = hivm.hir.pointer_cast(%c11360_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %83 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%collapse_shape_21, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %84 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_26 = memref.subview %84[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%83 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %85 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_27 = memref.subview %85[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %86 = hivm.hir.pointer_cast(%c16896_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_29 = memref.collapse_shape %86 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%collapse_shape_28 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
      %87 = hivm.hir.pointer_cast(%c16896_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %87 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_29, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
      %88 = hivm.hir.pointer_cast(%c16896_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_31 = memref.collapse_shape %88 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_33 = memref.subview %reinterpret_cast_32[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_34 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_35 = memref.subview %reinterpret_cast_34[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_37 = memref.subview %reinterpret_cast_36[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_38 = memref.subview %81[%56, 0] [%57, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_39 = memref.subview %subview_33[0, 0] [%57, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_38 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
      %subview_40 = memref.subview %83[%63, 0] [%64, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_41 = memref.subview %subview_37[0, 0] [%64, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_40 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
      %89 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%63, %11]
      %90 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%50, %11]
      %91 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%90, %63, %11]
      %92 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%91, %89]
      %93 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%89, %11]
      %subview_42 = memref.subview %88[%93, 0] [%92, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %94 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%11, %51, %63]
      %95 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%94, %51, %63]
      %subview_43 = memref.subview %subview_35[0, 0] [%95, 16] [1, 1] : memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_42 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_43 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    }
    %35 = arith.cmpi sgt, %27, %c2_i32 : i32
    scf.if %35 {
      %37 = arith.addi %25, %c2_i32 : i32
      %38 = arith.muli %37, %c32_i32 : i32
      %39 = arith.maxsi %38, %c0_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %40]
      %42 = arith.index_cast %arg6 : i32 to index
      %43 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%42, %40]
      %44 = affine.min affine_map<()[s0] -> (16, s0)>()[%43]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.addi %38, %c16_i32 : i32
      %46 = arith.maxsi %45, %c0_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %47]
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%32, %47]
      %50 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%42, %47]
      %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %52 = hivm.hir.pointer_cast(%c17408_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %53 = arith.subi %c0_i32, %38 : i32
      %54 = arith.maxsi %53, %c0_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%55, %43]
      %57 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%44, %56]
      %58 = arith.cmpi slt, %57, %c16 : index
      %subview_8 = memref.subview %reinterpret_cast[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %subview_8[0, 0] [%57, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %52[%56, 0] [%57, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %58 {
        %collapse_shape_44 = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_9 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %59 = hivm.hir.pointer_cast(%c17920_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %60 = arith.subi %c-16_i32, %38 : i32
      %61 = arith.maxsi %60, %c0_i32 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%62, %50]
      %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%51, %63]
      %65 = arith.cmpi slt, %64, %c16 : index
      %subview_11 = memref.subview %reinterpret_cast_7[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %subview_11[0, 0] [%64, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %59[%63, 0] [%64, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %65 {
        %collapse_shape_44 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_12 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %66 = hivm.hir.pointer_cast(%c18432_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %66 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c18944_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
      %68 = hivm.hir.pointer_cast(%c18432_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_15, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<128xf32, #hivm.address_space<ub>>)
      %69 = hivm.hir.pointer_cast(%c18432_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_16, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<128xf32, #hivm.address_space<ub>>)
      %70 = hivm.hir.pointer_cast(%c19008_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %70 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %71 = hivm.hir.pointer_cast(%c19520_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_18, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<16xf32, #hivm.address_space<ub>>)
      %72 = hivm.hir.pointer_cast(%c19008_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %72 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_19, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
      %73 = hivm.hir.pointer_cast(%c19008_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %73 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_20, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
      %74 = arith.subi %arg6, %38 : i32
      %75 = arith.minsi %74, %c16_i32 : i32
      %76 = arith.muli %37, %c4096_i32 : i32
      %77 = arith.index_cast %76 : i32 to index
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c2_i32 to %75 step %c1_i32  : i32 {
        %96 = arith.muli %arg11, %c128_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%97, %32, %77]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %99 = hivm.hir.pointer_cast(%c19584_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %100 = hivm.hir.pointer_cast(%c19584_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%99, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %101 = hivm.hir.pointer_cast(%c19584_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%100, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%101 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c19648_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = hivm.hir.pointer_cast(%c20160_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %69 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%103 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %104 = hivm.hir.pointer_cast(%c20672_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c20704_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%102 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%104 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %104 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %101[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c21216_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%106 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %expand_shape_48 = memref.expand_shape %106 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %107 = hivm.hir.pointer_cast(%c21248_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%107 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %108 = arith.index_cast %arg11 : i32 to index
        %109 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%108]
        %110 = arith.cmpi slt, %108, %c0 : index
        %111 = arith.cmpi sge, %108, %c16 : index
        %112 = arith.ori %110, %111 : i1
        scf.if %112 {
        } else {
          %subview_49 = memref.subview %69[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %107[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %107[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %69[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %78 = arith.minsi %74, %c32_i32 : i32
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c18_i32 to %78 step %c1_i32  : i32 {
        %96 = arith.muli %arg11, %c128_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 16)>()[%97, %32, %77]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %99 = hivm.hir.pointer_cast(%c21760_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %100 = hivm.hir.pointer_cast(%c21760_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%99, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %101 = hivm.hir.pointer_cast(%c21760_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%100, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%101 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c21824_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = hivm.hir.pointer_cast(%c22336_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %73 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%103 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %104 = hivm.hir.pointer_cast(%c22848_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c22880_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%102 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%104 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %104 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %101[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c23392_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%106 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %107 = arith.subi %arg11, %c16_i32 : i32
        %expand_shape_48 = memref.expand_shape %106 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %108 = hivm.hir.pointer_cast(%c23424_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%108 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %109 = arith.index_cast %107 : i32 to index
        %110 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%109]
        %111 = arith.cmpi slt, %109, %c0 : index
        %112 = arith.cmpi sge, %109, %c16 : index
        %113 = arith.ori %111, %112 : i1
        scf.if %113 {
        } else {
          %subview_49 = memref.subview %73[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %108[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %108[0, 0] [%110, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %73[0, 0] [%110, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %79 = hivm.hir.pointer_cast(%c23936_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %79 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %80 = hivm.hir.pointer_cast(%c24448_i64) : memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%80 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
      %81 = hivm.hir.pointer_cast(%c18432_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %81 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_17, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %82 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_24 = memref.subview %82[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%81 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %83 = hivm.hir.pointer_cast(%c19008_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %83 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%collapse_shape_21, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %84 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_26 = memref.subview %84[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%83 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %85 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_27 = memref.subview %85[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %86 = hivm.hir.pointer_cast(%c24544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_29 = memref.collapse_shape %86 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%collapse_shape_28 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      %87 = hivm.hir.pointer_cast(%c24544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %87 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_29, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
      %88 = hivm.hir.pointer_cast(%c24544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_31 = memref.collapse_shape %88 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_33 = memref.subview %reinterpret_cast_32[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_34 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_35 = memref.subview %reinterpret_cast_34[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_37 = memref.subview %reinterpret_cast_36[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_38 = memref.subview %81[%56, 0] [%57, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_39 = memref.subview %subview_33[0, 0] [%57, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_38 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
      %subview_40 = memref.subview %83[%63, 0] [%64, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_41 = memref.subview %subview_37[0, 0] [%64, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_40 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      %89 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%63, %11]
      %90 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%50, %11]
      %91 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%90, %63, %11]
      %92 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%91, %89]
      %93 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%89, %11]
      %subview_42 = memref.subview %88[%93, 0] [%92, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %94 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%11, %51, %63]
      %95 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%94, %51, %63]
      %subview_43 = memref.subview %subview_35[0, 0] [%95, 16] [1, 1] : memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_42 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_43 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    }
    %36 = arith.cmpi sgt, %27, %c3_i32 : i32
    scf.if %36 {
      %37 = arith.addi %25, %c3_i32 : i32
      %38 = arith.muli %37, %c32_i32 : i32
      %39 = arith.maxsi %38, %c0_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %40]
      %42 = arith.index_cast %arg6 : i32 to index
      %43 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%42, %40]
      %44 = affine.min affine_map<()[s0] -> (16, s0)>()[%43]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.addi %38, %c16_i32 : i32
      %46 = arith.maxsi %45, %c0_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %47]
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128 + 16)>()[%32, %47]
      %50 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%42, %47]
      %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %52 = hivm.hir.pointer_cast(%c25056_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %53 = arith.subi %c0_i32, %38 : i32
      %54 = arith.maxsi %53, %c0_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%55, %43]
      %57 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%44, %56]
      %58 = arith.cmpi slt, %57, %c16 : index
      %subview_8 = memref.subview %reinterpret_cast[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %subview_8[0, 0] [%57, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %52[%56, 0] [%57, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %58 {
        %collapse_shape_44 = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_9 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %59 = hivm.hir.pointer_cast(%c25568_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %60 = arith.subi %c-16_i32, %38 : i32
      %61 = arith.maxsi %60, %c0_i32 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%62, %50]
      %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%51, %63]
      %65 = arith.cmpi slt, %64, %c16 : index
      %subview_11 = memref.subview %reinterpret_cast_7[0, %12] [16, 8] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %subview_11[0, 0] [%64, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %59[%63, 0] [%64, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %65 {
        %collapse_shape_44 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_12 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %66 = hivm.hir.pointer_cast(%c26080_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %66 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c26592_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
      %68 = hivm.hir.pointer_cast(%c26080_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_15, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<128xf32, #hivm.address_space<ub>>)
      %69 = hivm.hir.pointer_cast(%c26080_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_16, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<128xf32, #hivm.address_space<ub>>)
      %70 = hivm.hir.pointer_cast(%c26656_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %59 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %70 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %71 = hivm.hir.pointer_cast(%c27168_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_18, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<16xf32, #hivm.address_space<ub>>)
      %72 = hivm.hir.pointer_cast(%c26656_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %72 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_19, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
      %73 = hivm.hir.pointer_cast(%c26656_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %73 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_20, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
      %74 = arith.subi %arg6, %38 : i32
      %75 = arith.minsi %74, %c16_i32 : i32
      %76 = arith.muli %37, %c4096_i32 : i32
      %77 = arith.index_cast %76 : i32 to index
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c2_i32 to %75 step %c1_i32  : i32 {
        %96 = arith.muli %arg11, %c128_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%97, %32, %77]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %99 = hivm.hir.pointer_cast(%c27232_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %100 = hivm.hir.pointer_cast(%c27232_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%99, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %101 = hivm.hir.pointer_cast(%c27232_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%100, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%101 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c27296_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = hivm.hir.pointer_cast(%c27808_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %69 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%103 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %104 = hivm.hir.pointer_cast(%c28320_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c28352_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%102 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%104 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %104 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %101[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c28864_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%106 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %expand_shape_48 = memref.expand_shape %106 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %107 = hivm.hir.pointer_cast(%c28896_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%107 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %108 = arith.index_cast %arg11 : i32 to index
        %109 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%108]
        %110 = arith.cmpi slt, %108, %c0 : index
        %111 = arith.cmpi sge, %108, %c16 : index
        %112 = arith.ori %110, %111 : i1
        scf.if %112 {
        } else {
          %subview_49 = memref.subview %69[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %107[0, 0] [%108, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %107[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %69[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %78 = arith.minsi %74, %c32_i32 : i32
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg11 = %c18_i32 to %78 step %c1_i32  : i32 {
        %96 = arith.muli %arg11, %c128_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 16)>()[%97, %32, %77]
        %reinterpret_cast_44 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %99 = hivm.hir.pointer_cast(%c29408_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.load ins(%reinterpret_cast_44 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%99 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %100 = hivm.hir.pointer_cast(%c29408_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%99, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<16xf32, #hivm.address_space<ub>>)
        %101 = hivm.hir.pointer_cast(%c29408_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%100, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%101 : memref<16xf32, #hivm.address_space<ub>>)
        %expand_shape_45 = memref.expand_shape %101 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c29472_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = hivm.hir.pointer_cast(%c29984_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%expand_shape_45, %73 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%102 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%103 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
        %104 = hivm.hir.pointer_cast(%c30496_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c30528_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%102 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%104 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        %collapse_shape_46 = memref.collapse_shape %104 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_47 = memref.subview %101[%12] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c31040_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%subview_47, %collapse_shape_46 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%106 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %107 = arith.subi %arg11, %c16_i32 : i32
        %expand_shape_48 = memref.expand_shape %106 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %108 = hivm.hir.pointer_cast(%c31072_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%expand_shape_48 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%108 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
        %109 = arith.index_cast %107 : i32 to index
        %110 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%109]
        %111 = arith.cmpi slt, %109, %c0 : index
        %112 = arith.cmpi sge, %109, %c16 : index
        %113 = arith.ori %111, %112 : i1
        scf.if %113 {
        } else {
          %subview_49 = memref.subview %73[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_50 = memref.subview %108[0, 0] [%109, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_51 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_52 = memref.collapse_shape %subview_50 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_51 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_53 = memref.subview %108[0, 0] [%110, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %73[0, 0] [%110, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_55 = memref.collapse_shape %subview_53 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%collapse_shape_55 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %79 = hivm.hir.pointer_cast(%c31584_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %79 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %80 = hivm.hir.pointer_cast(%c32096_i64) : memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%80 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
      %81 = hivm.hir.pointer_cast(%c26080_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %81 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_17, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %82 = memref_ext.alloc_workspace() from %arg2 offset = [%c12288] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_24 = memref.subview %82[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%81 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %83 = hivm.hir.pointer_cast(%c31584_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %83 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_21, %collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %84 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_26 = memref.subview %84[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%83 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_26 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 13
      %85 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_27 = memref.subview %85[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %86 = hivm.hir.pointer_cast(%c32192_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_29 = memref.collapse_shape %86 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%collapse_shape_28 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %87 = hivm.hir.pointer_cast(%c32192_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %87 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_29, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
      %88 = hivm.hir.pointer_cast(%c32192_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_31 = memref.collapse_shape %88 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_33 = memref.subview %reinterpret_cast_32[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_34 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_35 = memref.subview %reinterpret_cast_34[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_37 = memref.subview %reinterpret_cast_36[0, %12] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_38 = memref.subview %81[%56, 0] [%57, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_39 = memref.subview %subview_33[0, 0] [%57, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_38 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      %subview_40 = memref.subview %83[%63, 0] [%64, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_41 = memref.subview %subview_37[0, 0] [%64, 8] [1, 1] : memref<16x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_40 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x8xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %89 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%63, %11]
      %90 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%50, %11]
      %91 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%90, %63, %11]
      %92 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%91, %89]
      %93 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%89, %11]
      %subview_42 = memref.subview %88[%93, 0] [%92, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %94 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%11, %51, %63]
      %95 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%94, %51, %63]
      %subview_43 = memref.subview %subview_35[0, 0] [%95, 16] [1, 1] : memref<8x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_42 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_43 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    }
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 10
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
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
