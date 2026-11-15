// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_cumsum_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: f32, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c10816_i64 = arith.constant 10816 : i64
  %c9152_i64 = arith.constant 9152 : i64
  %c8640_i64 = arith.constant 8640 : i64
  %c10752_i64 = arith.constant 10752 : i64
  %c8576_i64 = arith.constant 8576 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c8064_i64 = arith.constant 8064 : i64
  %c8032_i64 = arith.constant 8032 : i64
  %c7520_i64 = arith.constant 7520 : i64
  %c7456_i64 = arith.constant 7456 : i64
  %c6944_i64 = arith.constant 6944 : i64
  %c6432_i64 = arith.constant 6432 : i64
  %c6400_i64 = arith.constant 6400 : i64
  %c6368_i64 = arith.constant 6368 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c5792_i64 = arith.constant 5792 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c4768_i64 = arith.constant 4768 : i64
  %c4736_i64 = arith.constant 4736 : i64
  %c4704_i64 = arith.constant 4704 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c3680_i64 = arith.constant 3680 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c10176_i64 = arith.constant 10176 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c9664_i64 = arith.constant 9664 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i64 = arith.constant 16 : i64
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst_0 = arith.constant 2.000000e+01 : f32
  %c16_i32 = arith.constant 16 : i32
  %c15_i32 = arith.constant 15 : i32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg22, %arg23 : i32
  %1 = arith.muli %0, %arg24 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg24, %arg23 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %6 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_2 : memref<128xf32, #hivm.address_space<ub>>)
  %7 = arith.addi %arg9, %arg11 : i32
  %8 = arith.subi %7, %c1_i32 : i32
  %9 = arith.addi %arg10, %c15_i32 : i32
  %10 = arith.divsi %9, %c16_i32 : i32
  %11 = arith.extsi %10 : i32 to i64
  %12 = arith.extsi %arg14 : i32 to i64
  %13 = arith.extsi %arg11 : i32 to i64
  %14 = arith.extsi %arg15 : i32 to i64
  %15 = arith.extsi %arg16 : i32 to i64
  %16 = arith.extsi %arg17 : i32 to i64
  %17 = arith.extsi %arg19 : i32 to i64
  %18 = arith.extsi %arg20 : i32 to i64
  %19 = arith.index_cast %arg18 : i32 to index
  %20 = arith.index_cast %arg21 : i32 to index
  %21 = arith.extsi %arg9 : i32 to i64
  %22 = arith.index_cast %arg15 : i32 to index
  %23 = arith.index_cast %arg10 : i32 to index
  %24 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %24 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg12 : f32) outs(%collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>)
  %25 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %25 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %26 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %26 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<128xi1, #hivm.address_space<ub>>)
  %27 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %27 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg13 : f32) outs(%collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>)
  %28 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %28 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<128xi1, #hivm.address_space<ub>>)
  %29 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %29 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_7 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<128xi1, #hivm.address_space<ub>>)
  %30 = arith.index_cast %arg11 : i32 to index
  %31 = affine.max affine_map<()[s0] -> (0, s0)>()[%30]
  %32 = affine.min affine_map<()[s0] -> (8, s0)>()[%31]
  %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%32]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg25 = %3 to %1 step %c48_i32  : i32 {
    %34 = arith.index_cast %arg25 : i32 to index
    %35 = arith.index_cast %3 : i32 to index
    %36 = arith.index_cast %1 : i32 to index
    %37 = arith.index_cast %c48_i32 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%34, %35, %37]
    %39 = arith.index_cast %38 : index to i1
    %c0_i64_9 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %40 = arith.select %39, %c0_i64_9, %c1_i64 : i64
    %41 = hivm.hir.pointer_cast(%c9152_i64, %c10816_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c8576_i64, %c10752_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c8064_i64, %c10240_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %43 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c3072_i64, %c10176_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    annotation.mark %44 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c2048_i64, %c9664_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<8x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %46 = arith.divsi %arg25, %4 : i32
    %47 = arith.remsi %46, %arg22 : i32
    %48 = arith.extsi %47 : i32 to i64
    %49 = arith.divsi %8, %arg11 : i32
    %50 = arith.extsi %49 : i32 to i64
    %51 = arith.muli %50, %11 : i64
    %52 = arith.divsi %48, %51 : i64
    %53 = arith.remsi %48, %51 : i64
    %54 = arith.divsi %53, %11 : i64
    %55 = arith.remsi %53, %11 : i64
    %56 = arith.muli %52, %12 : i64
    %57 = arith.muli %54, %13 : i64
    %58 = arith.muli %57, %14 : i64
    %59 = arith.addi %56, %58 : i64
    %60 = arith.index_cast %59 : i64 to index
    %61 = arith.muli %52, %15 : i64
    %62 = arith.muli %54, %16 : i64
    %63 = arith.addi %61, %62 : i64
    %64 = arith.muli %52, %17 : i64
    %65 = arith.muli %54, %18 : i64
    %66 = arith.addi %64, %65 : i64
    %67 = arith.muli %55, %c16_i64 : i64
    %68 = arith.index_cast %67 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %69 = arith.index_cast %63 : i64 to index
    %70 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%69, %68, %19]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%70], sizes: [16, 8], strides: [%19, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %71 = arith.index_cast %66 : i64 to index
    %72 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%71, %68, %20]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%72], sizes: [16, 8], strides: [%20, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %73 = arith.subi %21, %57 : i64
    %74 = arith.minsi %13, %73 : i64
    %75 = arith.index_cast %74 : i64 to index
    %76 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%60, %68]
    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%76], sizes: [8, 16], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %77 = affine.max affine_map<()[s0] -> (0, s0)>()[%75]
    %78 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%68, %23]
    %79 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%78, %68]
    %80 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%79, %68]
    %81 = affine.min affine_map<()[s0] -> (8, s0)>()[%77]
    %82 = affine.max affine_map<()[s0] -> (0, s0)>()[%81]
    %83 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%79, %68]
    %84 = affine.max affine_map<()[s0] -> (0, s0)>()[%83]
    %85 = arith.cmpi slt, %82, %c8 : index
    %86 = arith.cmpi slt, %84, %c16 : index
    %87 = arith.ori %85, %86 : i1
    %subview = memref.subview %reinterpret_cast_12[0, 0] [%82, %84] [1, 1] : memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_13 = memref.subview %45[0, 0] [%82, %84] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %87 {
      %collapse_shape_46 = memref.collapse_shape %45 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_46 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %88 = hivm.hir.pointer_cast(%c2560_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%45 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%88 : memref<16x8xf32, #hivm.address_space<ub>>) permutation = [1, 0]
    %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%68], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %89 = arith.cmpi slt, %80, %c16 : index
    %subview_15 = memref.subview %reinterpret_cast_14[0, 0] [%80, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_16 = memref.subview %44[0, 0] [%80, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %89 {
      %collapse_shape_46 = memref.collapse_shape %44 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_46 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_17 = memref.collapse_shape %subview_15 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_18 = memref.collapse_shape %subview_16 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %90 = hivm.hir.pointer_cast(%c2560_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %91 = hivm.hir.pointer_cast(%c3136_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%88, %44 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%90 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%91 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %92 = hivm.hir.pointer_cast(%c3648_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %90 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %92 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_19, %collapse_shape_2 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<128xi1, #hivm.address_space<ub>>) compare_mode = <le>
    %93 = hivm.hir.pointer_cast(%c3680_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %93 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
    %94 = hivm.hir.pointer_cast(%c3680_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %94 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_21, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>)
    %95 = hivm.hir.pointer_cast(%c3680_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %95 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
    %96 = hivm.hir.pointer_cast(%c4192_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %96 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %97 = hivm.hir.pointer_cast(%c4704_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_20, %collapse_shape_23, %collapse_shape_19 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%97 : memref<8xf32, #hivm.address_space<ub>>)
    %98 = hivm.hir.pointer_cast(%c4736_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %98 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_24, %collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xi1, #hivm.address_space<ub>>)
    %99 = hivm.hir.pointer_cast(%c4736_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %99 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_25 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<128xi1, #hivm.address_space<ub>>)
    %100 = hivm.hir.pointer_cast(%c4768_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %100 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_24, %arg12 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_27 : memref<128xf32, #hivm.address_space<ub>>)
    %101 = hivm.hir.pointer_cast(%c5280_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %101 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %102 = hivm.hir.pointer_cast(%c5792_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_26, %arg12, %collapse_shape_27 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%102 : memref<16xf32, #hivm.address_space<ub>>)
    %103 = hivm.hir.pointer_cast(%c5856_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %103 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %104 = hivm.hir.pointer_cast(%c6368_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_5, %collapse_shape_24, %collapse_shape_28 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%104 : memref<8xf32, #hivm.address_space<ub>>)
    %105 = hivm.hir.pointer_cast(%c6400_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %105 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_29, %collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<128xi1, #hivm.address_space<ub>>)
    %106 = hivm.hir.pointer_cast(%c6400_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %106 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_30 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<128xi1, #hivm.address_space<ub>>)
    %107 = hivm.hir.pointer_cast(%c6432_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %107 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_29, %arg13 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_32 : memref<128xf32, #hivm.address_space<ub>>)
    %108 = hivm.hir.pointer_cast(%c6944_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_33 = memref.collapse_shape %108 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %109 = hivm.hir.pointer_cast(%c7456_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_31, %arg13, %collapse_shape_32 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%109 : memref<16xf32, #hivm.address_space<ub>>)
    %110 = hivm.hir.pointer_cast(%c7520_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %110 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %111 = hivm.hir.pointer_cast(%c8032_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_8, %collapse_shape_29, %collapse_shape_33 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%111 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_35 = memref.subview %110[0, 0] [%84, %82] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %43 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, %40]
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_36 : memref<128xf32, #hivm.address_space<ub>>)
    %subview_37 = memref.subview %43[0, 0] [%84, %82] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_35 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_37 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_38 = memref.subview %43[0, 0] [%84, %33] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_39 = memref.subview %reinterpret_cast_10[0, 0] [%84, %33] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_38 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_40 = memref.subview %reinterpret_cast[0, 0] [%80, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_41 = memref.subview %42[0, 0] [%80, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %89 {
      %collapse_shape_46 = memref.collapse_shape %42 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_46 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_42 = memref.collapse_shape %subview_40 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_43 = memref.collapse_shape %subview_41 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_42 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_43 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %112 = hivm.hir.pointer_cast(%c8640_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%43, %42 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%43 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%112 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg26 = %c0 to %c16 step %c1 {
      %113 = memref.load %43[%arg26, %c0] : memref<16x8xf32, #hivm.address_space<ub>>
      memref.store %113, %41[%arg26, %c0] : memref<16x8xf32, #hivm.address_space<ub>>
      scf.for %arg27 = %c1 to %c8 step %c1 {
        %114 = memref.load %43[%arg26, %arg27] : memref<16x8xf32, #hivm.address_space<ub>>
        %115 = arith.subi %arg27, %c1 : index
        %116 = memref.load %41[%arg26, %115] : memref<16x8xf32, #hivm.address_space<ub>>
        %117 = arith.addf %114, %116 : f32
        memref.store %117, %41[%arg26, %arg27] : memref<16x8xf32, #hivm.address_space<ub>>
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, %40]
    %subview_44 = memref.subview %41[0, 0] [%84, %33] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_45 = memref.subview %reinterpret_cast_11[0, 0] [%84, %33] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_44 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_45 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
