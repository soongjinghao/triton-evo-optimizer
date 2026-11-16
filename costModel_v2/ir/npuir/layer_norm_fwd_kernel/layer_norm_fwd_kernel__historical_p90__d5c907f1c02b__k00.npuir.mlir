// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c90240_i64 = arith.constant 90240 : i64
  %c82048_i64 = arith.constant 82048 : i64
  %c73856_i64 = arith.constant 73856 : i64
  %c65664_i64 = arith.constant 65664 : i64
  %c65632_i64 = arith.constant 65632 : i64
  %c65600_i64 = arith.constant 65600 : i64
  %c164000_i64 = arith.constant 164000 : i64
  %c163968_i64 = arith.constant 163968 : i64
  %c131200_i64 = arith.constant 131200 : i64
  %c65568_i64 = arith.constant 65568 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i64 = arith.constant 0 : i64
  %c98432_i64 = arith.constant 98432 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c4_i32 = arith.constant 4 : i32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 4.8828125E-4 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c98432_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg9 : i32 to index
  %7 = arith.index_cast %arg10 : i32 to index
  %8 = arith.index_cast %arg12 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    %9 = arith.index_cast %arg17 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %1 : i32 to index
    %12 = arith.index_cast %c48_i32 : i32 to index
    %13 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%9, %10, %12]
    %14 = arith.index_cast %13 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_2, %c1_i64 : i64
    %16 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c82048_i64, %c90240_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c65664_i64, %c73856_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c65600_i64, %c65632_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c65536_i64, %c65568_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %22 = arith.divsi %arg17, %arg16 : i32
    %23 = arith.remsi %22, %arg15 : i32
    %24 = arith.divsi %arg17, %4 : i32
    %25 = arith.remsi %24, %arg14 : i32
    %26 = arith.muli %25, %c4_i32 : i32
    %27 = arith.muli %23, %c2048_i32 : i32
    %28 = arith.index_cast %26 : i32 to index
    %29 = arith.index_cast %27 : i32 to index
    %30 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%29, %28, %6]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [4, 2048], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %31 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%29, %28, %7]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [4, 2048], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %32 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%28, %8]
    %33 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%32, %28]
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%33, %28]
    %35 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%33, %28]
    %36 = affine.max affine_map<()[s0] -> (0, s0)>()[%35]
    %37 = arith.cmpi slt, %36, %c4 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%36, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %21[0, 0] [%36, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %37 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_28 = memref.collapse_shape %21 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_28 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %38 = hivm.hir.pointer_cast(%c65536_i64, %c65568_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c131200_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <sum> ins(%21 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%38 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%39 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_5 = memref.collapse_shape %38 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_5, %cst_1 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%20 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %40 = arith.muli %23, %arg12 : i32
    %41 = arith.index_cast %40 : i32 to index
    %42 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%41, %28]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%42], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %20[0] [%34] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %reinterpret_cast_6[0] [%34] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %expand_shape = memref.expand_shape %20 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c131200_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%21, %expand_shape : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%21 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%43 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %subview_9 = memref.subview %21[0, 0] [%36, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c131200_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_10 = memref.collapse_shape %44 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_10 : memref<8192xf32, #hivm.address_space<ub>>)
    %subview_11 = memref.subview %44[0, 0] [%36, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %subview_9 [[0, 1]] : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %subview_11 [[0, 1]] : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c131200_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %45 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_10, %collapse_shape_10 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<8192xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c163968_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c164000_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%45 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%46 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%47 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_15 = memref.collapse_shape %46 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %48 = hivm.hir.pointer_cast(%c163968_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_15, %cst_1 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%48 : memref<4xf32, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c163968_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%48, %arg13 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%49 : memref<4xf32, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c163968_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%49 : memref<4xf32, #hivm.address_space<ub>>) outs(%50 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%cst : f32) outs(%19 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%19, %50 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%19 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg8 to offset: [%42], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %19[0] [%34] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %reinterpret_cast_16[0] [%34] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_18 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_19 = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_20 = memref.collapse_shape %reinterpret_cast_19 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_21 = memref.collapse_shape %18 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_20 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_21 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_22 = memref.reinterpret_cast %arg6 to offset: [%29], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_23 = memref.collapse_shape %reinterpret_cast_22 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_24 = memref.collapse_shape %17 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_23 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_24 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape_25 = memref.expand_shape %19 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %51 = hivm.hir.pointer_cast(%c131200_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%21, %expand_shape_25 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%21 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%21, %18 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%21 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%21, %17 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%16 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_26 = memref.subview %16[0, 0] [%36, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_27 = memref.subview %reinterpret_cast_3[0, 0] [%36, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_26 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_27 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
