// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c155776_i64 = arith.constant 155776 : i64
  %c147584_i64 = arith.constant 147584 : i64
  %c139392_i64 = arith.constant 139392 : i64
  %c131200_i64 = arith.constant 131200 : i64
  %c98432_i64 = arith.constant 98432 : i64
  %c65664_i64 = arith.constant 65664 : i64
  %c65632_i64 = arith.constant 65632 : i64
  %c65600_i64 = arith.constant 65600 : i64
  %c164000_i64 = arith.constant 164000 : i64
  %c163968_i64 = arith.constant 163968 : i64
  %c65568_i64 = arith.constant 65568 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 4.8828125E-4 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = arith.index_cast %arg9 : i32 to index
  %6 = arith.index_cast %arg12 : i32 to index
  %7 = arith.index_cast %arg10 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg17 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_2, %c1_i64 : i64
    %15 = arith.index_cast %12 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %16 = arith.select %15, %c3_i64, %c4_i64 : i64
    %17 = arith.index_cast %12 : index to i1
    %c1_i64_3 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %18 = arith.select %17, %c1_i64_3, %c2_i64 : i64
    %19 = hivm.hir.pointer_cast(%c65664_i64, %c98432_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c147584_i64, %c155776_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c131200_i64, %c139392_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c65664_i64, %c98432_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c65600_i64, %c65632_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c65536_i64, %c65568_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %26 = arith.divsi %arg17, %arg16 : i32
    %27 = arith.remsi %26, %arg15 : i32
    %28 = arith.divsi %arg17, %4 : i32
    %29 = arith.remsi %28, %arg14 : i32
    %30 = arith.muli %29, %c4_i32 : i32
    %31 = arith.muli %27, %c2048_i32 : i32
    %32 = arith.index_cast %30 : i32 to index
    %33 = arith.index_cast %31 : i32 to index
    %34 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%33, %32, %5]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [4, 2048], strides: [%5, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %35 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%32, %6]
    %36 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%35, %32]
    %37 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%36, %32]
    %38 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%36, %32]
    %39 = affine.max affine_map<()[s0] -> (0, s0)>()[%38]
    %40 = arith.cmpi slt, %39, %c4 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%39, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %25[0, 0] [%39, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %40 {
      %collapse_shape_25 = memref.collapse_shape %25 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_25 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %18]
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %41 = hivm.hir.pointer_cast(%c65536_i64, %c65568_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c163968_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %18]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <sum> ins(%25 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%41 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape = memref.collapse_shape %41 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %collapse_shape_5 = memref.collapse_shape %25 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %25 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_5, %collapse_shape_5 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<8192xf32, #hivm.address_space<ub>>)
    %43 = hivm.hir.pointer_cast(%c163968_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c164000_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%25 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%43 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%44 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %18]
    %collapse_shape_7 = memref.collapse_shape %43 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape, %cst_0 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%24 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %45 = hivm.hir.pointer_cast(%c163968_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_7, %cst_0 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%45 : memref<4xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c164000_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%24, %24 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%46 : memref<4xf32, #hivm.address_space<ub>>)
    %47 = hivm.hir.pointer_cast(%c164000_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%45, %46 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%47 : memref<4xf32, #hivm.address_space<ub>>)
    %48 = arith.muli %27, %arg12 : i32
    %49 = arith.index_cast %48 : i32 to index
    %50 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%49, %32]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%50], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %24[0] [%37] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_10 = memref.subview %reinterpret_cast_8[0] [%37] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %18]
    %51 = hivm.hir.pointer_cast(%c164000_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%47, %arg13 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%51 : memref<4xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c164000_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%51 : memref<4xf32, #hivm.address_space<ub>>) outs(%52 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %16]
    hivm.hir.vbrc ins(%cst : f32) outs(%23 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%23, %52 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%23 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg8 to offset: [%50], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %23[0] [%37] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_13 = memref.subview %reinterpret_cast_11[0] [%37] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_13 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %16]
    %53 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%33, %32, %7]
    %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [4, 2048], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_15 = memref.subview %22[0, 0] [%39, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %40 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_25 = memref.collapse_shape %22 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_25 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_15 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape = memref.expand_shape %24 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c163968_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%22, %expand_shape : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%22 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %expand_shape_16 = memref.expand_shape %23 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %55 = hivm.hir.pointer_cast(%c163968_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%22, %expand_shape_16 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%22 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %reinterpret_cast_17 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_18 = memref.collapse_shape %reinterpret_cast_17 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_19 = memref.collapse_shape %21 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_18 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_19 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_20 = memref.reinterpret_cast %arg6 to offset: [%33], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_21 = memref.collapse_shape %reinterpret_cast_20 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_22 = memref.collapse_shape %20 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_21 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_22 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%22, %21 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%22 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%22, %20 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%19 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_23 = memref.subview %19[0, 0] [%39, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_24 = memref.subview %reinterpret_cast_14[0, 0] [%39, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_23 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_24 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
