// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fused_gdn_gating_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c50432_i64 = arith.constant 50432 : i64
  %c67872_i64 = arith.constant 67872 : i64
  %c42240_i64 = arith.constant 42240 : i64
  %c41984_i64 = arith.constant 41984 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c59680_i64 = arith.constant 59680 : i64
  %c25600_i64 = arith.constant 25600 : i64
  %c51488_i64 = arith.constant 51488 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c50976_i64 = arith.constant 50976 : i64
  %c16896_i64 = arith.constant 16896 : i64
  %c50464_i64 = arith.constant 50464 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16x128xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c8192_i64) : memref<16x128xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %6 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_3 : memref<2048xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg12 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c0_i64_4 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_4, %c1_i64 : i64
    %14 = arith.index_cast %11 : index to i1
    %c1_i64_5 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %15 = arith.select %14, %c1_i64_5, %c2_i64 : i64
    %16 = hivm.hir.pointer_cast(%c42240_i64, %c67872_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<16x128xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c25600_i64, %c59680_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<16x128xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c25600_i64, %c59680_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<16x128xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c17408_i64, %c51488_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<16x128xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c16896_i64, %c50976_i64) : memref<1x128xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<1x128xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c16384_i64, %c50464_i64) : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %22 = arith.divsi %arg12, %arg11 : i32
    %23 = arith.remsi %22, %arg10 : i32
    %24 = arith.divsi %arg12, %4 : i32
    %25 = arith.remsi %24, %arg9 : i32
    %26 = arith.muli %25, %c128_i32 : i32
    %27 = arith.muli %23, %c16_i32 : i32
    %28 = arith.index_cast %26 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%28], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %29 = affine.max affine_map<()[s0] -> (256, s0)>()[%28]
    %30 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%29, %28]
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %28]
    %32 = arith.cmpi slt, %31, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%31] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %21[0] [%31] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %32 {
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%21 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %15]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%28], sizes: [1, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [1, %31] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %20[0, 0] [1, %31] [1, 1] : memref<1x128xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %32 {
      %collapse_shape_38 = memref.collapse_shape %20 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_38 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_10 = memref.collapse_shape %subview_8 [[0, 1]] : memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%21 : memref<128xf32, #hivm.address_space<ub>>) outs(%21 : memref<128xf32, #hivm.address_space<ub>>)
    %33 = arith.index_cast %27 : i32 to index
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%28, %33]
    %reinterpret_cast_12 = memref.reinterpret_cast %arg6 to offset: [%34], sizes: [16, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %35 = affine.max affine_map<()[s0] -> (1, s0)>()[%33]
    %36 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%35, %33]
    %37 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%36, %33]
    %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
    %39 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%30, %28]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %41 = arith.cmpi slt, %38, %c16 : index
    %42 = arith.cmpi slt, %40, %c128 : index
    %43 = arith.ori %41, %42 : i1
    %subview_13 = memref.subview %reinterpret_cast_12[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_14 = memref.subview %19[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %43 {
      %collapse_shape_38 = memref.collapse_shape %19 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_38 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_13 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_15 = memref.reinterpret_cast %arg7 to offset: [%34], sizes: [16, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_16 = memref.subview %reinterpret_cast_15[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %18[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %43 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_38 = memref.collapse_shape %18 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_38 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%subview_16 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_17 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vadd ins(%19, %20 : memref<16x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>) outs(%19 : memref<16x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %44 = hivm.hir.pointer_cast(%c33792_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %19 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %44 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_18 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<2048xf32, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c33792_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %45 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_19, %cst_0 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_20 : memref<2048xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c33792_i64) : memref<16x128xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %46 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_20 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<2048xf32, #hivm.address_space<ub>>)
    %47 = hivm.hir.pointer_cast(%c41984_i64) : memref<16x128xi1, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %47 [[0, 1]] : memref<16x128xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%collapse_shape_18, %collapse_shape : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<2048xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %collapse_shape_23 = memref.collapse_shape %16 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %48 = hivm.hir.pointer_cast(%c50432_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_22, %collapse_shape_18, %collapse_shape_21 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<2048xf32, #hivm.address_space<ub>>) temp_buffer(%48 : memref<8xf32, #hivm.address_space<ub>>)
    %collapse_shape_24 = memref.collapse_shape %18 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %18 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_24, %cst_2 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_25 : memref<2048xf32, #hivm.address_space<ub>>)
    %collapse_shape_26 = memref.collapse_shape %18 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_25, %cst : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_26 : memref<2048xf32, #hivm.address_space<ub>>)
    %collapse_shape_27 = memref.collapse_shape %18 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_26 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<2048xf32, #hivm.address_space<ub>>)
    %collapse_shape_28 = memref.collapse_shape %18 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_27, %cst_0 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_28 : memref<2048xf32, #hivm.address_space<ub>>)
    %collapse_shape_29 = memref.collapse_shape %17 [[0, 1]] : memref<16x128xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_3, %collapse_shape_28 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<2048xf32, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c16384_i64, %c50464_i64) : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %49 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%21, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c16384_i64, %c50464_i64) : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %50 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%50, %16 : memref<1x128xf32, #hivm.address_space<ub>>, memref<16x128xf32, #hivm.address_space<ub>>) outs(%16 : memref<16x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %15]
    %reinterpret_cast_32 = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [16, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_33 = memref.subview %16[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_34 = memref.subview %reinterpret_cast_32[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_33 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_34 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    %reinterpret_cast_35 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [16, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_36 = memref.subview %17[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_37 = memref.subview %reinterpret_cast_35[0, 0] [%38, %40] [1, 1] : memref<16x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_36 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_37 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
