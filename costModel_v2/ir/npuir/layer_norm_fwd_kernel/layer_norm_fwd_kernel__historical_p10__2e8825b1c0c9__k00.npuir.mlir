// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c168928_i64 = arith.constant 168928 : i64
  %c160736_i64 = arith.constant 160736 : i64
  %c152544_i64 = arith.constant 152544 : i64
  %c152512_i64 = arith.constant 152512 : i64
  %c136128_i64 = arith.constant 136128 : i64
  %c136096_i64 = arith.constant 136096 : i64
  %c136032_i64 = arith.constant 136032 : i64
  %c103264_i64 = arith.constant 103264 : i64
  %c103008_i64 = arith.constant 103008 : i64
  %c86624_i64 = arith.constant 86624 : i64
  %c86592_i64 = arith.constant 86592 : i64
  %c53824_i64 = arith.constant 53824 : i64
  %c53568_i64 = arith.constant 53568 : i64
  %c37184_i64 = arith.constant 37184 : i64
  %c37088_i64 = arith.constant 37088 : i64
  %c37056_i64 = arith.constant 37056 : i64
  %c37024_i64 = arith.constant 37024 : i64
  %c20640_i64 = arith.constant 20640 : i64
  %c20544_i64 = arith.constant 20544 : i64
  %c16448_i64 = arith.constant 16448 : i64
  %c16416_i64 = arith.constant 16416 : i64
  %c8224_i64 = arith.constant 8224 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst_1 = arith.constant 4.8828125E-4 : f32
  %cst_2 = arith.constant 2.048000e+03 : f32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst_3 = arith.constant 0.000000e+00 : f16
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c8192_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : memref<4xi32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c8224_i64) : memref<2048xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%7 : memref<2048xi32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg9 : i32 to index
  %9 = arith.index_cast %arg10 : i32 to index
  %10 = hivm.hir.pointer_cast(%c8224_i64) : memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%7 : memref<2048xi32, #hivm.address_space<ub>>) outs(%10 : memref<2048xf32, #hivm.address_space<ub>>)
  %11 = arith.sitofp %arg12 : i32 to f32
  %12 = hivm.hir.pointer_cast(%c16416_i64) : memref<4x1xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %12 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%11 : f32) outs(%collapse_shape_4 : memref<4xf32, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c0_i64) : memref<1x2048xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %13 [[0, 1]] : memref<1x2048xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%10, %collapse_shape : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<2048xi1, #hivm.address_space<ub>>) compare_mode = <lt>
  %14 = hivm.hir.pointer_cast(%c16448_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %14 [[0, 1]] : memref<1x2048xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c20544_i64) : memref<48xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_5 : memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<2048xf16, #hivm.address_space<ub>>) temp_buffer(%15 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
  %16 = hivm.hir.pointer_cast(%c20640_i64) : memref<4x2048xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%14 : memref<1x2048xf16, #hivm.address_space<ub>>) outs(%16 : memref<4x2048xf16, #hivm.address_space<ub>>) broadcast_dims = [0]
  %17 = hivm.hir.pointer_cast(%c20640_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %16 [[0, 1]] : memref<4x2048xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %17 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_7, %cst_3 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_8 : memref<8192xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c20640_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
  %collapse_shape_9 = memref.collapse_shape %18 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_8 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<8192xi1, #hivm.address_space<ub>>)
  %19 = arith.index_cast %arg12 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg17, %arg16 : i32
    %21 = arith.remsi %20, %arg15 : i32
    %22 = arith.divsi %arg17, %4 : i32
    %23 = arith.remsi %22, %arg14 : i32
    %24 = arith.muli %23, %c4_i32 : i32
    %25 = hivm.hir.pointer_cast(%c37024_i64) : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%6, %24 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%25 : memref<4xi32, #hivm.address_space<ub>>)
    %26 = arith.muli %21, %c2048_i32 : i32
    %27 = arith.index_cast %24 : i32 to index
    %28 = arith.index_cast %26 : i32 to index
    %29 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%28, %27, %8]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [4, 2048], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %30 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%28, %27, %9]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%30], sizes: [4, 2048], strides: [%9, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %31 = hivm.hir.pointer_cast(%c37024_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %31 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%25 : memref<4xi32, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<4xf32, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c37024_i64) : memref<4x1xi1, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %32 [[0, 1]] : memref<4x1xi1, #hivm.address_space<ub>> into memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_11, %collapse_shape_4 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<4xi1, #hivm.address_space<ub>>) compare_mode = <lt>
    %33 = hivm.hir.pointer_cast(%c37056_i64) : memref<4x1xf16, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %33 [[0, 1]] : memref<4x1xf16, #hivm.address_space<ub>> into memref<4xf16, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c37088_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_12 : memref<4xi1, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<4xf16, #hivm.address_space<ub>>) temp_buffer(%34 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %35 = hivm.hir.pointer_cast(%c37184_i64) : memref<4x2048xf16, #hivm.address_space<ub>>
    %36 = hivm.hir.pointer_cast(%c53568_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%33 : memref<4x1xf16, #hivm.address_space<ub>>) outs(%35 : memref<4x2048xf16, #hivm.address_space<ub>>) temp_buffer(%36 : memref<128xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
    %37 = hivm.hir.pointer_cast(%c37184_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %35 [[0, 1]] : memref<4x2048xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %37 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_14, %cst_3 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c37184_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %38 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8192xi1, #hivm.address_space<ub>>)
    %39 = hivm.hir.pointer_cast(%c37184_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %39 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vand ins(%collapse_shape_16, %collapse_shape_9 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<8192xi1, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c53824_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %41 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%27, %19]
    %42 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%41, %27]
    %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%42, %27]
    %44 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%42, %27]
    %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
    %46 = arith.cmpi slt, %45, %c4 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%45, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %40[0, 0] [%45, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %46 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_39 = memref.collapse_shape %40 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_39 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_18 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %47 = hivm.hir.pointer_cast(%c86592_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %48 = hivm.hir.pointer_cast(%c86624_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <sum> ins(%40 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%47 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%48 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_19 = memref.collapse_shape %47 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c86592_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_19, %cst_1 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%49 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %50 = arith.muli %21, %arg12 : i32
    %51 = arith.index_cast %50 : i32 to index
    %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%51, %27]
    %reinterpret_cast_20 = memref.reinterpret_cast %arg7 to offset: [%52], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_21 = memref.subview %49[0] [%43] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_22 = memref.subview %reinterpret_cast_20[0] [%43] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_22 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %expand_shape = memref.expand_shape %49 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c53824_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c103008_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%40, %expand_shape : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%53 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %55 = hivm.hir.pointer_cast(%c103264_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %53 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %55 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%c136032_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_17, %collapse_shape_23, %cst_0 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%56 : memref<16xf32, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c103264_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %57 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_24, %collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<8192xf32, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c136096_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c136128_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%57 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%58 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_26 = memref.collapse_shape %58 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %60 = hivm.hir.pointer_cast(%c136096_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_26, %cst_1 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%60 : memref<4xf32, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c136096_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%60, %arg13 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%61 : memref<4xf32, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c136096_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%61 : memref<4xf32, #hivm.address_space<ub>>) outs(%62 : memref<4xf32, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c152512_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%63 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%63, %62 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%63 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_27 = memref.reinterpret_cast %arg8 to offset: [%52], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_28 = memref.subview %63[0] [%43] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_29 = memref.subview %reinterpret_cast_27[0] [%43] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_28 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_29 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_30 = memref.reinterpret_cast %arg5 to offset: [%28], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %64 = hivm.hir.pointer_cast(%c152544_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %reinterpret_cast_30 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_32 = memref.collapse_shape %64 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_31 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_32 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_33 = memref.reinterpret_cast %arg6 to offset: [%28], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %65 = hivm.hir.pointer_cast(%c160736_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %reinterpret_cast_33 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_35 = memref.collapse_shape %65 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_34 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_35 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape_36 = memref.expand_shape %63 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %66 = hivm.hir.pointer_cast(%c53824_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c168928_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%53, %expand_shape_36 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%66 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %68 = hivm.hir.pointer_cast(%c53824_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%66, %64 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%68 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    %69 = hivm.hir.pointer_cast(%c53824_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%68, %65 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%69 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_37 = memref.subview %69[0, 0] [%45, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_38 = memref.subview %reinterpret_cast_10[0, 0] [%45, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_37 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_38 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
