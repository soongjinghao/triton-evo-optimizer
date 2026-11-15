// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c90240_i64 = arith.constant 90240 : i64
  %c82048_i64 = arith.constant 82048 : i64
  %c73856_i64 = arith.constant 73856 : i64
  %c65664_i64 = arith.constant 65664 : i64
  %c65632_i64 = arith.constant 65632 : i64
  %c65600_i64 = arith.constant 65600 : i64
  %c147648_i64 = arith.constant 147648 : i64
  %c65568_i64 = arith.constant 65568 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i64 = arith.constant 0 : i64
  %c131264_i64 = arith.constant 131264 : i64
  %c98496_i64 = arith.constant 98496 : i64
  %c98464_i64 = arith.constant 98464 : i64
  %c114880_i64 = arith.constant 114880 : i64
  %c110752_i64 = arith.constant 110752 : i64
  %c114848_i64 = arith.constant 114848 : i64
  %c106656_i64 = arith.constant 106656 : i64
  %c106624_i64 = arith.constant 106624 : i64
  %c98432_i64 = arith.constant 98432 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst_1 = arith.constant 2.048000e+03 : f32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst_2 = arith.constant 0.000000e+00 : f16
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c98432_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c106624_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : memref<4xi32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c106656_i64) : memref<2048xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%7 : memref<2048xi32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg9 : i32 to index
  %9 = arith.index_cast %arg10 : i32 to index
  %10 = hivm.hir.pointer_cast(%c106656_i64) : memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%7 : memref<2048xi32, #hivm.address_space<ub>>) outs(%10 : memref<2048xf32, #hivm.address_space<ub>>)
  %11 = arith.sitofp %arg12 : i32 to f32
  %12 = hivm.hir.pointer_cast(%c114848_i64) : memref<4x1xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %12 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%11 : f32) outs(%collapse_shape_3 : memref<4xf32, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c98432_i64) : memref<1x2048xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %13 [[0, 1]] : memref<1x2048xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%10, %collapse_shape : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<2048xi1, #hivm.address_space<ub>>) compare_mode = <lt>
  %14 = hivm.hir.pointer_cast(%c106656_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %14 [[0, 1]] : memref<1x2048xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c110752_i64) : memref<48xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_4 : memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<2048xf16, #hivm.address_space<ub>>) temp_buffer(%15 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
  %16 = hivm.hir.pointer_cast(%c114880_i64) : memref<4x2048xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%14 : memref<1x2048xf16, #hivm.address_space<ub>>) outs(%16 : memref<4x2048xf16, #hivm.address_space<ub>>) broadcast_dims = [0]
  %17 = hivm.hir.pointer_cast(%c114880_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %16 [[0, 1]] : memref<4x2048xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %17 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_6, %cst_2 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_7 : memref<8192xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c114880_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %18 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_7 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<8192xi1, #hivm.address_space<ub>>)
  %19 = arith.index_cast %arg12 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    %20 = arith.index_cast %arg17 : i32 to index
    %21 = arith.index_cast %3 : i32 to index
    %22 = arith.index_cast %1 : i32 to index
    %23 = arith.index_cast %c48_i32 : i32 to index
    %24 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%20, %21, %23]
    %25 = arith.index_cast %24 : index to i1
    %c0_i64_9 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %26 = arith.select %25, %c0_i64_9, %c1_i64 : i64
    %27 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c82048_i64, %c90240_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c65664_i64, %c73856_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c65600_i64, %c65632_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c65536_i64, %c65568_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<4x2048xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %33 = arith.divsi %arg17, %arg16 : i32
    %34 = arith.remsi %33, %arg15 : i32
    %35 = arith.divsi %arg17, %4 : i32
    %36 = arith.remsi %35, %arg14 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = hivm.hir.pointer_cast(%c98432_i64) : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%6, %37 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%38 : memref<4xi32, #hivm.address_space<ub>>)
    %39 = arith.muli %34, %c2048_i32 : i32
    %40 = arith.index_cast %37 : i32 to index
    %41 = arith.index_cast %39 : i32 to index
    %42 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%41, %40, %8]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [4, 2048], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %43 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%41, %40, %9]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [4, 2048], strides: [%9, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %44 = hivm.hir.pointer_cast(%c98432_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %44 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%38 : memref<4xi32, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<4xf32, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c98432_i64) : memref<4x1xi1, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %45 [[0, 1]] : memref<4x1xi1, #hivm.address_space<ub>> into memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_11, %collapse_shape_3 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<4xi1, #hivm.address_space<ub>>) compare_mode = <lt>
    %46 = hivm.hir.pointer_cast(%c98464_i64) : memref<4x1xf16, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %46 [[0, 1]] : memref<4x1xf16, #hivm.address_space<ub>> into memref<4xf16, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c98496_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_12 : memref<4xi1, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<4xf16, #hivm.address_space<ub>>) temp_buffer(%47 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %48 = hivm.hir.pointer_cast(%c131264_i64) : memref<4x2048xf16, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c98496_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%46 : memref<4x1xf16, #hivm.address_space<ub>>) outs(%48 : memref<4x2048xf16, #hivm.address_space<ub>>) temp_buffer(%49 : memref<128xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
    %50 = hivm.hir.pointer_cast(%c131264_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %48 [[0, 1]] : memref<4x2048xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %50 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_14, %cst_2 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>)
    %51 = hivm.hir.pointer_cast(%c131264_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %51 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8192xi1, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c131264_i64) : memref<4x2048xi1, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %52 [[0, 1]] : memref<4x2048xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vand ins(%collapse_shape_16, %collapse_shape_8 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<8192xi1, #hivm.address_space<ub>>)
    %53 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%40, %19]
    %54 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%53, %40]
    %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%54, %40]
    %56 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%54, %40]
    %57 = affine.max affine_map<()[s0] -> (0, s0)>()[%56]
    %58 = arith.cmpi slt, %57, %c4 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%57, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %32[0, 0] [%57, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %58 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_39 = memref.collapse_shape %32 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_39 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_18 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %59 = hivm.hir.pointer_cast(%c65536_i64, %c65568_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %60 = hivm.hir.pointer_cast(%c147648_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <sum> ins(%32 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%59 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_19 = memref.collapse_shape %59 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c98432_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_19, %cst_1 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %62 = arith.muli %34, %arg12 : i32
    %63 = arith.index_cast %62 : i32 to index
    %64 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%63, %40]
    %reinterpret_cast_20 = memref.reinterpret_cast %arg7 to offset: [%64], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_21 = memref.subview %31[0] [%55] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_22 = memref.subview %reinterpret_cast_20[0] [%55] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_22 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %expand_shape = memref.expand_shape %31 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %65 = hivm.hir.pointer_cast(%c98432_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%32, %expand_shape : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%32 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%65 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %66 = hivm.hir.pointer_cast(%c147648_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %32 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %66 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c98432_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_17, %collapse_shape_23, %cst_0 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
    %68 = hivm.hir.pointer_cast(%c147648_i64) : memref<4x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %68 [[0, 1]] : memref<4x2048xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_24, %collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<8192xf32, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c98432_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %70 = hivm.hir.pointer_cast(%c131264_i64) : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%68 : memref<4x2048xf32, #hivm.address_space<ub>>) outs(%69 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_26 = memref.collapse_shape %69 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %71 = hivm.hir.pointer_cast(%c98432_i64) : memref<4xf32, #hivm.address_space<ub>>
    %72 = hivm.hir.pointer_cast(%c98464_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_26, %cst_1 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%71 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%72 : memref<8xf32, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c98432_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%71, %arg13 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%73 : memref<4xf32, #hivm.address_space<ub>>)
    %74 = hivm.hir.pointer_cast(%c98432_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%73 : memref<4xf32, #hivm.address_space<ub>>) outs(%74 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%cst : f32) outs(%30 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%30, %74 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%30 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_27 = memref.reinterpret_cast %arg8 to offset: [%64], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_28 = memref.subview %30[0] [%55] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_29 = memref.subview %reinterpret_cast_27[0] [%55] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_28 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_29 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_30 = memref.reinterpret_cast %arg5 to offset: [%41], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_31 = memref.collapse_shape %reinterpret_cast_30 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_32 = memref.collapse_shape %29 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_31 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_32 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_33 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_34 = memref.collapse_shape %reinterpret_cast_33 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_35 = memref.collapse_shape %28 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_34 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_35 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape_36 = memref.expand_shape %30 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %75 = hivm.hir.pointer_cast(%c98432_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%32, %expand_shape_36 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%32 : memref<4x2048xf32, #hivm.address_space<ub>>) temp_buffer(%75 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%32, %29 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%32 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%32, %28 : memref<4x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%27 : memref<4x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_37 = memref.subview %27[0, 0] [%57, 2048] [1, 1] : memref<4x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_38 = memref.subview %reinterpret_cast_10[0, 0] [%57, 2048] [1, 1] : memref<4x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_37 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_38 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
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
