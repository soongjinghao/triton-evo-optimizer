// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_fwd_kernel_ep_gather(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, false, true, false, true, false, true, false, true, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c34080_i64 = arith.constant 34080 : i64
  %c34016_i64 = arith.constant 34016 : i64
  %c25824_i64 = arith.constant 25824 : i64
  %c42784_i64 = arith.constant 42784 : i64
  %c17632_i64 = arith.constant 17632 : i64
  %c17376_i64 = arith.constant 17376 : i64
  %c13280_i64 = arith.constant 13280 : i64
  %c13184_i64 = arith.constant 13184 : i64
  %c13152_i64 = arith.constant 13152 : i64
  %c13088_i64 = arith.constant 13088 : i64
  %c13056_i64 = arith.constant 13056 : i64
  %c13120_i64 = arith.constant 13120 : i64
  %c42656_i64 = arith.constant 42656 : i64
  %c12928_i64 = arith.constant 12928 : i64
  %c42528_i64 = arith.constant 42528 : i64
  %c12800_i64 = arith.constant 12800 : i64
  %c50976_i64 = arith.constant 50976 : i64
  %c34336_i64 = arith.constant 34336 : i64
  %c12544_i64 = arith.constant 12544 : i64
  %c8448_i64 = arith.constant 8448 : i64
  %c8352_i64 = arith.constant 8352 : i64
  %c8320_i64 = arith.constant 8320 : i64
  %c8288_i64 = arith.constant 8288 : i64
  %c8256_i64 = arith.constant 8256 : i64
  %c8224_i64 = arith.constant 8224 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c4_i32 = arith.constant 4 : i32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c0_i64 = arith.constant 0 : i64
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4x512xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c8192_i64) : memref<4xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c0_i64 : i64) outs(%6 : memref<4xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %7 = hivm.hir.pointer_cast(%c8224_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%7 : memref<4xi32, #hivm.address_space<ub>>)
  %8 = arith.extsi %arg5 : i32 to i64
  %9 = arith.index_cast %arg11 : i32 to index
  %10 = arith.index_cast %arg3 : i32 to index
  %11 = arith.index_cast %arg9 : i32 to index
  %12 = arith.index_cast %arg13 : i32 to index
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    %13 = arith.index_cast %arg17 : i32 to index
    %14 = arith.index_cast %3 : i32 to index
    %15 = arith.index_cast %1 : i32 to index
    %16 = arith.index_cast %c48_i32 : i32 to index
    %17 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%13, %14, %16]
    %18 = arith.index_cast %17 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %19 = arith.select %18, %c0_i64_1, %c1_i64 : i64
    %20 = hivm.hir.pointer_cast(%c34336_i64, %c50976_i64) : memref<4x512xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<4x512xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %21 = arith.divsi %arg17, %4 : i32
    %22 = arith.remsi %21, %arg14 : i32
    %23 = arith.muli %22, %c4_i32 : i32
    %24 = hivm.hir.pointer_cast(%c8256_i64) : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%7, %23 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%24 : memref<4xi32, #hivm.address_space<ub>>)
    %25 = hivm.hir.pointer_cast(%c8288_i64) : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmax ins(%24, %arg3 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%25 : memref<4xi32, #hivm.address_space<ub>>)
    %26 = hivm.hir.pointer_cast(%c8256_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%25, %24 : memref<4xi32, #hivm.address_space<ub>>, memref<4xi32, #hivm.address_space<ub>>) outs(%26 : memref<4xi1, #hivm.address_space<ub>>)
    %27 = hivm.hir.pointer_cast(%c8256_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%26 : memref<4xi1, #hivm.address_space<ub>>) outs(%27 : memref<4xi1, #hivm.address_space<ub>>)
    %28 = arith.index_cast %23 : i32 to index
    %29 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%28, %10]
    %30 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%29, %28]
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %28]
    %32 = arith.cmpi slt, %31, %c4 : index
    %33 = hivm.hir.pointer_cast(%c8320_i64) : memref<4xf16, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c8352_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%27 : memref<4xi1, #hivm.address_space<ub>>) outs(%33 : memref<4xf16, #hivm.address_space<ub>>) temp_buffer(%34 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %expand_shape = memref.expand_shape %33 [[0, 1]] output_shape [4, 1] : memref<4xf16, #hivm.address_space<ub>> into memref<4x1xf16, #hivm.address_space<ub>>
    %35 = hivm.hir.pointer_cast(%c8448_i64) : memref<4x512xf16, #hivm.address_space<ub>>
    %36 = hivm.hir.pointer_cast(%c12544_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape : memref<4x1xf16, #hivm.address_space<ub>>) outs(%35 : memref<4x512xf16, #hivm.address_space<ub>>) temp_buffer(%36 : memref<128xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
    %37 = hivm.hir.pointer_cast(%c8448_i64) : memref<4x512xi1, #hivm.address_space<ub>>
    %collapse_shape_2 = memref.collapse_shape %35 [[0, 1]] : memref<4x512xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %37 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_2, %cst_0 : memref<2048xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_3 : memref<2048xi1, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c8448_i64) : memref<4x512xi1, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %38 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_3 : memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<2048xi1, #hivm.address_space<ub>>)
    %39 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%30, %28]
    %collapse_shape_5 = memref.collapse_shape %20 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %19]
    hivm.hir.copy ins(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<2048xf32, #hivm.address_space<ub>>)
    %40 = scf.for %arg18 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg19 = %20) -> (memref<4x512xf32, #hivm.address_space<ub>>)  : i32 {
      %42 = arith.index_cast %arg18 : i32 to index
      %43 = arith.index_cast %c0_i32 : i32 to index
      %44 = arith.index_cast %c2_i32 : i32 to index
      %45 = arith.index_cast %c1_i32 : i32 to index
      %46 = arith.index_cast %arg17 : i32 to index
      %47 = arith.index_cast %3 : i32 to index
      %48 = arith.index_cast %1 : i32 to index
      %49 = arith.index_cast %c48_i32 : i32 to index
      %50 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%42, %43, %44, %45, %46, %47, %49]
      %51 = arith.index_cast %50 : index to i1
      %c1_i64_7 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %52 = arith.select %51, %c1_i64_7, %c2_i64 : i64
      %53 = arith.index_cast %50 : index to i1
      %c0_i64_8 = arith.constant 0 : i64
      %c1_i64_9 = arith.constant 1 : i64
      %54 = arith.select %53, %c0_i64_8, %c1_i64_9 : i64
      %55 = hivm.hir.pointer_cast(%c17632_i64, %c42784_i64) : memref<4x512xf32, #hivm.address_space<ub>>
      annotation.mark %55 {hivm.multi_buffer = 2 : i32} : memref<4x512xf32, #hivm.address_space<ub>>
      %56 = hivm.hir.pointer_cast(%c12928_i64, %c42656_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
      annotation.mark %56 {hivm.multi_buffer = 2 : i32} : memref<4x8x1xf32, #hivm.address_space<ub>>
      %57 = hivm.hir.pointer_cast(%c12800_i64, %c42528_i64) : memref<4x4xi64, #hivm.address_space<ub>>
      annotation.mark %57 {hivm.multi_buffer = 2 : i32} : memref<4x4xi64, #hivm.address_space<ub>>
      %58 = arith.index_cast %arg18 : i32 to index
      %59 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%58, %28, %9]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg10 to offset: [%59], sizes: [4], strides: [%9] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %subview_11 = memref.subview %57[0, 0] [4, 1] [1, 1] : memref<4x4xi64, #hivm.address_space<ub>> to memref<4xi64, strided<[4]>, #hivm.address_space<ub>>
      %subview_12 = memref.subview %reinterpret_cast_10[0] [%31] [1] : memref<4xi64, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %subview_11[0] [%31] [1] : memref<4xi64, strided<[4]>, #hivm.address_space<ub>> to memref<?xi64, strided<[4]>, #hivm.address_space<ub>>
      scf.if %32 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %base_buffer_41, %offset_42, %sizes_43, %strides_44 = memref.extract_strided_metadata %subview_11 : memref<4xi64, strided<[4]>, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_45 = memref.reinterpret_cast %base_buffer_41 to offset: [0], sizes: [4, 1], strides: [4, 1] : memref<i64, #hivm.address_space<ub>> to memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%c0_i64 : i64) outs(%reinterpret_cast_45 : memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %54]
      hivm.hir.load ins(%subview_12 : memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?xi64, strided<[4]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i64 : i64 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %60 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%58, %28, %11]
      %reinterpret_cast_14 = memref.reinterpret_cast %arg8 to offset: [%60], sizes: [4, 1], strides: [%11, %11] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %subview_15 = memref.subview %56[0, 0, 0] [4, 1, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%31, 1] [1, 1] : memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %subview_17 = memref.subview %subview_15[0, 0] [%31, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %32 {
        %collapse_shape_41 = memref.collapse_shape %subview_15 [[0, 1]] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_42, %offset_43, %sizes_44, %strides_45 = memref.extract_strided_metadata %collapse_shape_41 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_46 = memref.reinterpret_cast %base_buffer_42 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_46 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_18 = memref.collapse_shape %subview_16 [[0, 1]] : memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_19 = memref.collapse_shape %subview_17 [[0, 1]] : memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %52]
      hivm.hir.load ins(%collapse_shape_18 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_19 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %61 = hivm.hir.pointer_cast(%c13120_i64) : memref<4xi1, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c13056_i64) : memref<4xi8, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_11 : memref<4xi64, strided<[4]>, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_20 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [4, 1] : memref<i64, #hivm.address_space<ub>> to memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>
      %base_buffer_21, %offset_22, %sizes_23, %strides_24 = memref.extract_strided_metadata %6 : memref<4xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_25 = memref.reinterpret_cast %base_buffer_21 to offset: [0], sizes: [4, 1], strides: [1, 1] : memref<i64, #hivm.address_space<ub>> to memref<4x1xi64, strided<[1, 1]>, #hivm.address_space<ub>>
      %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %62 : memref<4xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_30 = memref.reinterpret_cast %base_buffer_26 to offset: [0], sizes: [4, 1], strides: [1, 1] : memref<i8, #hivm.address_space<ub>> to memref<4x1xi8, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg20 = %c0 to %c4 step %c1 {
        scf.for %arg21 = %c0 to %c1 step %c1 {
          %76 = memref.load %reinterpret_cast_20[%arg20, %arg21] : memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>
          %77 = memref.load %reinterpret_cast_25[%arg20, %arg21] : memref<4x1xi64, strided<[1, 1]>, #hivm.address_space<ub>>
          %78 = arith.cmpi sge, %76, %77 : i64
          %79 = arith.extui %78 : i1 to i8
          memref.store %79, %reinterpret_cast_30[%arg20, %arg21] : memref<4x1xi8, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %63 = hivm.hir.pointer_cast(%c13088_i64) : memref<4xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%62 : memref<4xi8, #hivm.address_space<ub>>) outs(%63 : memref<4xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %64 = hivm.hir.pointer_cast(%c13120_i64) : memref<4xf16, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f16) outs(%64 : memref<4xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%63, %64 : memref<4xf16, #hivm.address_space<ub>>, memref<4xf16, #hivm.address_space<ub>>) outs(%61 : memref<4xi1, #hivm.address_space<ub>>) compare_mode = <ne>
      %65 = hivm.hir.pointer_cast(%c13152_i64) : memref<4xf16, #hivm.address_space<ub>>
      %66 = hivm.hir.pointer_cast(%c13184_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%61 : memref<4xi1, #hivm.address_space<ub>>) outs(%65 : memref<4xf16, #hivm.address_space<ub>>) temp_buffer(%66 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %expand_shape_31 = memref.expand_shape %65 [[0, 1]] output_shape [4, 1] : memref<4xf16, #hivm.address_space<ub>> into memref<4x1xf16, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xf16, #hivm.address_space<ub>>
      %68 = hivm.hir.pointer_cast(%c17376_i64) : memref<128xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_31 : memref<4x1xf16, #hivm.address_space<ub>>) outs(%67 : memref<4x512xf16, #hivm.address_space<ub>>) temp_buffer(%68 : memref<128xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
      %69 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xi1, #hivm.address_space<ub>>
      %collapse_shape_32 = memref.collapse_shape %67 [[0, 1]] : memref<4x512xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
      %collapse_shape_33 = memref.collapse_shape %69 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_32, %cst_0 : memref<2048xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_33 : memref<2048xi1, #hivm.address_space<ub>>)
      %70 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xi1, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %70 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_33 : memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<2048xi1, #hivm.address_space<ub>>)
      scf.for %arg20 = %c0 to %39 step %c1 {
        %76 = memref.load %subview_11[%arg20] {DiscreteMemAccess} : memref<4xi64, strided<[4]>, #hivm.address_space<ub>>
        %77 = arith.muli %76, %8 : i64
        %78 = arith.index_cast %77 : i64 to index
        %reinterpret_cast_41 = memref.reinterpret_cast %arg4 to offset: [%78], sizes: [1, 512], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_42 = memref.subview %55[%arg20, 0] [1, 512] [1, 1] : memref<4x512xf32, #hivm.address_space<ub>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_43 = memref.collapse_shape %reinterpret_cast_41 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_44 = memref.collapse_shape %subview_42 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape_43 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_44 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      } {ExtractedLoadOrStore, hivm.parallel_loop, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %54]
      %71 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xi1, #hivm.address_space<ub>>
      %collapse_shape_35 = memref.collapse_shape %71 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vand ins(%collapse_shape_4, %collapse_shape_34 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<2048xi1, #hivm.address_space<ub>>)
      %72 = hivm.hir.pointer_cast(%c25824_i64) : memref<4x512xf32, #hivm.address_space<ub>>
      %collapse_shape_36 = memref.collapse_shape %55 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_37 = memref.collapse_shape %72 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %73 = hivm.hir.pointer_cast(%c34016_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_35, %collapse_shape_36, %cst : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_37 : memref<2048xf32, #hivm.address_space<ub>>) temp_buffer(%73 : memref<16xf32, #hivm.address_space<ub>>)
      %74 = hivm.hir.pointer_cast(%c25824_i64) : memref<4x512xf32, #hivm.address_space<ub>>
      %75 = hivm.hir.pointer_cast(%c34080_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%72, %subview_15 : memref<4x512xf32, #hivm.address_space<ub>>, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%74 : memref<4x512xf32, #hivm.address_space<ub>>) temp_buffer(%75 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %52]
      %collapse_shape_38 = memref.collapse_shape %arg19 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_39 = memref.collapse_shape %74 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_40 = memref.collapse_shape %20 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_38, %collapse_shape_39 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<2048xf32, #hivm.address_space<ub>>)
      scf.yield %20 : memref<4x512xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%28, %12]
    %reinterpret_cast = memref.reinterpret_cast %arg12 to offset: [%41], sizes: [4, 512], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %40[0, 0] [%31, 512] [1, 1] : memref<4x512xf32, #hivm.address_space<ub>> to memref<?x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast[0, 0] [%31, 512] [1, 1] : memref<4x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<?x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %19]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
