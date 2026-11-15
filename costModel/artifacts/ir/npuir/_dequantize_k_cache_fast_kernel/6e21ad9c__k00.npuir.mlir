// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_dequantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c7680_i64 = arith.constant 7680 : i64
  %c3456_i64 = arith.constant 3456 : i64
  %c3200_i64 = arith.constant 3200 : i64
  %c7552_i64 = arith.constant 7552 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c5504_i64 = arith.constant 5504 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c4480_i64 = arith.constant 4480 : i64
  %c0_i64 = arith.constant 0 : i64
  %c128_i32 = arith.constant 128 : i32
  %c4_i32 = arith.constant 4 : i32
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg14 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_1, %c1_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %13 = arith.select %12, %c2_i64, %c3_i64 : i64
    %14 = arith.index_cast %9 : index to i1
    %c1_i64_2 = arith.constant 1 : i64
    %c2_i64_3 = arith.constant 2 : i64
    %15 = arith.select %14, %c1_i64_2, %c2_i64_3 : i64
    %16 = hivm.hir.pointer_cast(%c3456_i64, %c7680_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c1024_i64, %c5504_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c3072_i64, %c7552_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<4x8x1xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c0_i64, %c4480_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg14, %arg13 : i32
    %21 = arith.remsi %20, %arg12 : i32
    %22 = arith.divsi %arg14, %4 : i32
    %23 = arith.remsi %22, %arg11 : i32
    %24 = arith.muli %23, %c4_i32 : i32
    %25 = arith.cmpi slt, %21, %c4_i32 : i32
    scf.if %25 {
      %26 = arith.muli %21, %c128_i32 : i32
      %27 = arith.index_cast %24 : i32 to index
      %28 = arith.index_cast %arg8 : i32 to index
      %29 = arith.index_cast %26 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%29, %27, %28]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%30], sizes: [4, 128], strides: [%28, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.index_cast %arg9 : i32 to index
      %32 = arith.index_cast %21 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%32, %27, %31]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [4, 1], strides: [%31, %31] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max affine_map<()[s0] -> (4, s0)>()[%27]
      %35 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%34, %27]
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %27]
      %37 = affine.max affine_map<()[s0] -> (512, s0)>()[%29]
      %38 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%37, %29]
      %39 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%35, %27]
      %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
      %41 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%38, %29]
      %42 = affine.max affine_map<()[s0] -> (0, s0)>()[%41]
      %43 = arith.cmpi slt, %40, %c4 : index
      %44 = arith.cmpi slt, %42, %c128 : index
      %45 = arith.ori %43, %44 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%40, %42] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %19[0, 0] [%40, %42] [1, 1] : memref<4x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
      scf.if %45 {
        %collapse_shape_17 = memref.collapse_shape %19 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape_17 : memref<512xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
      hivm.hir.load ins(%subview : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %46 = hivm.hir.pointer_cast(%c1024_i64, %c5504_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %19 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
      %collapse_shape_6 = memref.collapse_shape %46 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%collapse_shape : memref<512xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
      %subview_7 = memref.subview %18[0, 0, 0] [4, 1, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %47 = arith.cmpi slt, %36, %c4 : index
      %subview_8 = memref.subview %reinterpret_cast_4[0, 0] [%36, 1] [1, 1] : memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %subview_7[0, 0] [%36, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %47 {
        %collapse_shape_17 = memref.collapse_shape %subview_7 [[0, 1]] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %collapse_shape_17 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%reinterpret_cast_18 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      } {hivm.unlikely_condition}
      %collapse_shape_10 = memref.collapse_shape %subview_8 [[0, 1]] : memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%collapse_shape_10 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_11 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %48 = hivm.hir.pointer_cast(%c1024_i64, %c5504_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c3200_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%46, %subview_7 : memref<4x128xf32, #hivm.address_space<ub>>, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%48 : memref<4x128xf32, #hivm.address_space<ub>>) temp_buffer(%49 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %11]
      %collapse_shape_12 = memref.collapse_shape %48 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_13 = memref.collapse_shape %17 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%collapse_shape_12 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<512xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %50 = arith.index_cast %arg7 : i32 to index
      %51 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%29, %27, %50]
      %reinterpret_cast_14 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [4, 128], strides: [%50, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_15 = memref.subview %17[0, 0] [%40, %42] [1, 1] : memref<4x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%40, %42] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_15 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_16 : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    } else {
      %26 = arith.subi %21, %c4_i32 : i32
      %27 = arith.muli %26, %c128_i32 : i32
      %28 = arith.index_cast %24 : i32 to index
      %29 = arith.index_cast %arg10 : i32 to index
      %30 = arith.index_cast %27 : i32 to index
      %31 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%30, %28, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%31], sizes: [4, 128], strides: [%29, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %32 = arith.index_cast %arg7 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2 + 512)>()[%30, %28, %32]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [4, 128], strides: [%32, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max affine_map<()[s0] -> (4, s0)>()[%28]
      %35 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%34, %28]
      %36 = affine.max affine_map<()[s0] -> (64, s0)>()[%30]
      %37 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%36, %30]
      %38 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%35, %28]
      %39 = affine.max affine_map<()[s0] -> (0, s0)>()[%38]
      %40 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%37, %30]
      %41 = affine.max affine_map<()[s0] -> (0, s0)>()[%40]
      %42 = arith.cmpi slt, %39, %c4 : index
      %43 = arith.cmpi slt, %41, %c128 : index
      %44 = arith.ori %42, %43 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%39, %41] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %16[0, 0] [%39, %41] [1, 1] : memref<4x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
      scf.if %44 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<512xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%subview : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%39, %41] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_5 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
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
