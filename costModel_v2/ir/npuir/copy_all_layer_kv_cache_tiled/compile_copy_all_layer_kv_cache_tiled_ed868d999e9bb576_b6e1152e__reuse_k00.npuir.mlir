// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @copy_all_layer_kv_cache_tiled(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1664_i64 = arith.constant 1664 : i64
  %c832_i64 = arith.constant 832 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c576_i64 = arith.constant 576 : i64
  %c1376_i64 = arith.constant 1376 : i64
  %c544_i64 = arith.constant 544 : i64
  %c1344_i64 = arith.constant 1344 : i64
  %c512_i64 = arith.constant 512 : i64
  %c64_i32 = arith.constant 64 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [4], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[1]>, #hivm.address_space<gm>>
  %5 = arith.index_cast %arg7 : i32 to index
  %6 = affine.max affine_map<()[s0] -> (0, s0)>()[%5]
  %7 = affine.min affine_map<()[s0] -> (4, s0)>()[%6]
  %8 = arith.cmpi slt, %7, %c4 : index
  %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<4xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [4], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[1]>, #hivm.address_space<gm>>
  %subview_1 = memref.subview %reinterpret_cast_0[0] [%7] [1] : memref<4xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
  %9 = affine.max affine_map<()[s0] -> (0, s0)>()[%7]
  %10 = affine.min affine_map<()[s0] -> (4, s0)>()[%9]
  %11 = hivm.hir.pointer_cast(%c0_i64) : memref<4x64xf16, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %11 [[0, 1]] : memref<4x64xf16, #hivm.address_space<ub>> into memref<256xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<256xf16, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %12 = arith.index_cast %arg11 : i32 to index
    %13 = arith.index_cast %3 : i32 to index
    %14 = arith.index_cast %1 : i32 to index
    %15 = arith.index_cast %c48_i32 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%12, %13, %15]
    %17 = arith.index_cast %16 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %18 = arith.select %17, %c0_i64_2, %c1_i64 : i64
    %19 = arith.index_cast %16 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %20 = arith.select %19, %c2_i64, %c3_i64 : i64
    %21 = hivm.hir.pointer_cast(%c832_i64, %c1664_i64) : memref<4x64xi8, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<4x64xi8, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c576_i64, %c1408_i64) : memref<4x64xi8, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<4x64xi8, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c544_i64, %c1376_i64) : memref<4xi64, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<4xi64, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c512_i64, %c1344_i64) : memref<4xi64, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<4xi64, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %25 = arith.divsi %arg11, %arg10 : i32
    %26 = arith.remsi %25, %arg9 : i32
    %27 = arith.divsi %arg11, %4 : i32
    %28 = arith.remsi %27, %arg8 : i32
    %29 = arith.index_cast %28 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%29], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %30 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %31 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %32 = arith.muli %26, %c64_i32 : i32
    %subview_5 = memref.subview %24[0] [%7] [1] : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
    scf.if %8 {
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i64 : i64) outs(%24 : memref<4xi64, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %20]
    hivm.hir.load ins(%subview : memref<?xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i64 : i64 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
    %subview_6 = memref.subview %23[0] [%7] [1] : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
    scf.if %8 {
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i64 : i64) outs(%23 : memref<4xi64, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %18]
    hivm.hir.load ins(%subview_1 : memref<?xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i64 : i64 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %33 = arith.index_cast %32 : i32 to index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg12 = %c0 to %10 step %c1 {
      %40 = memref.load %24[%arg12] {DiscreteMemAccess} : memref<4xi64, #hivm.address_space<ub>>
      %41 = arith.muli %40, %30 : i64
      %42 = arith.index_cast %41 : i64 to index
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%42, %33]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.addi %31, %44 : i64
      %46 = hivm.hir.pointer_cast(%45) [%c64] : memref<?xi8, #hivm.address_space<gm>>
      annotation.mark %46 {address_space = #hivm.address_space<gm>} : memref<?xi8, #hivm.address_space<gm>>
      %reinterpret_cast_12 = memref.reinterpret_cast %46 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %22[%arg12, 0] [1, 64] [1, 1] : memref<4x64xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %reinterpret_cast_12 [[0, 1]] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_15 = memref.collapse_shape %subview_13 [[0, 1]] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> into memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.load ins(%collapse_shape_14 : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_15 : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    } {ExtractedLoadOrStore, hivm.parallel_loop, pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %20]
    %34 = arith.index_cast %30 : i64 to index
    %35 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%33, %34]
    %36 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%35, %33]
    %37 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%36, %33]
    %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
    %subview_7 = memref.subview %22[0, 0] [%9, %38] [1, 1] : memref<4x64xi8, #hivm.address_space<ub>> to memref<?x?xi8, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %11[0, 0] [%9, %38] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%subview_7 : memref<?x?xi8, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>)
    %39 = hivm.hir.pointer_cast(%c832_i64, %c1664_i64) : memref<4x64xf16, #hivm.address_space<ub>>
    %collapse_shape_9 = memref.collapse_shape %39 [[0, 1]] : memref<4x64xf16, #hivm.address_space<ub>> into memref<256xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape : memref<256xf16, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<256xf16, #hivm.address_space<ub>>)
    %subview_10 = memref.subview %39[0, 0] [%9, %38] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_8 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>)
    %collapse_shape_11 = memref.collapse_shape %21 [[0, 1]] : memref<4x64xi8, #hivm.address_space<ub>> into memref<256xi8, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_9 : memref<256xf16, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<256xi8, #hivm.address_space<ub>>) round_mode = <trunc>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.for %arg12 = %c0 to %10 step %c1 {
      %40 = memref.load %23[%arg12] {DiscreteMemAccess} : memref<4xi64, #hivm.address_space<ub>>
      %41 = arith.muli %40, %30 : i64
      %42 = arith.index_cast %41 : i64 to index
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%42, %33]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.addi %31, %44 : i64
      %46 = hivm.hir.pointer_cast(%45) [%c64] : memref<?xi8, #hivm.address_space<gm>>
      annotation.mark %46 {address_space = #hivm.address_space<gm>} : memref<?xi8, #hivm.address_space<gm>>
      %reinterpret_cast_12 = memref.reinterpret_cast %46 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %21[%arg12, 0] [1, 64] [1, 1] {DiscreteMemAccess} : memref<4x64xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg12 : index to i32
      %48 = arith.cmpi slt, %47, %arg7 : i32
      %49 = arith.index_castui %48 : i1 to index
      %50 = affine.min affine_map<()[s0] -> (1, s0)>()[%49]
      %51 = affine.max affine_map<()[s0] -> (0, s0)>()[%50]
      %52 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 64)>()[%49, %36, %33]
      %53 = affine.max affine_map<()[s0] -> (0, s0)>()[%52]
      %subview_14 = memref.subview %subview_13[0, 0] [%51, %53] [1, 1] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_15 = memref.subview %reinterpret_cast_12[0, 0] [%51, %53] [1, 1] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_14 : memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_15 : memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>)
    } {ExtractedLoadOrStore, pipeline.veconly}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %18]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
