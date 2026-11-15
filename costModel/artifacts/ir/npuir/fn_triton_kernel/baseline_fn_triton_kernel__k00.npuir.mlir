// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fn_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c86016_i64 = arith.constant 86016 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c83968_i64 = arith.constant 83968 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c51200_i64 = arith.constant 51200 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.index_cast %arg6 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg10 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = arith.index_cast %10 : index to i1
    %c1_i64_1 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %14 = arith.select %13, %c1_i64_1, %c2_i64 : i64
    %15 = hivm.hir.pointer_cast(%c34816_i64, %c86016_i64) : memref<16x8x32xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<16x8x32xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c32768_i64, %c83968_i64) : memref<16x1x32xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<16x1x32xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c0_i64, %c51200_i64) : memref<16x8x64xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<16x8x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg10, %4 : i32
    %19 = arith.remsi %18, %arg7 : i32
    %20 = arith.muli %19, %c16_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = affine.apply affine_map<()[s0] -> (s0 * 512)>()[%21]
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [16, 8, 64], strides: [512, 64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
    %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%21, %5]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%23, %21]
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %21]
    %26 = arith.cmpi slt, %25, %c16 : index
    %subview = memref.subview %reinterpret_cast[0, 0, 0] [%25, 8, 64] [1, 1, 1] : memref<16x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %17[0, 0, 0] [%25, 8, 64] [1, 1, 1] : memref<16x8x64xf32, #hivm.address_space<ub>> to memref<?x8x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>>
    scf.if %26 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_18 = memref.collapse_shape %17 [[0, 1, 2]] : memref<16x8x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_18 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape = memref.collapse_shape %subview [[0, 1, 2]] : memref<?x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_3 = memref.collapse_shape %subview_2 [[0, 1, 2]] : memref<?x8x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %27 = affine.apply affine_map<()[s0] -> (s0 * 768)>()[%21]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [16, 8, 64], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %reinterpret_cast_4[0, 0, 0] [%25, 8, 64] [1, 1, 1] : memref<16x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_6 = memref.collapse_shape %subview_2 [[0, 1], [2]] : memref<?x8x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> into memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<?x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x64xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%collapse_shape_6 : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<?x64xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %12]
    %28 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%21]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg5 to offset: [%28], sizes: [16, 1, 32], strides: [32, 32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %reinterpret_cast_8[0, 0, 0] [%25, 1, 32] [1, 1, 1] : memref<16x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %16[0, 0, 0] [%25, 1, 32] [1, 1, 1] : memref<16x1x32xf32, #hivm.address_space<ub>> to memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
    scf.if %26 {
      %collapse_shape_18 = memref.collapse_shape %16 [[0, 1, 2]] : memref<16x1x32xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_18 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_12 = memref.collapse_shape %subview_10 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %14]
    hivm.hir.load ins(%collapse_shape_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %29 = affine.apply affine_map<()[s0] -> (s0 * 768 + 64)>()[%21]
    %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [16, 8, 32], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vbrc ins(%16 : memref<16x1x32xf32, #hivm.address_space<ub>>) outs(%15 : memref<16x8x32xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %14]
    %subview_14 = memref.subview %15[0, 0, 0] [%25, 8, 32] [1, 1, 1] : memref<16x8x32xf32, #hivm.address_space<ub>> to memref<?x8x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>>
    %subview_15 = memref.subview %reinterpret_cast_13[0, 0, 0] [%25, 8, 32] [1, 1, 1] : memref<16x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_16 = memref.collapse_shape %subview_14 [[0, 1], [2]] : memref<?x8x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>> into memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %subview_15 [[0, 1], [2]] : memref<?x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x32xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%collapse_shape_16 : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<?x32xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
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
