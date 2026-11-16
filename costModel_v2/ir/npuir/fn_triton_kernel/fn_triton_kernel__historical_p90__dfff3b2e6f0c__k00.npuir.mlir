// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fn_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c139264_i64 = arith.constant 139264 : i64
  %c135168_i64 = arith.constant 135168 : i64
  %c131072_i64 = arith.constant 131072 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c0_i64 = arith.constant 0 : i64
  %c8_i32 = arith.constant 8 : i32
  %c32_i32 = arith.constant 32 : i32
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.index_cast %arg6 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %6 = hivm.hir.pointer_cast(%c139264_i64, %c0_i64) : memref<32x8x32xf32, #hivm.address_space<ub>>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<32x8x32xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c131072_i64, %c135168_i64) : memref<32x1x32xf32, #hivm.address_space<ub>>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<32x1x32xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<32x8x64xf32, #hivm.address_space<ub>>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<32x8x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %9 = arith.divsi %arg10, %arg9 : i32
    %10 = arith.remsi %9, %arg8 : i32
    %11 = arith.divsi %arg10, %4 : i32
    %12 = arith.remsi %11, %arg7 : i32
    %13 = arith.muli %12, %c32_i32 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = arith.index_cast %13 : i32 to index
    %16 = arith.index_cast %14 : i32 to index
    %17 = affine.apply affine_map<()[s0, s1] -> (s0 * 512 + s1 * 64)>()[%15, %16]
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [32, 8, 64], strides: [512, 64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
    %18 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%15, %5]
    %19 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%18, %15]
    %20 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%19, %15]
    %21 = affine.max affine_map<()[s0] -> (8, s0)>()[%16]
    %22 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%21, %16]
    %23 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%19, %15]
    %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
    %25 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%22, %16]
    %26 = affine.max affine_map<()[s0] -> (0, s0)>()[%25]
    %27 = arith.cmpi slt, %24, %c32 : index
    %28 = arith.cmpi slt, %26, %c8 : index
    %29 = arith.ori %27, %28 : i1
    %subview = memref.subview %reinterpret_cast[0, 0, 0] [%24, %26, 64] [1, 1, 1] : memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %8[0, 0, 0] [%24, %26, 64] [1, 1, 1] : memref<32x8x64xf32, #hivm.address_space<ub>> to memref<?x?x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>>
    scf.if %29 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_12 = memref.collapse_shape %8 [[0, 1, 2]] : memref<32x8x64xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_12 : memref<16384xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape = memref.collapse_shape %subview [[0], [1, 2]] : memref<?x?x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [[0], [1, 2]] : memref<?x?x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> into memref<?x?xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_1 : memref<?x?xf32, strided<[512, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %30 = affine.apply affine_map<()[s0, s1] -> (s0 * 768 + s1 * 96)>()[%15, %16]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [32, 8, 64], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %reinterpret_cast_2[0, 0, 0] [%24, %26, 64] [1, 1, 1] : memref<32x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_0 : memref<?x?x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?x?x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %31 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%15]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [32, 1, 32], strides: [32, 32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
    %32 = arith.cmpi slt, %20, %c32 : index
    %subview_5 = memref.subview %reinterpret_cast_4[0, 0, 0] [%20, 1, 32] [1, 1, 1] : memref<32x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %7[0, 0, 0] [%20, 1, 32] [1, 1, 1] : memref<32x1x32xf32, #hivm.address_space<ub>> to memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
    scf.if %32 {
      %collapse_shape_12 = memref.collapse_shape %7 [[0, 1, 2]] : memref<32x1x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_12 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_7 = memref.collapse_shape %subview_5 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_8 = memref.collapse_shape %subview_6 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %33 = affine.apply affine_map<()[s0, s1] -> (s0 * 768 + s1 * 96 + 64)>()[%15, %16]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [32, 8, 32], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vbrc ins(%7 : memref<32x1x32xf32, #hivm.address_space<ub>>) outs(%6 : memref<32x8x32xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_10 = memref.subview %6[0, 0, 0] [%24, %26, 32] [1, 1, 1] : memref<32x8x32xf32, #hivm.address_space<ub>> to memref<?x?x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>>
    %subview_11 = memref.subview %reinterpret_cast_9[0, 0, 0] [%24, %26, 32] [1, 1, 1] : memref<32x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_10 : memref<?x?x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<?x?x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>)
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
