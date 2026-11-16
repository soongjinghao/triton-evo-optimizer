// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @l2norm_fwd_kernel2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: f32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c33024_i64 = arith.constant 33024 : i64
  %c32896_i64 = arith.constant 32896 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c34048_i64 = arith.constant 34048 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32x64xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg6 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg10 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_1, %c1_i64 : i64
    %14 = hivm.hir.pointer_cast(%c8192_i64, %c34048_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c8192_i64, %c34048_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg10, %4 : i32
    %17 = arith.remsi %16, %arg7 : i32
    %18 = arith.muli %17, %c32_i32 : i32
    %19 = arith.index_cast %18 : i32 to index
    %20 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%19]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [32, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %21 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%19, %6]
    %22 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%21, %19]
    %23 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%22, %19]
    %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
    %25 = arith.cmpi slt, %24, %c32 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%24, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %15[0, 0] [%24, 64] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_21 = memref.collapse_shape %15 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_21 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_3 = memref.collapse_shape %subview [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_4 = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%collapse_shape_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %26 = hivm.hir.pointer_cast(%c16384_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    %collapse_shape_5 = memref.collapse_shape %15 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %26 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_5, %collapse_shape_5 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<2048xf32, #hivm.address_space<ub>>)
    %subview_7 = memref.subview %26[0, 0] [%24, 64] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c24576_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %27 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<2048xf32, #hivm.address_space<ub>>)
    %subview_9 = memref.subview %27[0, 0] [%24, 64] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %collapse_shape_10 = memref.collapse_shape %subview_7 [[0, 1]] : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c32768_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%27 : memref<32x64xf32, #hivm.address_space<ub>>) outs(%28 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %29 = hivm.hir.pointer_cast(%c32768_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %28 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %29 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_12, %arg5 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_13 : memref<32xf32, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c32768_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %30 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%collapse_shape_13 : memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<32xf32, #hivm.address_space<ub>>)
    %31 = hivm.hir.pointer_cast(%c32896_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %31 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_15 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_15, %collapse_shape_14 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<32xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [32, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %32 = hivm.hir.pointer_cast(%c33024_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%15, %31 : memref<32x64xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%14 : memref<32x64xf32, #hivm.address_space<ub>>) temp_buffer(%32 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_17 = memref.subview %14[0, 0] [%24, 64] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %reinterpret_cast_16[0, 0] [%24, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_19 = memref.collapse_shape %subview_17 [[0, 1]] : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %subview_18 [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%collapse_shape_19 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
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
