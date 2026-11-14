// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @mean_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c82048_i64 = arith.constant 82048 : i64
  %c49248_i64 = arith.constant 49248 : i64
  %c32832_i64 = arith.constant 32832 : i64
  %c32800_i64 = arith.constant 32800 : i64
  %c49280_i64 = arith.constant 49280 : i64
  %c32_i64 = arith.constant 32 : i64
  %c49216_i64 = arith.constant 49216 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c8192_i32 = arith.constant 8192 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8192 = arith.constant 8192 : index
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst, %5[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg11 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c49248_i64, %c82048_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg11, %4 : i32
    %15 = arith.remsi %14, %arg8 : i32
    %16 = arith.cmpi sge, %15, %arg6 : i32
    scf.if %16 {
    } else {
      %17 = arith.muli %15, %arg5 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %arg7 : i32 to index
      %20 = hivm.hir.pointer_cast(%c49216_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%5 : memref<1xf32, #hivm.address_space<ub>>) outs(%20 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %21 = scf.for %arg12 = %c0_i32 to %arg7 step %c8192_i32 iter_args(%arg13 = %20) -> (memref<1xf32, #hivm.address_space<ub>>)  : i32 {
        %24 = hivm.hir.pointer_cast(%c32_i64, %c49280_i64) : memref<8192x1xf32, #hivm.address_space<ub>>
        annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8192x1xf32, #hivm.address_space<ub>>
        %expand_shape = memref.expand_shape %arg13 [[0, 1]] output_shape [1, 1] : memref<1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
        %25 = arith.index_cast %arg12 : i32 to index
        %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%18, %25]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [8192, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
        %27 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%25, %19]
        %28 = affine.min affine_map<()[s0, s1] -> (s1 + 8192, s0)>()[%27, %25]
        %29 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%28, %25]
        %30 = arith.cmpi slt, %29, %c8192 : index
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%29, 1] [1, 1] : memref<8192x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %24[0, 0] [%29, 1] [1, 1] : memref<8192x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.if %30 {
          %collapse_shape_7 = memref.collapse_shape %24 [[0, 1]] : memref<8192x1xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_7 : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_3 = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %31 = hivm.hir.pointer_cast(%c32800_i64) : memref<1x1xf32, #hivm.address_space<ub>>
        %collapse_shape_4 = memref.collapse_shape %24 [[0, 1]] : memref<8192x1xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %collapse_shape_5 = memref.collapse_shape %31 [[0, 1]] : memref<1x1xf32, #hivm.address_space<ub>> into memref<1xf32, #hivm.address_space<ub>>
        %32 = hivm.hir.pointer_cast(%c32832_i64) : memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vreduce <sum> ins(%collapse_shape_4 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%32 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %33 = hivm.hir.pointer_cast(%c49216_i64) : memref<1x1xf32, #hivm.address_space<ub>>
        %34 = memref.load %expand_shape[%c0, %c0] : memref<1x1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %35 = memref.load %31[%c0, %c0] : memref<1x1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
        %36 = arith.addf %34, %35 : f32
        memref.store %36, %33[%c0, %c0] : memref<1x1xf32, #hivm.address_space<ub>>
        %collapse_shape_6 = memref.collapse_shape %33 [[0, 1]] : memref<1x1xf32, #hivm.address_space<ub>> into memref<1xf32, #hivm.address_space<ub>>
        scf.yield %collapse_shape_6 : memref<1xf32, #hivm.address_space<ub>>
      }
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %22 = arith.sitofp %arg7 : i32 to f32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %12]
      memref.store %22, %13[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%21, %13 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%13 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %23 = arith.index_cast %15 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%13 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
