// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @experts_combine_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c40960_i64 = arith.constant 40960 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c28672_i64 = arith.constant 28672 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.707106769 : f32
  %c1024_i32 = arith.constant 1024 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg9 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c24576_i64, %c40960_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c4096_i64, %c32768_i64) : memref<2x1024xf32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x1024xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c0_i64, %c28672_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg9, %4 : i32
    %16 = arith.remsi %15, %arg6 : i32
    %17 = arith.muli %16, %c1024_i32 : i32
    %18 = arith.muli %16, %c2048_i32 : i32
    %19 = arith.index_cast %17 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%19], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%reinterpret_cast : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%14 : memref<1024xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    %20 = arith.index_cast %18 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [2, 1024], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast_1 [[0, 1]] : memref<2x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_2 = memref.collapse_shape %13 [[0, 1]] : memref<2x1024xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_2 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %21 = hivm.hir.pointer_cast(%c12288_i64) : memref<1x1024xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c16384_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <sum> ins(%13 : memref<2x1024xf32, #hivm.address_space<ub>>) outs(%21 : memref<1x1024xf32, #hivm.address_space<ub>>) temp_buffer(%22 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_3 = memref.collapse_shape %21 [[0, 1]] : memref<1x1024xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_3, %14 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%14 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %11]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%14, %cst : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%12 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %11]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%12 : memref<1024xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %11]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
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
