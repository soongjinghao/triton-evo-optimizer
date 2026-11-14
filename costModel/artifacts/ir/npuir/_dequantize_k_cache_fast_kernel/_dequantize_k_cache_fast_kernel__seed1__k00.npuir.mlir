// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_dequantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c6592_i64 = arith.constant 6592 : i64
  %c3360_i64 = arith.constant 3360 : i64
  %c3104_i64 = arith.constant 3104 : i64
  %c6560_i64 = arith.constant 6560 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c4512_i64 = arith.constant 4512 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c3488_i64 = arith.constant 3488 : i64
  %c0_i64 = arith.constant 0 : i64
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
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
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %11 = arith.select %10, %c2_i64, %c3_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_0, %c1_i64 : i64
    %14 = hivm.hir.pointer_cast(%c3360_i64, %c6592_i64) : memref<64xbf16, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<64xbf16, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c1024_i64, %c4512_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c3072_i64, %c6560_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x1xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c0_i64, %c3488_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg14, %4 : i32
    %19 = arith.remsi %18, %arg11 : i32
    %20 = arith.muli %19, %arg8 : i32
    %21 = arith.index_cast %20 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [4, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_1 = memref.collapse_shape %17 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%collapse_shape : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_1 : memref<512xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %22 = hivm.hir.pointer_cast(%c1024_i64, %c4512_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %collapse_shape_2 = memref.collapse_shape %22 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %13]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%collapse_shape_1 : memref<512xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<512xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    %23 = arith.muli %19, %arg9 : i32
    %24 = arith.index_cast %23 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%24], sizes: [4, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_4 = memref.collapse_shape %reinterpret_cast_3 [[0, 1]] : memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_5 = memref.collapse_shape %16 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%collapse_shape_4 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_5 : memref<4xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %25 = hivm.hir.pointer_cast(%c1024_i64, %c4512_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c3104_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%22, %16 : memref<4x128xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%25 : memref<4x128xf32, #hivm.address_space<ub>>) temp_buffer(%26 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %11]
    %collapse_shape_6 = memref.collapse_shape %25 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %15 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_6 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<512xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %27 = arith.muli %19, %arg7 : i32
    %28 = arith.index_cast %27 : i32 to index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [4, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_9 = memref.collapse_shape %reinterpret_cast_8 [[0, 1]] : memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%collapse_shape_7 : memref<512xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %13]
    %29 = arith.muli %19, %arg10 : i32
    %30 = arith.index_cast %29 : i32 to index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%30], sizes: [64], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%reinterpret_cast_10 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%14 : memref<64xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %31 = affine.apply affine_map<()[s0] -> (s0 + 512)>()[%28]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [64], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%14 : memref<64xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
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
