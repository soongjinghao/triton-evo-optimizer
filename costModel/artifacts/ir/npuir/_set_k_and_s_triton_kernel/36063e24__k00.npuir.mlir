// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_set_k_and_s_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c256_i64 = arith.constant 256 : i64
  %c0_i64 = arith.constant 0 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c4_i64 = arith.constant 4 : i64
  %c128_i64 = arith.constant 128 : i64
  %c8448_i64 = arith.constant 8448 : i64
  %c63_i64 = arith.constant 63 : i64
  %c6_i64 = arith.constant 6 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg12, %4 : i32
    %6 = arith.remsi %5, %arg9 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%7], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %8 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %9 = arith.muli %6, %arg8 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%10], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %11 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%reinterpret_cast_0 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%11 : memref<128xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%7], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %12 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %13 = arith.shrsi %8, %c6_i64 : i64
    %14 = arith.andi %8, %c63_i64 : i64
    %15 = arith.muli %13, %c8448_i64 : i64
    %16 = arith.muli %14, %c128_i64 : i64
    %17 = arith.addi %15, %16 : i64
    %18 = arith.divsi %15, %c4_i64 : i64
    %19 = arith.addi %18, %c2048_i64 : i64
    %20 = arith.addi %19, %14 : i64
    %21 = arith.index_cast %17 : i64 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%11 : memref<128xf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    %22 = hivm.hir.pointer_cast(%c256_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    memref.store %12, %22[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %23 = arith.index_cast %20 : i64 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%22 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
