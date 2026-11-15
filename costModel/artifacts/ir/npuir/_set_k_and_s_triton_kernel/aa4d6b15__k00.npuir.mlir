// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_set_k_and_s_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c544_i64 = arith.constant 544 : i64
  %c256_i64 = arith.constant 256 : i64
  %c288_i64 = arith.constant 288 : i64
  %c0_i64 = arith.constant 0 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c4_i64 = arith.constant 4 : i64
  %c128_i64 = arith.constant 128 : i64
  %c8448_i64 = arith.constant 8448 : i64
  %c64_i64 = arith.constant 64 : i64
  %c128_i32 = arith.constant 128 : i32
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg12 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c256_i64, %c544_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c0_i64, %c288_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg12, %4 : i32
    %15 = arith.remsi %14, %arg9 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%16], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %17 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = arith.muli %15, %c128_i32 : i32
    %19 = arith.index_cast %18 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%19], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%reinterpret_cast_1 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%13 : memref<128xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%16], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = memref.load %reinterpret_cast_2[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %21 = arith.divsi %17, %c64_i64 : i64
    %22 = arith.remsi %17, %c64_i64 : i64
    %23 = arith.muli %21, %c8448_i64 : i64
    %24 = arith.muli %22, %c128_i64 : i64
    %25 = arith.addi %23, %24 : i64
    %26 = arith.divsi %23, %c4_i64 : i64
    %27 = arith.addi %26, %c2048_i64 : i64
    %28 = arith.addi %27, %22 : i64
    %29 = arith.index_cast %25 : i64 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%13 : memref<128xf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %11]
    memref.store %20, %12[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %30 = arith.index_cast %28 : i64 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%30], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%12 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %11]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
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
