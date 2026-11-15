// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c5_i32 = arith.constant 5 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg4, %arg5 : i32
  %1 = arith.muli %0, %arg6 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg6, %arg5 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg7 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg7, %4 : i32
    %6 = arith.remsi %5, %arg4 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = scf.for %arg8 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg9 = %c0_i32) -> (i32)  : i32 {
      %9 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %10 = arith.muli %arg8, %c4_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%11, %7]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %13 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %14 = arith.addi %arg9, %13 : i32
      memref.store %14, %9[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%9 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.yield %14 : i32
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
