// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c5_i32 = arith.constant 5 : i32
  %true = arith.constant true
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
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  memref.store %true, %5[%c0] : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg7 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg7, %4 : i32
    %7 = arith.remsi %6, %arg4 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9:2 = scf.for %arg8 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg9 = %c0_i32, %arg10 = %c4_i32) -> (i32, i32)  : i32 {
      %10 = hivm.hir.pointer_cast(%c160_i64, %c192_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %11 = arith.index_cast %arg10 : i32 to index
      %12 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%11, %8]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %13 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %14 = hivm.hir.pointer_cast(%c32_i64) : memref<1xi32, #hivm.address_space<ub>>
      memref.store %13, %14[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %15 = hivm.hir.pointer_cast(%c64_i64) : memref<1xi32, #hivm.address_space<ub>>
      %16 = hivm.hir.pointer_cast(%c96_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%5, %14, %c0_i32 : memref<1xi1, #hivm.address_space<ub>>, memref<1xi32, #hivm.address_space<ub>>, i32) outs(%15 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%16 : memref<16xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %17 = memref.load %15[%c0] : memref<1xi32, #hivm.address_space<ub>>
      %18 = arith.addi %arg9, %17 : i32
      memref.store %18, %10[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%10 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %19 = arith.addi %arg10, %c4_i32 : i32
      scf.yield %18, %19 : i32, i32
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
