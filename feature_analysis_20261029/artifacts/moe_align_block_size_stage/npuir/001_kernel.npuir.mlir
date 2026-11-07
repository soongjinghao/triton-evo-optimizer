// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg7 : i32
  %7 = arith.muli %6, %arg6 : i32
  %8 = arith.addi %6, %c1_i32 : i32
  %9 = arith.muli %8, %c4_i32 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg10 = %c0_i32 to %arg6 step %c1_i32  : i32 {
    %10 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    %11 = arith.addi %7, %arg10 : i32
    %12 = arith.cmpi slt, %11, %arg5 : i32
    scf.if %12 {
      %13 = arith.index_cast %7 : i32 to index
      %14 = arith.index_cast %arg10 : i32 to index
      %15 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %14]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %16 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %17 = arith.index_cast %9 : i32 to index
      %18 = arith.index_cast %16 : i32 to index
      %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%17, %18]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%19], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %20 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %21 = arith.addi %20, %c1_i32 : i32
      memref.store %21, %10[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%10 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    }
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
