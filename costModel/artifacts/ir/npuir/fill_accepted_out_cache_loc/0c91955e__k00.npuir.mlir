// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fill_accepted_out_cache_loc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c1088_i64 = arith.constant 1088 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c-1_i64 = arith.constant -1 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<128xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c1_i64 : i64) outs(%5 : memref<128xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %6 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xi64, #hivm.address_space<ub>>
  memref.store %c0_i64, %6[%c0] : memref<1xi64, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg9 = %c0 to %c128 step %c1 {
    %8 = memref.load %5[%arg9] : memref<128xi64, #hivm.address_space<ub>>
    %9 = arith.cmpi eq, %arg9, %c0 : index
    scf.if %9 {
      memref.store %c0_i64, %6[%c0] : memref<1xi64, #hivm.address_space<ub>>
    }
    %10 = memref.load %6[%c0] : memref<1xi64, #hivm.address_space<ub>>
    %11 = arith.addi %8, %10 : i64
    memref.store %11, %6[%c0] : memref<1xi64, #hivm.address_space<ub>>
  }
  %7 = memref.load %6[%c0] : memref<1xi64, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg9 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_0, %c1_i64_1 : i64
    %15 = hivm.hir.pointer_cast(%c1056_i64, %c1088_i64) : memref<1xi64, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<1xi64, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg9, %4 : i32
    %17 = arith.remsi %16, %arg6 : i32
    %18 = arith.index_cast %17 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = arith.cmpi sgt, %19, %c-1_i64 : i64
    scf.if %20 {
      %21 = arith.index_cast %19 : i64 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %22 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %14]
      memref.store %22, %15[%c0] : memref<1xi64, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %23 = arith.index_cast %7 : i64 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%23], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%15 : memref<1xi64, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %14]
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
