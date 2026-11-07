// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage3(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32_i64 = arith.constant 32 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c5_i32 = arith.constant 5 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c31_i32 = arith.constant 31 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  %2 = scf.for %arg9 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg10 = %c0_i32) -> (i32)  : i32 {
    %4 = arith.index_cast %arg9 : i32 to index
    %5 = arith.index_cast %c1_i32 : i32 to index
    %6 = arith.index_cast %c5_i32 : i32 to index
    %7 = arith.index_cast %c1_i32 : i32 to index
    %8 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%4, %5, %7]
    %9 = arith.index_cast %8 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %10 = arith.select %9, %c0_i64_0, %c1_i64 : i64
    %11 = hivm.hir.pointer_cast(%c0_i64, %c64_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = affine.apply affine_map<()[s0] -> (s0 + 15)>()[%12]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %14 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = arith.addi %14, %c31_i32 : i32
    %16 = arith.divsi %15, %c32_i32 : i32
    %17 = arith.muli %16, %c32_i32 : i32
    %18 = arith.addi %arg10, %17 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %10]
    memref.store %18, %11[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%11 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %10]
    scf.yield %18 : i32
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  %3 = hivm.hir.pointer_cast(%c32_i64) : memref<1xi32, #hivm.address_space<ub>>
  memref.store %2, %3[%c0] : memref<1xi32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%3 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
