// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage2_3_fused(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c160_i64 = arith.constant 160 : i64
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c32_i64 = arith.constant 32 : i64
  %c96_i64 = arith.constant 96 : i64
  %c0_i64 = arith.constant 0 : i64
  %c31_i32 = arith.constant 31 : i32
  %c5_i32 = arith.constant 5 : i32
  %c32_i32 = arith.constant 32 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %5 = hivm.hir.pointer_cast(%c64_i64, %c160_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg9, %4 : i32
    %7 = arith.remsi %6, %arg6 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = scf.for %arg10 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %c0_i32) -> (i32)  : i32 {
      %11 = hivm.hir.pointer_cast(%c0_i64, %c96_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %12 = arith.muli %arg10, %c4_i32 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %8]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      %15 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %16 = arith.addi %arg11, %15 : i32
      memref.store %16, %11[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%11 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      scf.yield %16 : i32
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    %10 = scf.for %arg10 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %c0_i32) -> (i32)  : i32 {
      %11 = arith.index_cast %arg10 : i32 to index
      %12 = arith.index_cast %c1_i32 : i32 to index
      %13 = arith.index_cast %c5_i32 : i32 to index
      %14 = arith.index_cast %c1_i32 : i32 to index
      %15 = arith.index_cast %arg9 : i32 to index
      %16 = arith.index_cast %3 : i32 to index
      %17 = arith.index_cast %1 : i32 to index
      %18 = arith.index_cast %c48_i32 : i32 to index
      %19 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%11, %12, %13, %14, %15, %16, %18]
      %20 = arith.index_cast %19 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %21 = arith.select %20, %c1_i64, %c2_i64 : i64
      %22 = hivm.hir.pointer_cast(%c32_i64, %c128_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %23 = arith.index_cast %arg10 : i32 to index
      %24 = affine.apply affine_map<()[s0] -> (s0 + 15)>()[%23]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %25 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %26 = arith.addi %25, %c31_i32 : i32
      %27 = arith.divsi %26, %c32_i32 : i32
      %28 = arith.muli %27, %c32_i32 : i32
      %29 = arith.addi %arg11, %28 : i32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %21]
      memref.store %29, %22[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%23], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%22 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %21]
      scf.yield %29 : i32
    } {pipeline.veconly}
    memref.store %10, %5[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%5 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
