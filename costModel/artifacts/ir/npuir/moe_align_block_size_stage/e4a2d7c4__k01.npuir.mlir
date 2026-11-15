// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage4(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c160_i64 = arith.constant 160 : i64
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c32_i64 = arith.constant 32 : i64
  %c96_i64 = arith.constant 96 : i64
  %c0_i64 = arith.constant 0 : i64
  %c31_i32 = arith.constant 31 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg13 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c0_i64, %c96_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %13 = arith.divsi %arg13, %arg12 : i32
    %14 = arith.remsi %13, %arg11 : i32
    %15 = arith.divsi %arg13, %4 : i32
    %16 = arith.remsi %15, %arg10 : i32
    %17 = arith.index_cast %16 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%17]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%19], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %21 = arith.subi %20, %18 : i32
    %22 = arith.addi %21, %c31_i32 : i32
    %23 = arith.divsi %22, %c32_i32 : i32
    %24 = arith.cmpi slt, %14, %23 : i32
    scf.if %24 {
      %30 = arith.muli %14, %c32_i32 : i32
      %31 = arith.addi %18, %30 : i32
      %32 = arith.divsi %31, %c32_i32 : i32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %11]
      memref.store %16, %12[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %33 = arith.index_cast %32 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%12 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %11]
    } {pipeline.veconly}
    %25 = arith.muli %16, %arg9 : i32
    %26 = arith.muli %16, %c4_i32 : i32
    %27 = arith.addi %25, %arg9 : i32
    %28 = arith.minsi %27, %arg8 : i32
    %29 = arith.index_cast %26 : i32 to index
    scf.for %arg14 = %25 to %28 step %c1_i32  : i32 {
      %30 = hivm.hir.pointer_cast(%c64_i64, %c160_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %31 = hivm.hir.pointer_cast(%c32_i64, %c128_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %32 = arith.index_cast %arg14 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%32], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %33 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = arith.index_cast %33 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %34]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%35], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      %36 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%34], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %37 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %38 = arith.addi %36, %37 : i32
      %39 = arith.index_cast %38 : i32 to index
      memref.store %arg14, %31[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%31 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      %40 = arith.addi %36, %c1_i32 : i32
      memref.store %40, %30[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%30 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
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
