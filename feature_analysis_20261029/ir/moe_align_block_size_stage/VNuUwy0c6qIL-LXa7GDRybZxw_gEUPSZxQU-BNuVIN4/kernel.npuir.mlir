// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage4(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c96_i64 = arith.constant 96 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c32_i32 = arith.constant 32 : i32
  %c4_i32 = arith.constant 4 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg10 : i32
  %7 = arith.index_cast %6 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%7], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %8 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %9 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%7]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %10 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %11 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi32, #hivm.address_space<ub>>
  memref.store %6, %11[%c0] : memref<1xi32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  scf.for %arg13 = %8 to %10 step %c32_i32  : i32 {
    %17 = arith.divsi %arg13, %c32_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%11 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  }
  %12 = arith.muli %6, %arg9 : i32
  %13 = arith.muli %6, %c4_i32 : i32
  %14 = arith.addi %12, %arg9 : i32
  %15 = arith.minsi %14, %arg8 : i32
  %16 = arith.index_cast %13 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg13 = %12 to %15 step %c1_i32  : i32 {
    %17 = hivm.hir.pointer_cast(%c64_i64, %c128_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c32_i64, %c96_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    %19 = arith.index_cast %arg13 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %21 = arith.index_cast %20 : i32 to index
    %22 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%16, %21]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    %23 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%21], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = arith.addi %23, %24 : i32
    %26 = arith.index_cast %25 : i32 to index
    memref.store %arg13, %18[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%18 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %27 = arith.addi %23, %c1_i32 : i32
    memref.store %27, %17[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%17 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
