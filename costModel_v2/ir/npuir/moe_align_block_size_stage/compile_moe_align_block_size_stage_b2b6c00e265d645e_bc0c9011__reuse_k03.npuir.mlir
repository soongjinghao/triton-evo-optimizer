// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_token(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c608_i64 = arith.constant 608 : i64
  %c288_i64 = arith.constant 288 : i64
  %c576_i64 = arith.constant 576 : i64
  %c256_i64 = arith.constant 256 : i64
  %c320_i64 = arith.constant 320 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
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
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
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
    %12 = hivm.hir.pointer_cast(%c0_i64, %c320_i64) : memref<64xi32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<64xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %13 = arith.divsi %arg13, %4 : i32
    %14 = arith.remsi %13, %arg10 : i32
    %15 = arith.index_cast %14 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %17 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%15]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = arith.divsi %16, %c32_i32 : i32
    %20 = arith.divsi %18, %c32_i32 : i32
    %21 = arith.subi %20, %19 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %11]
    hivm.hir.vbrc ins(%14 : i32) outs(%12 : memref<64xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %22 = arith.index_cast %19 : i32 to index
    %23 = arith.index_cast %21 : i32 to index
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.for %arg14 = %c0_i32 to %21 step %c64_i32  : i32 {
      %29 = arith.index_cast %arg14 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%22, %29]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%30], sizes: [64], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %23]
      %32 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%31, %29]
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %29]
      %subview = memref.subview %12[0] [%33] [1] : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%33] [1] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %11]
    %24 = arith.muli %14, %arg9 : i32
    %25 = arith.muli %14, %c4_i32 : i32
    %26 = arith.addi %24, %arg9 : i32
    %27 = arith.minsi %26, %arg8 : i32
    %28 = arith.index_cast %25 : i32 to index
    scf.for %arg14 = %24 to %27 step %c1_i32  : i32 {
      %29 = arith.index_cast %arg14 : i32 to index
      %30 = arith.index_cast %24 : i32 to index
      %31 = arith.index_cast %27 : i32 to index
      %32 = arith.index_cast %c1_i32 : i32 to index
      %33 = arith.index_cast %arg13 : i32 to index
      %34 = arith.index_cast %3 : i32 to index
      %35 = arith.index_cast %1 : i32 to index
      %36 = arith.index_cast %c48_i32 : i32 to index
      %37 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%29, %30, %31, %32, %33, %34, %36]
      %38 = arith.index_cast %37 : index to i1
      %c1_i64_2 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %39 = arith.select %38, %c1_i64_2, %c2_i64 : i64
      %40 = hivm.hir.pointer_cast(%c288_i64, %c608_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c256_i64, %c576_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %42 = arith.index_cast %arg14 : i32 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %43 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %44 = arith.index_cast %43 : i32 to index
      %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %44]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%45], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %46 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %47 = arith.addi %46, %c1_i32 : i32
      memref.store %47, %41[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%41 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %48 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %49 = arith.addi %46, %48 : i32
      %50 = arith.index_cast %49 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %39]
      memref.store %arg14, %40[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%40 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %39]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
