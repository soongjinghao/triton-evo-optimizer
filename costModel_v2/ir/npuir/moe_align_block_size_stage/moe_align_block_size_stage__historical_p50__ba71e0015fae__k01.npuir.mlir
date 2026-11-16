// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c224_i64 = arith.constant 224 : i64
  %c96_i64 = arith.constant 96 : i64
  %c192_i64 = arith.constant 192 : i64
  %c64_i64 = arith.constant 64 : i64
  %c160_i64 = arith.constant 160 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg10, %4 : i32
    %6 = arith.remsi %5, %arg7 : i32
    %7 = arith.muli %6, %arg6 : i32
    %8 = arith.addi %6, %c1_i32 : i32
    %9 = arith.muli %8, %c4_i32 : i32
    scf.for %arg11 = %c0_i32 to %arg6 step %c4_i32  : i32 {
      %10 = hivm.hir.pointer_cast(%c96_i64, %c224_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %11 = hivm.hir.pointer_cast(%c64_i64, %c192_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %12 = hivm.hir.pointer_cast(%c32_i64, %c160_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %13 = hivm.hir.pointer_cast(%c0_i64, %c128_i64) : memref<1xi32, #hivm.address_space<ub>>
      annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
      %14 = arith.addi %arg11, %c1_i32 : i32
      %15 = arith.addi %arg11, %c2_i32 : i32
      %16 = arith.addi %arg11, %c3_i32 : i32
      %17 = arith.addi %7, %arg11 : i32
      %18 = arith.cmpi slt, %17, %arg5 : i32
      %19 = arith.addi %7, %14 : i32
      %20 = arith.cmpi slt, %19, %arg5 : i32
      %21 = arith.addi %7, %15 : i32
      %22 = arith.cmpi slt, %21, %arg5 : i32
      %23 = arith.addi %7, %16 : i32
      %24 = arith.cmpi slt, %23, %arg5 : i32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      scf.if %18 {
        %25 = arith.index_cast %7 : i32 to index
        %26 = arith.index_cast %arg11 : i32 to index
        %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %28 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %29 = arith.index_cast %9 : i32 to index
        %30 = arith.index_cast %28 : i32 to index
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = arith.addi %32, %c1_i32 : i32
        memref.store %33, %13[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%13 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.if %20 {
        %25 = arith.index_cast %7 : i32 to index
        %26 = arith.index_cast %14 : i32 to index
        %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %28 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %29 = arith.index_cast %9 : i32 to index
        %30 = arith.index_cast %28 : i32 to index
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = arith.addi %32, %c1_i32 : i32
        memref.store %33, %12[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%12 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.if %22 {
        %25 = arith.index_cast %7 : i32 to index
        %26 = arith.index_cast %15 : i32 to index
        %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %28 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %29 = arith.index_cast %9 : i32 to index
        %30 = arith.index_cast %28 : i32 to index
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = arith.addi %32, %c1_i32 : i32
        memref.store %33, %11[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%11 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.if %24 {
        %25 = arith.index_cast %7 : i32 to index
        %26 = arith.index_cast %16 : i32 to index
        %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %28 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %29 = arith.index_cast %9 : i32 to index
        %30 = arith.index_cast %28 : i32 to index
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = arith.addi %32, %c1_i32 : i32
        memref.store %33, %10[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%10 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
