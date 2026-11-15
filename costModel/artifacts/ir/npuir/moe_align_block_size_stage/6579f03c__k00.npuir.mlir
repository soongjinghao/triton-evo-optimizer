// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage4(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c704_i64 = arith.constant 704 : i64
  %c672_i64 = arith.constant 672 : i64
  %c544_i64 = arith.constant 544 : i64
  %c416_i64 = arith.constant 416 : i64
  %c288_i64 = arith.constant 288 : i64
  %c160_i64 = arith.constant 160 : i64
  %c736_i64 = arith.constant 736 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c0_i32 : i32) outs(%5 : memref<32xi32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg13 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c128_i64, %c736_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg13, %4 : i32
    %15 = arith.remsi %14, %arg10 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %17 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%16]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    memref.store %15, %13[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.for %arg14 = %17 to %19 step %c32_i32  : i32 {
      %26 = arith.divsi %arg14, %c32_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%13 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    %20 = arith.muli %15, %arg9 : i32
    %21 = arith.muli %15, %c4_i32 : i32
    %22 = arith.addi %20, %arg9 : i32
    %23 = arith.minsi %22, %arg8 : i32
    %24 = arith.extsi %21 : i32 to i64
    %25 = scf.while (%arg14 = %20) : (i32) -> i32 {
      %26 = arith.cmpi slt, %arg14, %23 : i32
      scf.condition(%26) %arg14 : i32
    } do {
    ^bb0(%arg14: i32):
      %26 = arith.subi %23, %arg14 : i32
      %27 = arith.minsi %26, %c32_i32 : i32
      %28 = arith.index_cast %arg14 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [32], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<32xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %29 = hivm.hir.pointer_cast(%c160_i64) : memref<32xi32, #hivm.address_space<ub>>
      %30 = arith.index_cast %27 : i32 to index
      %31 = affine.max affine_map<()[s0] -> (0, s0)>()[%30]
      %32 = affine.min affine_map<()[s0] -> (32, s0)>()[%31]
      %33 = arith.cmpi slt, %32, %c32 : index
      %subview = memref.subview %reinterpret_cast_2[0] [%32] [1] : memref<32xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %29[0] [%32] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %33 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%c0_i32 : i32) outs(%29 : memref<32xi32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %34 = hivm.hir.pointer_cast(%c288_i64) : memref<32xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg15 = %c0 to %32 step %c1 {
        %38 = memref.load %29[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
        %39 = arith.extsi %38 : i32 to i64
        %40 = arith.addi %24, %39 : i64
        %41 = arith.index_cast %40 : i64 to index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %42 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %42, %34[%arg15] : memref<32xi32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %subview_4 = memref.subview %34[0] [%32] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %35 = hivm.hir.pointer_cast(%c416_i64) : memref<32xi32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%5 : memref<32xi32, #hivm.address_space<ub>>) outs(%35 : memref<32xi32, #hivm.address_space<ub>>)
      %subview_5 = memref.subview %35[0] [%32] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_4 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %36 = hivm.hir.pointer_cast(%c544_i64) : memref<32xi32, #hivm.address_space<ub>>
      scf.for %arg15 = %c0 to %c32 step %c1 {
        %38 = memref.load %29[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
        %39 = arith.index_cast %38 : i32 to index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %40 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %40, %36[%arg15] : memref<32xi32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      scf.for %arg15 = %c0 to %32 step %c1 {
        %38 = memref.load %35[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
        %39 = memref.load %36[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
        %40 = arith.addi %38, %39 : i32
        %41 = arith.index_cast %arg15 : index to i32
        %42 = arith.addi %arg14, %41 : i32
        %43 = arith.cmpi slt, %41, %27 : i32
        scf.if %43 {
          %44 = hivm.hir.pointer_cast(%c672_i64) : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
          memref.store %42, %44[%c0] : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %45 = arith.index_cast %40 : i32 to index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%44 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        } {pipeline.veconly}
      } {ExtractedLoadOrStore}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      scf.for %arg15 = %c0 to %32 step %c1 {
        %38 = memref.load %29[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
        %39 = arith.extsi %38 : i32 to i64
        %40 = arith.addi %24, %39 : i64
        %41 = memref.load %35[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
        %42 = arith.addi %41, %c1_i32 : i32
        %43 = arith.index_cast %arg15 : index to i32
        %44 = arith.cmpi slt, %43, %27 : i32
        scf.if %44 {
          %45 = hivm.hir.pointer_cast(%c704_i64) : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
          memref.store %42, %45[%c0] : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %46 = arith.index_cast %40 : i64 to index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%45 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        } {pipeline.veconly}
      } {ExtractedLoadOrStore}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
      %37 = arith.addi %arg14, %27 : i32
      scf.yield %37 : i32
    } attributes {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
