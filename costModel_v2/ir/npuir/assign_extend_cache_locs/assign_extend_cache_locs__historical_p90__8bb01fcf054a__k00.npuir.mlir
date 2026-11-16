// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @assign_extend_cache_locs(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c736_i64 = arith.constant 736 : i64
  %c160_i64 = arith.constant 160 : i64
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c704_i64 = arith.constant 704 : i64
  %c32_i64 = arith.constant 32 : i64
  %c672_i64 = arith.constant 672 : i64
  %c0_i64 = arith.constant 0 : i64
  %c128_i32 = arith.constant 128 : i32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_1, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c160_i64, %c736_i64) : memref<128xi32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<128xi32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c32_i64, %c704_i64) : memref<8xi32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<8xi32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c0_i64, %c672_i64) : memref<8xi32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg11, %4 : i32
    %16 = arith.remsi %15, %arg8 : i32
    %17 = arith.index_cast %16 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = arith.subi %19, %18 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %21 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = arith.muli %21, %c128_i32 : i32
    %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%17]
    %24 = affine.min affine_map<()[s0] -> (8, s0)>()[%23]
    %25 = arith.cmpi slt, %24, %c8 : index
    %subview = memref.subview %reinterpret_cast[0] [%24] [1] : memref<8xi32, strided<[1]>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %14[0] [%24] [1] : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%14 : memref<8xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %subview_6 = memref.subview %reinterpret_cast_0[0] [%24] [1] : memref<8xi32, strided<[1]>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %13[0] [%24] [1] : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%13 : memref<8xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_6 : memref<?xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsub ins(%13, %14 : memref<8xi32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>) outs(%13 : memref<8xi32, #hivm.address_space<ub>>)
    %26 = hivm.hir.pointer_cast(%c64_i64) : memref<1xi32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c96_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%13 : memref<8xi32, #hivm.address_space<ub>>) outs(%26 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%27 : memref<16xi32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %28 = memref.load %26[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %29 = arith.index_cast %22 : i32 to index
    %30 = arith.index_cast %18 : i32 to index
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [128], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %32 = arith.index_cast %20 : i32 to index
    %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%32]
    %34 = affine.min affine_map<()[s0] -> (128, s0)>()[%33]
    %35 = arith.cmpi slt, %34, %c128 : index
    %subview_9 = memref.subview %reinterpret_cast_8[0] [%34] [1] : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %12[0] [%34] [1] : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %35 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%12 : memref<128xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%subview_9 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %36 = arith.index_cast %28 : i32 to index
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%36], sizes: [128], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %reinterpret_cast_11[0] [%34] [1] : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_10 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
