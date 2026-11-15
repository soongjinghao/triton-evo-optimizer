// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_set_k_and_s_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c2208_i64 = arith.constant 2208 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c2176_i64 = arith.constant 2176 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c32_i64 = arith.constant 32 : i64
  %c1120_i64 = arith.constant 1120 : i64
  %c4_i32 = arith.constant 4 : i32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64_i64 = arith.constant 64 : i64
  %c8448_i64 = arith.constant 8448 : i64
  %c128_i64 = arith.constant 128 : i64
  %c4_i64 = arith.constant 4 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = arith.muli %arg9, %c4_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = arith.index_cast %arg8 : i32 to index
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg12 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_1, %c1_i64 : i64
    %15 = hivm.hir.pointer_cast(%c1056_i64, %c2176_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c32_i64, %c1152_i64) : memref<4x128xf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x128xf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c0_i64, %c1120_i64) : memref<4xi64, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<4xi64, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg12, %4 : i32
    %19 = arith.remsi %18, %arg9 : i32
    %20 = arith.muli %19, %c4_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [4], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%21, %6]
    %23 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%22, %21]
    %24 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%23, %21]
    %25 = arith.cmpi slt, %24, %c4 : index
    %subview = memref.subview %reinterpret_cast[0] [%24] [1] : memref<4xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %17[0] [%24] [1] : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i64 : i64) outs(%17 : memref<4xi64, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %14]
    hivm.hir.load ins(%subview : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i64 : i64 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%21, %7]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%26], sizes: [4, 128], strides: [%7, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<4x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%24, 128] [1, 1] : memref<4x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %16[0, 0] [%24, 128] [1, 1] : memref<4x128xf16, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<4x128xf16, #hivm.address_space<ub>> into memref<512xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<512xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
    hivm.hir.load ins(%subview_4 : memref<?x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%21], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%24] [1] : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %15[0] [%24] [1] : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%15 : memref<4xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %27 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%23, %21]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.for %arg13 = %c0 to %27 step %c1 {
      %28 = memref.load %17[%arg13] {DiscreteMemAccess} : memref<4xi64, #hivm.address_space<ub>>
      %29 = arith.divsi %28, %c64_i64 : i64
      %30 = arith.muli %29, %c8448_i64 : i64
      %31 = arith.remsi %28, %c64_i64 : i64
      %32 = arith.muli %31, %c128_i64 : i64
      %33 = arith.addi %30, %32 : i64
      %34 = arith.index_cast %33 : i64 to index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [1, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %16[%arg13, 0] [1, 128] [1, 1] {DiscreteMemAccess} : memref<4x128xf16, #hivm.address_space<ub>> to memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %35 = arith.index_cast %arg13 : index to i32
      %36 = arith.addi %20, %35 : i32
      %37 = arith.cmpi slt, %36, %5 : i32
      %38 = arith.index_castui %37 : i1 to index
      %39 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%38]
      %subview_11 = memref.subview %subview_10[0, 0] [%38, %39] [1, 1] : memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_12 = memref.subview %reinterpret_cast_9[0, 0] [%38, %39] [1, 1] : memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_11 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    } {ExtractedLoadOrStore, pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg13 = %c0 to %27 step %c1 {
      %28 = arith.index_cast %arg12 : i32 to index
      %29 = arith.index_cast %3 : i32 to index
      %30 = arith.index_cast %1 : i32 to index
      %31 = arith.index_cast %c48_i32 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> ((s0 + ((s2 - s3) floordiv s4) * s1) mod 2)>()[%arg13, %27, %28, %29, %31]
      %33 = arith.index_cast %32 : index to i1
      %c0_i64_9 = arith.constant 0 : i64
      %c1_i64_10 = arith.constant 1 : i64
      %34 = arith.select %33, %c0_i64_9, %c1_i64_10 : i64
      %35 = hivm.hir.pointer_cast(%c1088_i64, %c2208_i64) : memref<1xf32, #hivm.address_space<ub>>
      annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
      %36 = memref.load %17[%arg13] {DiscreteMemAccess} : memref<4xi64, #hivm.address_space<ub>>
      %37 = arith.divsi %36, %c64_i64 : i64
      %38 = arith.muli %37, %c8448_i64 : i64
      %39 = arith.divsi %38, %c4_i64 : i64
      %40 = arith.addi %39, %c2048_i64 : i64
      %41 = arith.remsi %36, %c64_i64 : i64
      %42 = arith.addi %40, %41 : i64
      %43 = memref.load %15[%arg13] {DiscreteMemAccess} : memref<4xf32, #hivm.address_space<ub>>
      %44 = arith.index_cast %arg13 : index to i32
      %45 = arith.addi %20, %44 : i32
      %46 = arith.cmpi slt, %45, %5 : i32
      scf.if %46 {
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %34]
        memref.store %43, %35[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %47 = arith.index_cast %42 : i64 to index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%35 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %34]
      } {pipeline.veconly}
    } {ExtractedLoadOrStore, pipeline.veconly}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %14]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
