// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c34304_i64 = arith.constant 34304 : i64
  %c36352_i64 = arith.constant 36352 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c35328_i64 = arith.constant 35328 : i64
  %c128_i64 = arith.constant 128 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c256_i32 = arith.constant 256 : i32
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<32xf32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg9 : i32 to index
  %7 = arith.index_cast %arg8 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %8 = arith.divsi %arg13, %4 : i32
    %9 = arith.remsi %8, %arg10 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.subi %arg7, %10 : i32
    %12 = arith.muli %10, %arg9 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.index_cast %11 : i32 to index
    %15 = affine.max affine_map<()[s0] -> (0, s0)>()[%14]
    %16 = affine.min affine_map<()[s0] -> (32, s0)>()[%15]
    %17 = affine.max affine_map<()[s0] -> (0, s0)>()[%16]
    %18 = arith.cmpi slt, %17, %c32 : index
    %19 = hivm.hir.pointer_cast(%c1408_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%19 : memref<32xf32, #hivm.address_space<ub>>)
    %20 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %19) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %23 = arith.index_cast %arg14 : i32 to index
      %24 = arith.index_cast %c0_i32 : i32 to index
      %25 = arith.index_cast %arg3 : i32 to index
      %26 = arith.index_cast %c8_i32 : i32 to index
      %27 = arith.index_cast %arg13 : i32 to index
      %28 = arith.index_cast %3 : i32 to index
      %29 = arith.index_cast %1 : i32 to index
      %30 = arith.index_cast %c48_i32 : i32 to index
      %31 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%23, %24, %25, %26, %27, %28, %30]
      %32 = arith.index_cast %31 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %33 = arith.select %32, %c1_i64, %c2_i64 : i64
      %34 = hivm.hir.pointer_cast(%c128_i64, %c35328_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %35 = arith.subi %arg3, %arg14 : i32
      %36 = arith.index_cast %arg14 : i32 to index
      %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %36]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [32, 8], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %38 = arith.index_cast %35 : i32 to index
      %39 = affine.max affine_map<()[s0] -> (0, s0)>()[%38]
      %40 = affine.min affine_map<()[s0] -> (8, s0)>()[%39]
      %41 = affine.max affine_map<()[s0] -> (0, s0)>()[%40]
      %42 = arith.cmpi slt, %41, %c8 : index
      %43 = arith.ori %18, %42 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%17, %41] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %34[0, 0] [%17, %41] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %43 {
        %collapse_shape_3 = memref.collapse_shape %34 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %33]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %44 = hivm.hir.pointer_cast(%c1152_i64) : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vreduce <sum> ins(%34 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%44 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %33]
      %collapse_shape = memref.collapse_shape %44 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_1 = memref.subview %collapse_shape[0] [%16] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %45 = hivm.hir.pointer_cast(%c1280_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%45 : memref<32xf32, #hivm.address_space<ub>>)
      %subview_2 = memref.subview %45[0] [%16] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %46 = hivm.hir.pointer_cast(%c1408_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg15, %45 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%46 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %46 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %21 = arith.muli %10, %arg8 : i32
    %expand_shape = memref.expand_shape %20 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %22 = arith.index_cast %21 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
      %23 = arith.index_cast %arg14 : i32 to index
      %24 = arith.index_cast %c0_i32 : i32 to index
      %25 = arith.index_cast %arg8 : i32 to index
      %26 = arith.index_cast %c256_i32 : i32 to index
      %27 = arith.index_cast %arg13 : i32 to index
      %28 = arith.index_cast %3 : i32 to index
      %29 = arith.index_cast %1 : i32 to index
      %30 = arith.index_cast %c48_i32 : i32 to index
      %31 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%23, %24, %25, %26, %27, %28, %30]
      %32 = arith.index_cast %31 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %33 = arith.select %32, %c0_i64_0, %c1_i64 : i64
      %34 = hivm.hir.pointer_cast(%c1536_i64, %c36352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %35 = hivm.hir.pointer_cast(%c1536_i64, %c36352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %36 = arith.subi %arg8, %arg14 : i32
      %37 = arith.index_cast %arg14 : i32 to index
      %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%22, %37]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %39 = arith.index_cast %36 : i32 to index
      %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
      %41 = affine.min affine_map<()[s0] -> (256, s0)>()[%40]
      %42 = affine.max affine_map<()[s0] -> (0, s0)>()[%41]
      %43 = arith.cmpi slt, %42, %c256 : index
      %44 = arith.ori %18, %43 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%17, %42] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %35[0, 0] [%17, %42] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %44 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %35 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %33]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %45 = hivm.hir.pointer_cast(%c34304_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%35, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%34 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%45 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %34[0, 0] [%17, %42] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%17, %42] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %33]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
