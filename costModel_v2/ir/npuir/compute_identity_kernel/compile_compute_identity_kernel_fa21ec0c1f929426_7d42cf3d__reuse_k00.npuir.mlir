// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c35552_i64 = arith.constant 35552 : i64
  %c36704_i64 = arith.constant 36704 : i64
  %c2784_i64 = arith.constant 2784 : i64
  %c2528_i64 = arith.constant 2528 : i64
  %c2400_i64 = arith.constant 2400 : i64
  %c1376_i64 = arith.constant 1376 : i64
  %c352_i64 = arith.constant 352 : i64
  %c320_i64 = arith.constant 320 : i64
  %c36576_i64 = arith.constant 36576 : i64
  %c192_i64 = arith.constant 192 : i64
  %c2656_i64 = arith.constant 2656 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c256_i32 = arith.constant 256 : i32
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
  %6 = hivm.hir.pointer_cast(%c128_i64) : memref<1x8xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %6 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<8xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c160_i64) : memref<1x8xf32, #hivm.address_space<ub>>
  %collapse_shape_1 = memref.collapse_shape %7 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_1 : memref<8xf32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg8 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %9 = arith.divsi %arg13, %4 : i32
    %10 = arith.remsi %9, %arg10 : i32
    %11 = arith.muli %10, %c32_i32 : i32
    %12 = arith.subi %arg7, %11 : i32
    %13 = arith.index_cast %11 : i32 to index
    %14 = arith.index_cast %12 : i32 to index
    %15 = affine.max affine_map<()[s0] -> (0, s0)>()[%14]
    %16 = affine.min affine_map<()[s0] -> (32, s0)>()[%15]
    %17 = arith.cmpi slt, %16, %c32 : index
    %18 = hivm.hir.pointer_cast(%c2656_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%18 : memref<32xf32, #hivm.address_space<ub>>)
    %19 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %18) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %24 = arith.index_cast %arg14 : i32 to index
      %25 = arith.index_cast %c0_i32 : i32 to index
      %26 = arith.index_cast %arg3 : i32 to index
      %27 = arith.index_cast %c8_i32 : i32 to index
      %28 = arith.index_cast %arg13 : i32 to index
      %29 = arith.index_cast %3 : i32 to index
      %30 = arith.index_cast %1 : i32 to index
      %31 = arith.index_cast %c48_i32 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%24, %25, %26, %27, %28, %29, %31]
      %33 = arith.index_cast %32 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %34 = arith.select %33, %c1_i64, %c2_i64 : i64
      %35 = hivm.hir.pointer_cast(%c192_i64, %c36576_i64) : memref<32x1xf32, #hivm.address_space<ub>>
      annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<32x1xf32, #hivm.address_space<ub>>
      %36 = arith.subi %arg3, %arg14 : i32
      %37 = arith.muli %arg14, %arg9 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%38, %13]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [32, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%16, 1] [1, 1] : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %35[0, 0] [%16, 1] [1, 1] : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %17 {
        %collapse_shape_13 = memref.collapse_shape %35 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_13 : memref<32xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_3 = memref.collapse_shape %subview [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_4 = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %34]
      hivm.hir.load ins(%collapse_shape_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %40 = arith.index_cast %36 : i32 to index
      %41 = affine.max affine_map<()[s0] -> (0, s0)>()[%40]
      %42 = affine.min affine_map<()[s0] -> (8, s0)>()[%41]
      %subview_5 = memref.subview %6[0, 0] [1, %42] [1, 1] : memref<1x8xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %43 = hivm.hir.pointer_cast(%c320_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %collapse_shape_6 = memref.collapse_shape %43 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%collapse_shape_1 : memref<8xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<8xf32, #hivm.address_space<ub>>)
      %subview_7 = memref.subview %43[0, 0] [1, %42] [1, 1] : memref<1x8xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %collapse_shape_8 = memref.collapse_shape %subview_5 [[0, 1]] : memref<1x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_9 = memref.collapse_shape %subview_7 [[0, 1]] : memref<1x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%collapse_shape_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c352_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      %45 = hivm.hir.pointer_cast(%c1376_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%35, %43 : memref<32x1xf32, #hivm.address_space<ub>>, memref<1x8xf32, #hivm.address_space<ub>>) outs(%44 : memref<32x8xf32, #hivm.address_space<ub>>) temp_buffer(%45 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [0, 1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %34]
      %46 = hivm.hir.pointer_cast(%c2400_i64) : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%44 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%46 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      %collapse_shape_10 = memref.collapse_shape %46 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_11 = memref.subview %collapse_shape_10[0] [%16] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %47 = hivm.hir.pointer_cast(%c2528_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%47 : memref<32xf32, #hivm.address_space<ub>>)
      %subview_12 = memref.subview %47[0] [%16] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %48 = hivm.hir.pointer_cast(%c2656_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg15, %47 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%48 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %48 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %20 = arith.muli %11, %arg8 : i32
    %expand_shape = memref.expand_shape %19 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %21 = arith.index_cast %20 : i32 to index
    %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%16]
    %23 = arith.cmpi slt, %22, %c32 : index
    hivm.hir.pipe_barrier[<PIPE_V>]
    scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
      %24 = arith.index_cast %arg14 : i32 to index
      %25 = arith.index_cast %c0_i32 : i32 to index
      %26 = arith.index_cast %arg8 : i32 to index
      %27 = arith.index_cast %c256_i32 : i32 to index
      %28 = arith.index_cast %arg13 : i32 to index
      %29 = arith.index_cast %3 : i32 to index
      %30 = arith.index_cast %1 : i32 to index
      %31 = arith.index_cast %c48_i32 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%24, %25, %26, %27, %28, %29, %31]
      %33 = arith.index_cast %32 : index to i1
      %c0_i64_2 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %34 = arith.select %33, %c0_i64_2, %c1_i64 : i64
      %35 = hivm.hir.pointer_cast(%c2784_i64, %c36704_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %36 = hivm.hir.pointer_cast(%c2784_i64, %c36704_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %37 = arith.subi %arg8, %arg14 : i32
      %38 = arith.index_cast %arg14 : i32 to index
      %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %38]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [32, 256], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %40 = arith.index_cast %37 : i32 to index
      %41 = affine.max affine_map<()[s0] -> (0, s0)>()[%40]
      %42 = affine.min affine_map<()[s0] -> (256, s0)>()[%41]
      %43 = affine.max affine_map<()[s0] -> (0, s0)>()[%42]
      %44 = arith.cmpi slt, %43, %c256 : index
      %45 = arith.ori %23, %44 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%22, %43] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %36[0, 0] [%22, %43] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %45 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_7 = memref.collapse_shape %36 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_7 : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %34]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %46 = hivm.hir.pointer_cast(%c35552_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%36, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%35 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%46 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [32, 256], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %35[0, 0] [%22, %43] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%22, %43] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_5 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %34]
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
