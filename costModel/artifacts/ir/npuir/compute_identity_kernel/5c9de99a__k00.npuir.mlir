// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c35680_i64 = arith.constant 35680 : i64
  %c37728_i64 = arith.constant 37728 : i64
  %c2912_i64 = arith.constant 2912 : i64
  %c2656_i64 = arith.constant 2656 : i64
  %c1632_i64 = arith.constant 1632 : i64
  %c36704_i64 = arith.constant 36704 : i64
  %c608_i64 = arith.constant 608 : i64
  %c2784_i64 = arith.constant 2784 : i64
  %c512_i64 = arith.constant 512 : i64
  %c384_i64 = arith.constant 384 : i64
  %c256_i64 = arith.constant 256 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<32xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c128_i64) : memref<32xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : memref<32xi32, #hivm.address_space<ub>>)
  %7 = arith.index_cast %arg9 : i32 to index
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
    %13 = hivm.hir.pointer_cast(%c256_i64) : memref<32xi32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%6, %12 : memref<32xi32, #hivm.address_space<ub>>, i32) outs(%13 : memref<32xi32, #hivm.address_space<ub>>)
    %14 = hivm.hir.pointer_cast(%c256_i64) : memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%13, %6 : memref<32xi32, #hivm.address_space<ub>>, memref<32xi32, #hivm.address_space<ub>>) outs(%14 : memref<32xi1, #hivm.address_space<ub>>)
    %15 = hivm.hir.pointer_cast(%c256_i64) : memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%14 : memref<32xi1, #hivm.address_space<ub>>) outs(%15 : memref<32xi1, #hivm.address_space<ub>>)
    %16 = arith.muli %11, %arg9 : i32
    %17 = hivm.hir.pointer_cast(%c384_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %17 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c512_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%15 : memref<32xi1, #hivm.address_space<ub>>) outs(%collapse_shape : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%18 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
    %19 = arith.index_cast %16 : i32 to index
    %20 = arith.index_cast %12 : i32 to index
    %21 = affine.max affine_map<()[s0] -> (0, s0)>()[%20]
    %22 = affine.min affine_map<()[s0] -> (32, s0)>()[%21]
    %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
    %24 = arith.cmpi slt, %23, %c32 : index
    %25 = hivm.hir.pointer_cast(%c2784_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%25 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    %26 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %25) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %29 = arith.index_cast %arg14 : i32 to index
      %30 = arith.index_cast %c0_i32 : i32 to index
      %31 = arith.index_cast %arg3 : i32 to index
      %32 = arith.index_cast %c8_i32 : i32 to index
      %33 = arith.index_cast %arg13 : i32 to index
      %34 = arith.index_cast %3 : i32 to index
      %35 = arith.index_cast %1 : i32 to index
      %36 = arith.index_cast %c48_i32 : i32 to index
      %37 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%29, %30, %31, %32, %33, %34, %36]
      %38 = arith.index_cast %37 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %39 = arith.select %38, %c1_i64, %c2_i64 : i64
      %40 = hivm.hir.pointer_cast(%c608_i64, %c36704_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %41 = arith.subi %arg3, %arg14 : i32
      %42 = arith.index_cast %arg14 : i32 to index
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %42]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%43], sizes: [32, 8], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %44 = arith.index_cast %41 : i32 to index
      %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
      %46 = affine.min affine_map<()[s0] -> (8, s0)>()[%45]
      %47 = affine.max affine_map<()[s0] -> (0, s0)>()[%46]
      %48 = arith.cmpi slt, %47, %c8 : index
      %49 = arith.ori %24, %48 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%23, %47] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %40[0, 0] [%23, %47] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %49 {
        %collapse_shape_2 = memref.collapse_shape %40 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_2 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %39]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %50 = hivm.hir.pointer_cast(%c1632_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%40, %17 : memref<32x8xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%40 : memref<32x8xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
      %51 = hivm.hir.pointer_cast(%c2656_i64) : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%40 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%51 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %39]
      %collapse_shape_1 = memref.collapse_shape %51 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %52 = hivm.hir.pointer_cast(%c2784_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg15, %collapse_shape_1 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%52 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %52 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %27 = arith.muli %11, %arg8 : i32
    %expand_shape = memref.expand_shape %26 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %28 = arith.index_cast %27 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
      %29 = arith.index_cast %arg14 : i32 to index
      %30 = arith.index_cast %c0_i32 : i32 to index
      %31 = arith.index_cast %arg8 : i32 to index
      %32 = arith.index_cast %c256_i32 : i32 to index
      %33 = arith.index_cast %arg13 : i32 to index
      %34 = arith.index_cast %3 : i32 to index
      %35 = arith.index_cast %1 : i32 to index
      %36 = arith.index_cast %c48_i32 : i32 to index
      %37 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%29, %30, %31, %32, %33, %34, %36]
      %38 = arith.index_cast %37 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %39 = arith.select %38, %c0_i64_0, %c1_i64 : i64
      %40 = hivm.hir.pointer_cast(%c2912_i64, %c37728_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c2912_i64, %c37728_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %42 = arith.subi %arg8, %arg14 : i32
      %43 = arith.index_cast %arg14 : i32 to index
      %44 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %43]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [32, 256], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.index_cast %42 : i32 to index
      %46 = affine.max affine_map<()[s0] -> (0, s0)>()[%45]
      %47 = affine.min affine_map<()[s0] -> (256, s0)>()[%46]
      %48 = affine.max affine_map<()[s0] -> (0, s0)>()[%47]
      %49 = arith.cmpi slt, %48, %c256 : index
      %50 = arith.ori %24, %49 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%23, %48] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %41[0, 0] [%23, %48] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %50 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_5 = memref.collapse_shape %41 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_5 : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %39]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %51 = hivm.hir.pointer_cast(%c35680_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%41, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%40 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%44], sizes: [32, 256], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %40[0, 0] [%23, %48] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%23, %48] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %39]
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
