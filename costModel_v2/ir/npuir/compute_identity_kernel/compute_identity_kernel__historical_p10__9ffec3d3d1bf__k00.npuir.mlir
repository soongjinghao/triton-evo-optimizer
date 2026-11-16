// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c68608_i64 = arith.constant 68608 : i64
  %c72704_i64 = arith.constant 72704 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c70656_i64 = arith.constant 70656 : i64
  %c256_i64 = arith.constant 256 : i64
  %c2816_i64 = arith.constant 2816 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c512_i32 = arith.constant 512 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128_i32 = arith.constant 128 : i32
  %c32768_i32 = arith.constant 32768 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c256_i32 = arith.constant 256 : i32
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg11, %4 : i32
    %7 = arith.remsi %6, %arg8 : i32
    %8 = arith.muli %7, %c64_i32 : i32
    %9 = arith.subi %arg7, %8 : i32
    %10 = arith.muli %7, %c128_i32 : i32
    %11 = arith.muli %7, %c32768_i32 : i32
    %12 = arith.index_cast %10 : i32 to index
    %13 = arith.index_cast %9 : i32 to index
    %14 = affine.max affine_map<()[s0] -> (0, s0)>()[%13]
    %15 = affine.min affine_map<()[s0] -> (64, s0)>()[%14]
    %16 = affine.max affine_map<()[s0] -> (0, s0)>()[%15]
    %17 = arith.cmpi slt, %16, %c64 : index
    %18 = hivm.hir.pointer_cast(%c2816_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%5 : memref<64xf32, #hivm.address_space<ub>>) outs(%18 : memref<64xf32, #hivm.address_space<ub>>)
    %19 = scf.for %arg12 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg13 = %18) -> (memref<64xf32, #hivm.address_space<ub>>)  : i32 {
      %21 = arith.index_cast %arg12 : i32 to index
      %22 = arith.index_cast %c0_i32 : i32 to index
      %23 = arith.index_cast %arg3 : i32 to index
      %24 = arith.index_cast %c8_i32 : i32 to index
      %25 = arith.index_cast %arg11 : i32 to index
      %26 = arith.index_cast %3 : i32 to index
      %27 = arith.index_cast %1 : i32 to index
      %28 = arith.index_cast %c48_i32 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%21, %22, %23, %24, %25, %26, %28]
      %30 = arith.index_cast %29 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %31 = arith.select %30, %c1_i64, %c2_i64 : i64
      %32 = hivm.hir.pointer_cast(%c256_i64, %c70656_i64) : memref<64x8xf32, #hivm.address_space<ub>>
      annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<64x8xf32, #hivm.address_space<ub>>
      %33 = arith.subi %arg3, %arg12 : i32
      %34 = arith.index_cast %arg12 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%12, %34]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%35], sizes: [64, 8], strides: [2, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x8xf32, strided<[2, 1], offset: ?>, #hivm.address_space<gm>>
      %36 = arith.index_cast %33 : i32 to index
      %37 = affine.max affine_map<()[s0] -> (0, s0)>()[%36]
      %38 = affine.min affine_map<()[s0] -> (8, s0)>()[%37]
      %39 = affine.max affine_map<()[s0] -> (0, s0)>()[%38]
      %40 = arith.cmpi slt, %39, %c8 : index
      %41 = arith.ori %17, %40 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%16, %39] [1, 1] : memref<64x8xf32, strided<[2, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[2, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %32[0, 0] [%16, %39] [1, 1] : memref<64x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %41 {
        %collapse_shape_3 = memref.collapse_shape %32 [[0, 1]] : memref<64x8xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_3 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %31]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[2, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %42 = hivm.hir.pointer_cast(%c2304_i64) : memref<64x1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vreduce <sum> ins(%32 : memref<64x8xf32, #hivm.address_space<ub>>) outs(%42 : memref<64x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %31]
      %collapse_shape = memref.collapse_shape %42 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      %subview_1 = memref.subview %collapse_shape[0] [%15] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %43 = hivm.hir.pointer_cast(%c2560_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%5 : memref<64xf32, #hivm.address_space<ub>>) outs(%43 : memref<64xf32, #hivm.address_space<ub>>)
      %subview_2 = memref.subview %43[0] [%15] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c2816_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg13, %43 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%44 : memref<64xf32, #hivm.address_space<ub>>)
      scf.yield %44 : memref<64xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %expand_shape = memref.expand_shape %19 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
    %20 = arith.index_cast %11 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    scf.for %arg12 = %c0_i32 to %c512_i32 step %c256_i32  : i32 {
      %21 = arith.index_cast %arg12 : i32 to index
      %22 = arith.index_cast %c0_i32 : i32 to index
      %23 = arith.index_cast %c512_i32 : i32 to index
      %24 = arith.index_cast %c256_i32 : i32 to index
      %25 = arith.index_cast %arg11 : i32 to index
      %26 = arith.index_cast %3 : i32 to index
      %27 = arith.index_cast %1 : i32 to index
      %28 = arith.index_cast %c48_i32 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%21, %22, %23, %24, %25, %26, %28]
      %30 = arith.index_cast %29 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %31 = arith.select %30, %c0_i64_0, %c1_i64 : i64
      %32 = hivm.hir.pointer_cast(%c3072_i64, %c72704_i64) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
      %33 = hivm.hir.pointer_cast(%c3072_i64, %c72704_i64) : memref<64x256xf32, #hivm.address_space<ub>>
      annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
      %34 = arith.subi %c512_i32, %arg12 : i32
      %35 = arith.index_cast %arg12 : i32 to index
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%20, %35]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [64, 256], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %37 = arith.index_cast %34 : i32 to index
      %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
      %39 = affine.min affine_map<()[s0] -> (256, s0)>()[%38]
      %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
      %41 = arith.cmpi slt, %40, %c256 : index
      %42 = arith.ori %17, %41 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%16, %40] [1, 1] : memref<64x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %33[0, 0] [%16, %40] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %42 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %33 [[0, 1]] : memref<64x256xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<16384xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %31]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %43 = hivm.hir.pointer_cast(%c68608_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%33, %expand_shape : memref<64x256xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%32 : memref<64x256xf32, #hivm.address_space<ub>>) temp_buffer(%43 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%36], sizes: [64, 256], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %32[0, 0] [%16, %40] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%16, %40] [1, 1] : memref<64x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %31]
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
