// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @rwkv_mm_sparsity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c37888_i64 = arith.constant 37888 : i64
  %c37632_i64 = arith.constant 37632 : i64
  %c33536_i64 = arith.constant 33536 : i64
  %c71424_i64 = arith.constant 71424 : i64
  %c768_i64 = arith.constant 768 : i64
  %c70912_i64 = arith.constant 70912 : i64
  %c256_i64 = arith.constant 256 : i64
  %c104192_i64 = arith.constant 104192 : i64
  %c70656_i64 = arith.constant 70656 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg9 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c70656_i64, %c104192_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg9, %4 : i32
    %15 = arith.remsi %14, %arg6 : i32
    %16 = arith.muli %15, %c64_i32 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = affine.max affine_map<()[s0] -> (512, s0)>()[%17]
    %19 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%18, %17]
    %20 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%19, %17]
    %21 = affine.max affine_map<()[s0] -> (0, s0)>()[%20]
    %22 = arith.cmpi slt, %21, %c64 : index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %12]
    hivm.hir.copy ins(%5 : memref<64xf32, #hivm.address_space<ub>>) outs(%13 : memref<64xf32, #hivm.address_space<ub>>)
    %23 = scf.for %arg10 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg11 = %13) -> (memref<64xf32, #hivm.address_space<ub>>)  : i32 {
      %25 = arith.index_cast %arg10 : i32 to index
      %26 = arith.index_cast %c0_i32 : i32 to index
      %27 = arith.index_cast %c8_i32 : i32 to index
      %28 = arith.index_cast %c1_i32 : i32 to index
      %29 = arith.index_cast %arg9 : i32 to index
      %30 = arith.index_cast %3 : i32 to index
      %31 = arith.index_cast %1 : i32 to index
      %32 = arith.index_cast %c48_i32 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%25, %26, %27, %28, %29, %30, %32]
      %34 = arith.index_cast %33 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %35 = arith.select %34, %c3_i64, %c4_i64 : i64
      %36 = arith.index_cast %33 : index to i1
      %c1_i64_2 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %37 = arith.select %36, %c1_i64_2, %c2_i64 : i64
      %38 = hivm.hir.pointer_cast(%c768_i64, %c71424_i64) : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      %39 = hivm.hir.pointer_cast(%c256_i64, %c70912_i64) : memref<128x1xf32, #hivm.address_space<ub>>
      annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<128x1xf32, #hivm.address_space<ub>>
      %40 = arith.muli %arg10, %c128_i32 : i32
      %41 = arith.index_cast %40 : i32 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [128, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %42 = affine.max affine_map<()[s0] -> (1024, s0)>()[%41]
      %43 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%42, %41]
      %44 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%43, %41]
      %45 = arith.cmpi slt, %44, %c128 : index
      %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%44, 1] [1, 1] : memref<128x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %39[0, 0] [%44, 1] [1, 1] : memref<128x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %45 {
        %collapse_shape_11 = memref.collapse_shape %39 [[0, 1]] : memref<128x1xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_11 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_4 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %37]
      hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%17, %41]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [128, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%43, %41]
      %48 = affine.max affine_map<()[s0] -> (0, s0)>()[%47]
      %49 = arith.cmpi slt, %48, %c128 : index
      %50 = arith.ori %49, %22 : i1
      %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%48, %21] [1, 1] : memref<128x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %38[0, 0] [%48, %21] [1, 1] : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %50 {
        %collapse_shape_11 = memref.collapse_shape %38 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_11 : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %35]
      hivm.hir.load ins(%subview_8 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %51 = hivm.hir.pointer_cast(%c33536_i64) : memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%39, %38 : memref<128x1xf32, #hivm.address_space<ub>>, memref<128x64xf32, #hivm.address_space<ub>>) outs(%38 : memref<128x64xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<1024xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %37]
      %52 = hivm.hir.pointer_cast(%c37632_i64) : memref<1x64xf32, #hivm.address_space<ub>>
      %53 = hivm.hir.pointer_cast(%c37888_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%38 : memref<128x64xf32, #hivm.address_space<ub>>) outs(%52 : memref<1x64xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %35]
      %collapse_shape_10 = memref.collapse_shape %52 [[0, 1]] : memref<1x64xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg11, %collapse_shape_10 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%13 : memref<64xf32, #hivm.address_space<ub>>)
      scf.yield %13 : memref<64xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%17], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%19, %17]
    %subview = memref.subview %23[0] [%24] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %reinterpret_cast[0] [%24] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %12]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
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
