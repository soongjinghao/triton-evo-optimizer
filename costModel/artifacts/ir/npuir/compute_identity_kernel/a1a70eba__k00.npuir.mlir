// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c133120_i64 = arith.constant 133120 : i64
  %c100352_i64 = arith.constant 100352 : i64
  %c67584_i64 = arith.constant 67584 : i64
  %c166400_i64 = arith.constant 166400 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c166144_i64 = arith.constant 166144 : i64
  %c166016_i64 = arith.constant 166016 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c166272_i64 = arith.constant 166272 : i64
  %c165888_i64 = arith.constant 165888 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c512_i32 = arith.constant 512 : i32
  %c256_i32 = arith.constant 256 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c165888_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<32xf32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg9 : i32 to index
  %7 = arith.index_cast %arg8 : i32 to index
  %8 = arith.remsi %arg8, %c512_i32 : i32
  %9 = arith.cmpi ne, %8, %c0_i32 : i32
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %10 = hivm.hir.pointer_cast(%c133120_i64, %c0_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c133120_i64, %c0_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %12 = arith.divsi %arg13, %4 : i32
    %13 = arith.remsi %12, %arg10 : i32
    %14 = arith.muli %13, %c32_i32 : i32
    %15 = arith.subi %arg7, %14 : i32
    %16 = arith.muli %14, %arg9 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.index_cast %15 : i32 to index
    %19 = affine.max affine_map<()[s0] -> (0, s0)>()[%18]
    %20 = affine.min affine_map<()[s0] -> (32, s0)>()[%19]
    %21 = affine.max affine_map<()[s0] -> (0, s0)>()[%20]
    %22 = arith.cmpi slt, %21, %c32 : index
    %23 = hivm.hir.pointer_cast(%c166272_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%23 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %24 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %23) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %27 = arith.index_cast %arg14 : i32 to index
      %28 = arith.index_cast %c0_i32 : i32 to index
      %29 = arith.index_cast %arg3 : i32 to index
      %30 = arith.index_cast %c8_i32 : i32 to index
      %31 = arith.index_cast %arg13 : i32 to index
      %32 = arith.index_cast %3 : i32 to index
      %33 = arith.index_cast %1 : i32 to index
      %34 = arith.index_cast %c48_i32 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%27, %28, %29, %30, %31, %32, %34]
      %36 = arith.index_cast %35 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %37 = arith.select %36, %c2_i64, %c3_i64 : i64
      %38 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %39 = arith.subi %arg3, %arg14 : i32
      %40 = arith.index_cast %arg14 : i32 to index
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%17, %40]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%41], sizes: [32, 8], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %42 = arith.index_cast %39 : i32 to index
      %43 = affine.max affine_map<()[s0] -> (0, s0)>()[%42]
      %44 = affine.min affine_map<()[s0] -> (8, s0)>()[%43]
      %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
      %46 = arith.cmpi slt, %45, %c8 : index
      %47 = arith.ori %22, %46 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%21, %45] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %38[0, 0] [%21, %45] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %47 {
        %collapse_shape_3 = memref.collapse_shape %38 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %37]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %48 = hivm.hir.pointer_cast(%c166016_i64) : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vreduce <sum> ins(%38 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%48 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %37]
      %collapse_shape = memref.collapse_shape %48 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_1 = memref.subview %collapse_shape[0] [%20] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c166144_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%49 : memref<32xf32, #hivm.address_space<ub>>)
      %subview_2 = memref.subview %49[0] [%20] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %50 = hivm.hir.pointer_cast(%c166272_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg15, %49 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%50 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %50 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %25 = arith.muli %14, %arg8 : i32
    %expand_shape = memref.expand_shape %24 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %26 = arith.index_cast %25 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    scf.for %arg14 = %c0_i32 to %arg8 step %c512_i32  : i32 {
      %27 = arith.index_cast %arg14 : i32 to index
      %28 = arith.index_cast %c0_i32 : i32 to index
      %29 = arith.index_cast %arg8 : i32 to index
      %30 = arith.index_cast %c512_i32 : i32 to index
      %31 = arith.index_cast %arg13 : i32 to index
      %32 = arith.index_cast %3 : i32 to index
      %33 = arith.index_cast %1 : i32 to index
      %34 = arith.index_cast %c48_i32 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%27, %28, %29, %30, %31, %32, %34]
      %36 = arith.index_cast %35 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %37 = arith.select %36, %c0_i64_0, %c1_i64 : i64
      %38 = arith.index_cast %35 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %39 = arith.select %38, %c2_i64, %c3_i64 : i64
      %40 = hivm.hir.pointer_cast(%c67584_i64, %c100352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c67584_i64, %c100352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %42 = hivm.hir.pointer_cast(%c2048_i64, %c34816_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %43 = hivm.hir.pointer_cast(%c2048_i64, %c34816_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %43 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %44 = arith.subi %arg8, %arg14 : i32
      %45 = arith.index_cast %arg14 : i32 to index
      %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%26, %45]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = arith.index_cast %44 : i32 to index
      %48 = affine.max affine_map<()[s0] -> (0, s0)>()[%47]
      %49 = affine.min affine_map<()[s0] -> (256, s0)>()[%48]
      %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
      %51 = arith.cmpi slt, %50, %c256 : index
      %52 = arith.ori %22, %51 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%21, %50] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %43[0, 0] [%21, %50] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %52 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %43 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %39]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c166400_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%43, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%42 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %42[0, 0] [%21, %50] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%21, %50] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %39]
      %54 = arith.addi %arg14, %c256_i32 : i32
      %55 = arith.addi %arg14, %c512_i32 : i32
      %56 = arith.cmpi sle, %55, %arg8 : i32
      scf.if %56 {
        %57 = arith.subi %arg8, %54 : i32
        %58 = arith.index_cast %54 : i32 to index
        %59 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%26, %58]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %60 = arith.index_cast %57 : i32 to index
        %61 = affine.max affine_map<()[s0] -> (0, s0)>()[%60]
        %62 = affine.min affine_map<()[s0] -> (256, s0)>()[%61]
        %63 = affine.max affine_map<()[s0] -> (0, s0)>()[%62]
        %64 = arith.cmpi slt, %63, %c256 : index
        %65 = arith.ori %22, %64 : i1
        %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%21, %63] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %41[0, 0] [%21, %63] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %65 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %41 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %37]
        hivm.hir.load ins(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %66 = hivm.hir.pointer_cast(%c166400_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%41, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%40 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%66 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%59], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %40[0, 0] [%21, %63] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_10 = memref.subview %reinterpret_cast_8[0, 0] [%21, %63] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_9 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %37]
      } {pipeline.veconly}
    } {pipeline.veconly}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.if %9 {
      %27 = arith.divsi %arg8, %c512_i32 : i32
      %28 = arith.muli %27, %c512_i32 : i32
      %29 = arith.subi %arg8, %28 : i32
      %30 = arith.index_cast %28 : i32 to index
      %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%26, %30]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %32 = arith.index_cast %29 : i32 to index
      %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%32]
      %34 = affine.min affine_map<()[s0] -> (256, s0)>()[%33]
      %35 = affine.max affine_map<()[s0] -> (0, s0)>()[%34]
      %36 = arith.cmpi slt, %35, %c256 : index
      %37 = arith.ori %22, %36 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%21, %35] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %11[0, 0] [%21, %35] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %37 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %11 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %38 = hivm.hir.pointer_cast(%c166400_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%11, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%10 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%38 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%31], sizes: [32, 256], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %10[0, 0] [%21, %35] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%21, %35] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
