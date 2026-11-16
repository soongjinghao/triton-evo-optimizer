// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c21216_i64 = arith.constant 21216 : i64
  %c18976_i64 = arith.constant 18976 : i64
  %c17952_i64 = arith.constant 17952 : i64
  %c16928_i64 = arith.constant 16928 : i64
  %c16864_i64 = arith.constant 16864 : i64
  %c16352_i64 = arith.constant 16352 : i64
  %c15328_i64 = arith.constant 15328 : i64
  %c21152_i64 = arith.constant 21152 : i64
  %c15264_i64 = arith.constant 15264 : i64
  %c15232_i64 = arith.constant 15232 : i64
  %c13696_i64 = arith.constant 13696 : i64
  %c13600_i64 = arith.constant 13600 : i64
  %c13568_i64 = arith.constant 13568 : i64
  %c12544_i64 = arith.constant 12544 : i64
  %c11520_i64 = arith.constant 11520 : i64
  %c11456_i64 = arith.constant 11456 : i64
  %c11392_i64 = arith.constant 11392 : i64
  %c10368_i64 = arith.constant 10368 : i64
  %c10304_i64 = arith.constant 10304 : i64
  %c9792_i64 = arith.constant 9792 : i64
  %c8768_i64 = arith.constant 8768 : i64
  %c21088_i64 = arith.constant 21088 : i64
  %c8704_i64 = arith.constant 8704 : i64
  %c7680_i64 = arith.constant 7680 : i64
  %c7616_i64 = arith.constant 7616 : i64
  %c7104_i64 = arith.constant 7104 : i64
  %c6080_i64 = arith.constant 6080 : i64
  %c21024_i64 = arith.constant 21024 : i64
  %c6016_i64 = arith.constant 6016 : i64
  %c14208_i64 = arith.constant 14208 : i64
  %c5952_i64 = arith.constant 5952 : i64
  %c4928_i64 = arith.constant 4928 : i64
  %c20000_i64 = arith.constant 20000 : i64
  %c3904_i64 = arith.constant 3904 : i64
  %c3808_i64 = arith.constant 3808 : i64
  %c2784_i64 = arith.constant 2784 : i64
  %c2752_i64 = arith.constant 2752 : i64
  %c1728_i64 = arith.constant 1728 : i64
  %c1216_i64 = arith.constant 1216 : i64
  %c192_i64 = arith.constant 192 : i64
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c-2_i32 = arith.constant -2 : i32
  %c2_i32 = arith.constant 2 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<16xi32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x1xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %6 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%5 : memref<16xi32, #hivm.address_space<ub>>) outs(%collapse_shape : memref<16xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c128_i64) : memref<1x16xf32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %7 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%5 : memref<16xi32, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<16xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c192_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %9 = hivm.hir.pointer_cast(%c1216_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%6 : memref<16x1xf32, #hivm.address_space<ub>>) outs(%8 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%9 : memref<128xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %10 = hivm.hir.pointer_cast(%c1728_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%7 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%10 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %11 = hivm.hir.pointer_cast(%c2752_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %8 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %10 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %11 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_4 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<256xi1, #hivm.address_space<ub>>) compare_mode = <gt>
  %12 = hivm.hir.pointer_cast(%c192_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %12 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_4 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
  %13 = arith.index_cast %arg5 : i32 to index
  %14 = affine.max affine_map<()[s0] -> (0, s0)>()[%13]
  %15 = affine.min affine_map<()[s0] -> (16, s0)>()[%14]
  %16 = affine.min affine_map<()[s0] -> (0, 16, s0)>()[%14]
  %17 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%15, %16]
  %18 = arith.cmpi slt, %17, %c16 : index
  %19 = arith.minsi %arg5, %c16_i32 : i32
  %20 = arith.andi %19, %c-2_i32 : i32
  %21 = arith.index_cast %19 : i32 to index
  %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
  %23 = affine.min affine_map<()[s0] -> (16, s0)>()[%22]
  %24 = arith.cmpi slt, %23, %c16 : index
  %25 = arith.cmpi slt, %20, %19 : i32
  %26 = hivm.hir.pointer_cast(%c2784_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %26 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %27 = hivm.hir.pointer_cast(%c3808_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%27 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %28 = arith.index_cast %arg9 : i32 to index
    %29 = arith.index_cast %3 : i32 to index
    %30 = arith.index_cast %1 : i32 to index
    %31 = arith.index_cast %c48_i32 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%28, %29, %31]
    %33 = arith.index_cast %32 : index to i1
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %34 = arith.select %33, %c5_i64, %c6_i64 : i64
    %35 = arith.index_cast %32 : index to i1
    %c0_i64_8 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %36 = arith.select %35, %c0_i64_8, %c1_i64 : i64
    %37 = hivm.hir.pointer_cast(%c18976_i64, %c21216_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    %38 = hivm.hir.pointer_cast(%c15264_i64, %c21152_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c3904_i64, %c20000_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %40 = arith.divsi %arg9, %4 : i32
    %41 = arith.remsi %40, %arg6 : i32
    %42 = arith.divsi %41, %c8_i32 : i32
    %43 = arith.remsi %41, %c8_i32 : i32
    %44 = arith.muli %42, %arg5 : i32
    %45 = arith.muli %44, %c8_i32 : i32
    %46 = arith.addi %45, %43 : i32
    %47 = arith.muli %46, %c16_i32 : i32
    %48 = arith.index_cast %47 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%17, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %39[%16, 0] [%17, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %18 {
      %collapse_shape_19 = memref.collapse_shape %39 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_19 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %34]
    hivm.hir.load ins(%subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %49 = hivm.hir.pointer_cast(%c4928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_10 = memref.collapse_shape %39 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %49 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %50 = hivm.hir.pointer_cast(%c5952_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_5, %collapse_shape_10, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_11 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %34]
    %51 = hivm.hir.pointer_cast(%c4928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %51 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_11, %cst_1 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c14208_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %52 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_12, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>>)
    %53 = scf.for %arg10 = %c1_i32 to %20 step %c2_i32 iter_args(%arg11 = %52) -> (memref<16x16xf32, #hivm.address_space<ub>>)  : i32 {
      %54 = arith.index_cast %arg10 : i32 to index
      %55 = arith.index_cast %c1_i32 : i32 to index
      %56 = arith.index_cast %20 : i32 to index
      %57 = arith.index_cast %c2_i32 : i32 to index
      %58 = arith.index_cast %arg9 : i32 to index
      %59 = arith.index_cast %3 : i32 to index
      %60 = arith.index_cast %1 : i32 to index
      %61 = arith.index_cast %c48_i32 : i32 to index
      %62 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%54, %55, %56, %57, %58, %59, %61]
      %63 = arith.index_cast %62 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %64 = arith.select %63, %c3_i64, %c4_i64 : i64
      %65 = hivm.hir.pointer_cast(%c8704_i64, %c21088_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %65 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %66 = hivm.hir.pointer_cast(%c6016_i64, %c21024_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %66 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %67 = arith.muli %arg10, %c128_i32 : i32
      %68 = arith.index_cast %67 : i32 to index
      %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%48, %68]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_20 = memref.subview %reinterpret_cast_19[0] [%23] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_21 = memref.subview %66[0] [%23] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %24 {
        hivm.hir.vbrc ins(%cst : f32) outs(%66 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %64]
      hivm.hir.load ins(%subview_20 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%66, %cst_1 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%66 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%66, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%66 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape = memref.expand_shape %66 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %70 = hivm.hir.pointer_cast(%c6080_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %71 = hivm.hir.pointer_cast(%c7104_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape, %arg11 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%70 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %72 = hivm.hir.pointer_cast(%c7616_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %73 = hivm.hir.pointer_cast(%c7680_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%70 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%72 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%73 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_22 = memref.collapse_shape %72 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%66, %collapse_shape_22 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%66 : memref<16xf32, #hivm.address_space<ub>>)
      %74 = arith.addi %arg10, %c1_i32 : i32
      %75 = arith.muli %74, %c128_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%48, %76]
      %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%77], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_24 = memref.subview %reinterpret_cast_23[0] [%23] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_25 = memref.subview %65[0] [%23] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %24 {
        hivm.hir.vbrc ins(%cst : f32) outs(%65 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_24 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_25 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%65, %cst_1 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%65 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%65, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%65 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_26 = memref.expand_shape %65 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %78 = hivm.hir.pointer_cast(%c8768_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %79 = hivm.hir.pointer_cast(%c9792_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_26, %arg11 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%78 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%79 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %80 = hivm.hir.pointer_cast(%c10304_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %81 = hivm.hir.pointer_cast(%c10368_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%78 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%80 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%81 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_27 = memref.collapse_shape %80 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %82 = hivm.hir.pointer_cast(%c10304_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%65, %collapse_shape_27 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%82 : memref<16xf32, #hivm.address_space<ub>>)
      %83 = hivm.hir.pointer_cast(%c11392_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%arg10 : i32) outs(%83 : memref<16xi32, #hivm.address_space<ub>>)
      %84 = hivm.hir.pointer_cast(%c11392_i64) : memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%5, %83 : memref<16xi32, #hivm.address_space<ub>>, memref<16xi32, #hivm.address_space<ub>>) outs(%84 : memref<16xi1, #hivm.address_space<ub>>)
      %85 = hivm.hir.pointer_cast(%c11456_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%74 : i32) outs(%85 : memref<16xi32, #hivm.address_space<ub>>)
      %86 = hivm.hir.pointer_cast(%c11456_i64) : memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%5, %85 : memref<16xi32, #hivm.address_space<ub>>, memref<16xi32, #hivm.address_space<ub>>) outs(%86 : memref<16xi1, #hivm.address_space<ub>>)
      %87 = hivm.hir.pointer_cast(%c11392_i64) : memref<16x1xi1, #hivm.address_space<ub>>
      %collapse_shape_28 = memref.collapse_shape %87 [[0, 1]] : memref<16x1xi1, #hivm.address_space<ub>> into memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vor ins(%84, %86 : memref<16xi1, #hivm.address_space<ub>>, memref<16xi1, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<16xi1, #hivm.address_space<ub>>)
      %expand_shape_29 = memref.expand_shape %66 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %88 = hivm.hir.pointer_cast(%c11520_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_29 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%88 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %64]
      %expand_shape_30 = memref.expand_shape %82 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %89 = hivm.hir.pointer_cast(%c12544_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_30 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%89 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %90 = arith.index_cast %arg10 : i32 to index
      %91 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%90]
      %92 = arith.cmpi slt, %90, %c0 : index
      %93 = arith.cmpi sge, %90, %c16 : index
      %94 = arith.ori %92, %93 : i1
      scf.if %94 {
      } else {
        %subview_38 = memref.subview %89[0, 0] [%90, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_39 = memref.subview %88[0, 0] [%90, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_40 = memref.collapse_shape %subview_38 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_41 = memref.collapse_shape %subview_39 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_40 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_41 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_42 = memref.subview %88[0, 0] [%91, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_43 = memref.subview %89[0, 0] [%91, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_44 = memref.collapse_shape %subview_42 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_45 = memref.collapse_shape %subview_43 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_44 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_45 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
      %95 = hivm.hir.pointer_cast(%c13568_i64) : memref<16x1xf16, #hivm.address_space<ub>>
      %collapse_shape_31 = memref.collapse_shape %95 [[0, 1]] : memref<16x1xf16, #hivm.address_space<ub>> into memref<16xf16, #hivm.address_space<ub>>
      %96 = hivm.hir.pointer_cast(%c13600_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%collapse_shape_28 : memref<16xi1, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<16xf16, #hivm.address_space<ub>>) temp_buffer(%96 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %97 = hivm.hir.pointer_cast(%c13696_i64) : memref<16x16xf16, #hivm.address_space<ub>>
      %98 = hivm.hir.pointer_cast(%c14208_i64) : memref<0xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%95 : memref<16x1xf16, #hivm.address_space<ub>>) outs(%97 : memref<16x16xf16, #hivm.address_space<ub>>) temp_buffer(%98 : memref<0xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
      %99 = hivm.hir.pointer_cast(%c13696_i64) : memref<16x16xi1, #hivm.address_space<ub>>
      %collapse_shape_32 = memref.collapse_shape %97 [[0, 1]] : memref<16x16xf16, #hivm.address_space<ub>> into memref<256xf16, #hivm.address_space<ub>>
      %collapse_shape_33 = memref.collapse_shape %99 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_32, %cst_0 : memref<256xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_33 : memref<256xi1, #hivm.address_space<ub>>)
      %100 = hivm.hir.pointer_cast(%c13696_i64) : memref<16x16xi1, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %100 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_33 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<256xi1, #hivm.address_space<ub>>)
      %101 = hivm.hir.pointer_cast(%c14208_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_35 = memref.collapse_shape %89 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_36 = memref.collapse_shape %arg11 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_37 = memref.collapse_shape %101 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %102 = hivm.hir.pointer_cast(%c15232_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_34, %collapse_shape_35, %collapse_shape_36 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_37 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%102 : memref<8xf32, #hivm.address_space<ub>>)
      scf.yield %101 : memref<16x16xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    scf.if %25 {
      %54 = arith.muli %20, %c128_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%48, %55]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_20 = memref.subview %reinterpret_cast_19[0] [%23] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_21 = memref.subview %38[0] [%23] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %24 {
        hivm.hir.vbrc ins(%cst : f32) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %36]
      hivm.hir.load ins(%subview_20 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%38, %cst_1 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%38, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape = memref.expand_shape %38 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %57 = hivm.hir.pointer_cast(%c15328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %58 = hivm.hir.pointer_cast(%c16352_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape, %53 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%57 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%58 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %59 = hivm.hir.pointer_cast(%c16864_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %60 = hivm.hir.pointer_cast(%c16928_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%57 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%59 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_22 = memref.collapse_shape %59 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%38, %collapse_shape_22 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_23 = memref.expand_shape %38 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %61 = hivm.hir.pointer_cast(%c17952_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_23 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%61 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %36]
      %62 = arith.index_cast %20 : i32 to index
      %63 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%62]
      %64 = arith.cmpi slt, %62, %c0 : index
      %65 = arith.cmpi sge, %62, %c16 : index
      %66 = arith.ori %64, %65 : i1
      scf.if %66 {
      } else {
        %subview_24 = memref.subview %53[0, 0] [%62, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_25 = memref.subview %61[0, 0] [%62, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_26 = memref.collapse_shape %subview_24 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_27 = memref.collapse_shape %subview_25 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_26 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_28 = memref.subview %61[0, 0] [%63, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_29 = memref.subview %53[0, 0] [%63, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_30 = memref.collapse_shape %subview_28 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_31 = memref.collapse_shape %subview_29 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_30 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    }
    %collapse_shape_14 = memref.collapse_shape %53 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %37 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %36]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_14, %collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %37[%16, 0] [%17, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %reinterpret_cast_16[0, 0] [%17, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_17 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_18 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %36]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
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
