// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_selective_scan_update_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32, %arg39: i32, %arg40: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<41xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c17600_i64 = arith.constant 17600 : i64
  %c10560_i64 = arith.constant 10560 : i64
  %c10432_i64 = arith.constant 10432 : i64
  %c9408_i64 = arith.constant 9408 : i64
  %c8384_i64 = arith.constant 8384 : i64
  %c15552_i64 = arith.constant 15552 : i64
  %c6336_i64 = arith.constant 6336 : i64
  %c15424_i64 = arith.constant 15424 : i64
  %c6208_i64 = arith.constant 6208 : i64
  %c15296_i64 = arith.constant 15296 : i64
  %c6080_i64 = arith.constant 6080 : i64
  %c15232_i64 = arith.constant 15232 : i64
  %c6016_i64 = arith.constant 6016 : i64
  %c15168_i64 = arith.constant 15168 : i64
  %c5952_i64 = arith.constant 5952 : i64
  %c4928_i64 = arith.constant 4928 : i64
  %c13120_i64 = arith.constant 13120 : i64
  %c2880_i64 = arith.constant 2880 : i64
  %c2848_i64 = arith.constant 2848 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c12992_i64 = arith.constant 12992 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c12864_i64 = arith.constant 12864 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c12736_i64 = arith.constant 12736 : i64
  %c2176_i64 = arith.constant 2176 : i64
  %c10688_i64 = arith.constant 10688 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %cst_2 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg38, %arg39 : i32
  %1 = arith.muli %0, %arg40 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg40, %arg39 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%5 : memref<32xf32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg21 : i32 to index
  %7 = arith.index_cast %arg28 : i32 to index
  %8 = arith.index_cast %arg16 : i32 to index
  %9 = arith.index_cast %arg17 : i32 to index
  %10 = affine.max affine_map<()[s0] -> (0, s0)>()[%9]
  %11 = affine.min affine_map<()[s0] -> (16, s0)>()[%10]
  %12 = affine.max affine_map<()[s0] -> (0, s0)>()[%11]
  %13 = arith.cmpi slt, %12, %c16 : index
  %14 = arith.cmpi slt, %11, %c16 : index
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg41 = %3 to %1 step %c48_i32  : i32 {
    %15 = arith.index_cast %arg41 : i32 to index
    %16 = arith.index_cast %3 : i32 to index
    %17 = arith.index_cast %1 : i32 to index
    %18 = arith.index_cast %c48_i32 : i32 to index
    %19 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%15, %16, %18]
    %20 = arith.index_cast %19 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %21 = arith.select %20, %c3_i64, %c4_i64 : i64
    %22 = arith.index_cast %19 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %23 = arith.select %22, %c1_i64, %c2_i64 : i64
    %24 = hivm.hir.pointer_cast(%c10560_i64, %c17600_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c6336_i64, %c15552_i64) : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<32x16xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c6208_i64, %c15424_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c6080_i64, %c15296_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c6016_i64, %c15232_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<1x16xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c5952_i64, %c15168_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<1x16xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c2880_i64, %c13120_i64) : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<32x16xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c2432_i64, %c12992_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c2304_i64, %c12864_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c2176_i64, %c12736_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<32x1xf32, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c128_i64, %c10688_i64) : memref<32x16xf32, #hivm.address_space<ub>>
    annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<32x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %35 = arith.remsi %arg41, %arg40 : i32
    %36 = arith.divsi %arg41, %arg40 : i32
    %37 = arith.remsi %36, %arg39 : i32
    %38 = arith.divsi %arg41, %4 : i32
    %39 = arith.remsi %38, %arg38 : i32
    %40 = arith.muli %37, %arg19 : i32
    %41 = arith.muli %35, %arg20 : i32
    %42 = arith.addi %40, %41 : i32
    %43 = arith.muli %37, %arg22 : i32
    %44 = arith.muli %35, %arg23 : i32
    %45 = arith.addi %43, %44 : i32
    %46 = arith.muli %37, %arg24 : i32
    %47 = arith.muli %35, %arg25 : i32
    %48 = arith.addi %46, %47 : i32
    %49 = arith.muli %35, %arg26 : i32
    %50 = arith.muli %35, %arg27 : i32
    %51 = arith.muli %37, %arg29 : i32
    %52 = arith.divsi %35, %arg18 : i32
    %53 = arith.muli %52, %arg30 : i32
    %54 = arith.addi %51, %53 : i32
    %55 = arith.muli %37, %arg31 : i32
    %56 = arith.muli %52, %arg32 : i32
    %57 = arith.addi %55, %56 : i32
    %58 = arith.muli %37, %arg34 : i32
    %59 = arith.muli %35, %arg35 : i32
    %60 = arith.addi %58, %59 : i32
    %61 = arith.muli %37, %arg36 : i32
    %62 = arith.muli %35, %arg37 : i32
    %63 = arith.addi %61, %62 : i32
    %64 = arith.muli %39, %c32_i32 : i32
    %65 = arith.index_cast %42 : i32 to index
    %66 = arith.index_cast %64 : i32 to index
    %67 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%65, %66, %6]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [32, 16], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %68 = arith.index_cast %45 : i32 to index
    %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%68, %66]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%69], sizes: [32, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %70 = arith.index_cast %48 : i32 to index
    %71 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%70, %66]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%71], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %72 = arith.index_cast %49 : i32 to index
    %73 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%72, %66]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%73], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %74 = arith.muli %35, %arg33 : i32
    %75 = arith.index_cast %50 : i32 to index
    %76 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%75, %66, %7]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%76], sizes: [32, 16], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %77 = arith.index_cast %54 : i32 to index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%77], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %78 = arith.index_cast %57 : i32 to index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg9 to offset: [%78], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %79 = arith.index_cast %74 : i32 to index
    %80 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%79, %66]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%80], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %81 = arith.index_cast %60 : i32 to index
    %82 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%81, %66]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg11 to offset: [%82], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %83 = arith.index_cast %63 : i32 to index
    %84 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%83, %66]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg12 to offset: [%84], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %85 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%66, %8]
    %86 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%85, %66]
    %87 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%86, %66]
    %88 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%86, %66]
    %89 = affine.max affine_map<()[s0] -> (0, s0)>()[%88]
    %90 = arith.cmpi slt, %89, %c32 : index
    %91 = arith.ori %90, %13 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%89, %12] [1, 1] : memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %34[0, 0] [%89, %12] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %91 {
      %collapse_shape_45 = memref.collapse_shape %34 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_12 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %92 = arith.cmpi slt, %87, %c32 : index
    %subview_13 = memref.subview %reinterpret_cast_3[0, 0] [%87, 1] [1, 1] : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_14 = memref.subview %33[0, 0] [%87, 1] [1, 1] : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %92 {
      %collapse_shape_45 = memref.collapse_shape %33 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape = memref.collapse_shape %subview_13 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_15 = memref.collapse_shape %subview_14 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
    hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %collapse_shape_16 = memref.collapse_shape %33 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %subview_17 = memref.subview %reinterpret_cast_4[0] [%87] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %32[0] [%87] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %92 {
      hivm.hir.vbrc ins(%cst : f32) outs(%32 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %subview_19 = memref.subview %reinterpret_cast_5[0] [%87] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_20 = memref.subview %31[0] [%87] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %92 {
      hivm.hir.vbrc ins(%cst : f32) outs(%31 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_19 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_20 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ins(%32, %31 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%32 : memref<32xf32, #hivm.address_space<ub>>)
    %93 = hivm.hir.pointer_cast(%c2560_i64) : memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%32, %5 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%93 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <le>
    %94 = hivm.hir.pointer_cast(%c2592_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%32 : memref<32xf32, #hivm.address_space<ub>>) outs(%94 : memref<32xf32, #hivm.address_space<ub>>)
    %95 = hivm.hir.pointer_cast(%c2592_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%94, %cst_0 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%95 : memref<32xf32, #hivm.address_space<ub>>)
    %96 = hivm.hir.pointer_cast(%c2592_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%95 : memref<32xf32, #hivm.address_space<ub>>) outs(%96 : memref<32xf32, #hivm.address_space<ub>>)
    %97 = hivm.hir.pointer_cast(%c2720_i64) : memref<32xf32, #hivm.address_space<ub>>
    %98 = hivm.hir.pointer_cast(%c2848_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%93, %96, %32 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%97 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%98 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_21 = memref.subview %reinterpret_cast_6[0, 0] [%89, %12] [1, 1] : memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_22 = memref.subview %30[0, 0] [%89, %12] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %91 {
      %collapse_shape_45 = memref.collapse_shape %30 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_21 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_22 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape = memref.expand_shape %97 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %99 = hivm.hir.pointer_cast(%c4928_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%30, %expand_shape : memref<32x16xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%30 : memref<32x16xf32, #hivm.address_space<ub>>) temp_buffer(%99 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %collapse_shape_23 = memref.collapse_shape %30 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %30 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_23 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<512xf32, #hivm.address_space<ub>>)
    %subview_25 = memref.subview %reinterpret_cast_7[0, 0] [1, %11] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_26 = memref.subview %29[0, 0] [1, %11] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %14 {
      %collapse_shape_45 = memref.collapse_shape %29 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_27 = memref.collapse_shape %subview_25 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_28 = memref.collapse_shape %subview_26 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_27 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_28 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    %subview_29 = memref.subview %reinterpret_cast_8[0, 0] [1, %11] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_30 = memref.subview %28[0, 0] [1, %11] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %14 {
      %collapse_shape_45 = memref.collapse_shape %28 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_45 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_31 = memref.collapse_shape %subview_29 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_32 = memref.collapse_shape %subview_30 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_31 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_32 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    %subview_33 = memref.subview %reinterpret_cast_9[0] [%87] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_34 = memref.subview %27[0] [%87] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %92 {
      hivm.hir.vbrc ins(%cst : f32) outs(%27 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %21]
    hivm.hir.load ins(%subview_33 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_34 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %subview_35 = memref.subview %reinterpret_cast_10[0] [%87] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_36 = memref.subview %26[0] [%87] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %92 {
      hivm.hir.vbrc ins(%cst : f32) outs(%26 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_35 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_36 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %100 = hivm.hir.pointer_cast(%c8384_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.vmul ins(%29, %expand_shape : memref<1x16xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%25 : memref<32x16xf32, #hivm.address_space<ub>>) temp_buffer(%100 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [0, 1]
    %collapse_shape_37 = memref.collapse_shape %34 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %collapse_shape_38 = memref.collapse_shape %30 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_37, %collapse_shape_24 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_38 : memref<512xf32, #hivm.address_space<ub>>)
    %101 = hivm.hir.pointer_cast(%c9408_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%25, %33 : memref<32x16xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%25 : memref<32x16xf32, #hivm.address_space<ub>>) temp_buffer(%101 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %collapse_shape_39 = memref.collapse_shape %25 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %collapse_shape_40 = memref.collapse_shape %25 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_38, %collapse_shape_39 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<512xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_41 = memref.subview %25[0, 0] [%89, %12] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_41 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%25, %28 : memref<32x16xf32, #hivm.address_space<ub>>, memref<1x16xf32, #hivm.address_space<ub>>) outs(%25 : memref<32x16xf32, #hivm.address_space<ub>>) broadcast = [0]
    %102 = hivm.hir.pointer_cast(%c10432_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%25 : memref<32x16xf32, #hivm.address_space<ub>>) outs(%102 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_42 = memref.collapse_shape %102 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vmul ins(%collapse_shape_16, %27 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%27 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_42, %27 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%27 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%26, %cst_2 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%24 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%24, %cst : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%24 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%24 : memref<32xf32, #hivm.address_space<ub>>) outs(%24 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%24, %cst_0 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%24 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%26, %24 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%24 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%27, %24 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%24 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %21]
    %subview_43 = memref.subview %24[0] [%87] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_44 = memref.subview %reinterpret_cast_11[0] [%87] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_43 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_44 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
