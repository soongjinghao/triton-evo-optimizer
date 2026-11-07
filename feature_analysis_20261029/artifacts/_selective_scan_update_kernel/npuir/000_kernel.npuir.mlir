// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_selective_scan_update_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32, %arg39: i32, %arg40: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<41xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c10528_i64 = arith.constant 10528 : i64
  %c10400_i64 = arith.constant 10400 : i64
  %c9376_i64 = arith.constant 9376 : i64
  %c8352_i64 = arith.constant 8352 : i64
  %c6304_i64 = arith.constant 6304 : i64
  %c6176_i64 = arith.constant 6176 : i64
  %c6048_i64 = arith.constant 6048 : i64
  %c5984_i64 = arith.constant 5984 : i64
  %c5920_i64 = arith.constant 5920 : i64
  %c4896_i64 = arith.constant 4896 : i64
  %c2848_i64 = arith.constant 2848 : i64
  %c2816_i64 = arith.constant 2816 : i64
  %c2688_i64 = arith.constant 2688 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c2176_i64 = arith.constant 2176 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant -1.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 2.000000e+01 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %cst_2 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg38, %arg39 : i32
  %1 = arith.muli %0, %arg40 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg40 : i32
  %5 = arith.divsi %3, %arg40 : i32
  %6 = arith.remsi %5, %arg39 : i32
  %7 = arith.muli %arg40, %arg39 : i32
  %8 = arith.divsi %3, %7 : i32
  %9 = arith.remsi %8, %arg38 : i32
  %10 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : memref<32xf32, #hivm.address_space<ub>>)
  %11 = arith.muli %6, %arg19 : i32
  %12 = arith.muli %4, %arg20 : i32
  %13 = arith.addi %11, %12 : i32
  %14 = arith.index_cast %13 : i32 to index
  %15 = arith.muli %6, %arg22 : i32
  %16 = arith.muli %4, %arg23 : i32
  %17 = arith.addi %15, %16 : i32
  %18 = arith.index_cast %17 : i32 to index
  %19 = arith.muli %6, %arg24 : i32
  %20 = arith.muli %4, %arg25 : i32
  %21 = arith.addi %19, %20 : i32
  %22 = arith.index_cast %21 : i32 to index
  %23 = arith.muli %4, %arg26 : i32
  %24 = arith.index_cast %23 : i32 to index
  %25 = arith.muli %4, %arg27 : i32
  %26 = arith.index_cast %25 : i32 to index
  %27 = arith.muli %6, %arg29 : i32
  %28 = arith.divsi %4, %arg18 : i32
  %29 = arith.muli %28, %arg30 : i32
  %30 = arith.addi %27, %29 : i32
  %31 = arith.index_cast %30 : i32 to index
  %32 = arith.muli %6, %arg31 : i32
  %33 = arith.muli %28, %arg32 : i32
  %34 = arith.addi %32, %33 : i32
  %35 = arith.index_cast %34 : i32 to index
  %36 = arith.muli %6, %arg34 : i32
  %37 = arith.muli %4, %arg35 : i32
  %38 = arith.addi %36, %37 : i32
  %39 = arith.index_cast %38 : i32 to index
  %40 = arith.muli %6, %arg36 : i32
  %41 = arith.muli %4, %arg37 : i32
  %42 = arith.addi %40, %41 : i32
  %43 = arith.index_cast %42 : i32 to index
  %44 = arith.muli %9, %c32_i32 : i32
  %45 = arith.index_cast %44 : i32 to index
  %46 = arith.index_cast %arg21 : i32 to index
  %47 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%14, %45, %46]
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%47], sizes: [32, 16], strides: [%46, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%18, %45]
  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [32, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%22, %45]
  %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%49], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %50 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %45]
  %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %51 = arith.muli %4, %arg33 : i32
  %52 = arith.index_cast %51 : i32 to index
  %53 = arith.index_cast %arg28 : i32 to index
  %54 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%26, %45, %53]
  %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%54], sizes: [32, 16], strides: [%53, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%31], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
  %reinterpret_cast_8 = memref.reinterpret_cast %arg9 to offset: [%35], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
  %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%52, %45]
  %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%55], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%39, %45]
  %reinterpret_cast_10 = memref.reinterpret_cast %arg11 to offset: [%56], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%43, %45]
  %reinterpret_cast_11 = memref.reinterpret_cast %arg12 to offset: [%57], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %58 = hivm.hir.pointer_cast(%c128_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %59 = arith.index_cast %arg16 : i32 to index
  %60 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%45, %59]
  %61 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%60, %45]
  %62 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%61, %45]
  %63 = arith.index_cast %arg17 : i32 to index
  %64 = affine.max affine_map<()[s0] -> (0, s0)>()[%63]
  %65 = affine.min affine_map<()[s0] -> (16, s0)>()[%64]
  %66 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%61, %45]
  %67 = arith.cmpi slt, %66, %c32 : index
  %68 = arith.cmpi slt, %65, %c16 : index
  %69 = arith.ori %67, %68 : i1
  %subview = memref.subview %reinterpret_cast[0, 0] [%66, %65] [1, 1] : memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_12 = memref.subview %58[0, 0] [%66, %65] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  scf.if %69 {
    %collapse_shape_45 = memref.collapse_shape %58 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_45 : memref<512xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_12 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %70 = hivm.hir.pointer_cast(%c2176_i64) : memref<32x1xf32, #hivm.address_space<ub>>
  %71 = arith.cmpi slt, %62, %c32 : index
  %subview_13 = memref.subview %reinterpret_cast_3[0, 0] [%62, 1] [1, 1] : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_14 = memref.subview %70[0, 0] [%62, 1] [1, 1] : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  scf.if %71 {
    %collapse_shape_45 = memref.collapse_shape %70 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_45 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  %collapse_shape = memref.collapse_shape %subview_13 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %collapse_shape_15 = memref.collapse_shape %subview_14 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %collapse_shape_16 = memref.collapse_shape %70 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
  %72 = hivm.hir.pointer_cast(%c2304_i64) : memref<32xf32, #hivm.address_space<ub>>
  %subview_17 = memref.subview %reinterpret_cast_4[0] [%62] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_18 = memref.subview %72[0] [%62] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  scf.if %71 {
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%72 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %73 = hivm.hir.pointer_cast(%c2432_i64) : memref<32xf32, #hivm.address_space<ub>>
  %subview_19 = memref.subview %reinterpret_cast_5[0] [%62] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_20 = memref.subview %73[0] [%62] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  scf.if %71 {
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%73 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_19 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_20 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %74 = hivm.hir.pointer_cast(%c2432_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vadd ins(%72, %73 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%74 : memref<32xf32, #hivm.address_space<ub>>)
  %75 = hivm.hir.pointer_cast(%c0_i64) : memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%74, %10 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%75 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <le>
  %76 = hivm.hir.pointer_cast(%c2560_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vexp ins(%74 : memref<32xf32, #hivm.address_space<ub>>) outs(%76 : memref<32xf32, #hivm.address_space<ub>>)
  %77 = hivm.hir.pointer_cast(%c2560_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%76, %cst_1 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%77 : memref<32xf32, #hivm.address_space<ub>>)
  %78 = hivm.hir.pointer_cast(%c2560_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%77 : memref<32xf32, #hivm.address_space<ub>>) outs(%78 : memref<32xf32, #hivm.address_space<ub>>)
  %79 = hivm.hir.pointer_cast(%c2688_i64) : memref<32xf32, #hivm.address_space<ub>>
  %80 = hivm.hir.pointer_cast(%c2816_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%75, %78, %74 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%79 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%80 : memref<8xf32, #hivm.address_space<ub>>)
  %81 = hivm.hir.pointer_cast(%c2848_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %subview_21 = memref.subview %reinterpret_cast_6[0, 0] [%66, %65] [1, 1] : memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_22 = memref.subview %81[0, 0] [%66, %65] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  scf.if %69 {
    %collapse_shape_45 = memref.collapse_shape %81 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_45 : memref<512xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_21 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_22 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %expand_shape = memref.expand_shape %79 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
  %82 = hivm.hir.pointer_cast(%c2848_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %83 = hivm.hir.pointer_cast(%c4896_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%81, %expand_shape : memref<32x16xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%82 : memref<32x16xf32, #hivm.address_space<ub>>) temp_buffer(%83 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
  %84 = hivm.hir.pointer_cast(%c2848_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %collapse_shape_23 = memref.collapse_shape %82 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  %collapse_shape_24 = memref.collapse_shape %84 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vexp ins(%collapse_shape_23 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<512xf32, #hivm.address_space<ub>>)
  %85 = hivm.hir.pointer_cast(%c5920_i64) : memref<1x16xf32, #hivm.address_space<ub>>
  %subview_25 = memref.subview %reinterpret_cast_7[0, 0] [1, %65] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_26 = memref.subview %85[0, 0] [1, %65] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  scf.if %68 {
    %collapse_shape_45 = memref.collapse_shape %85 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_45 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  %collapse_shape_27 = memref.collapse_shape %subview_25 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %collapse_shape_28 = memref.collapse_shape %subview_26 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%collapse_shape_27 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_28 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
  %86 = hivm.hir.pointer_cast(%c5984_i64) : memref<1x16xf32, #hivm.address_space<ub>>
  %subview_29 = memref.subview %reinterpret_cast_8[0, 0] [1, %65] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_30 = memref.subview %86[0, 0] [1, %65] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  scf.if %68 {
    %collapse_shape_45 = memref.collapse_shape %86 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_45 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  %collapse_shape_31 = memref.collapse_shape %subview_29 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %collapse_shape_32 = memref.collapse_shape %subview_30 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%collapse_shape_31 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_32 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
  %87 = hivm.hir.pointer_cast(%c6048_i64) : memref<32xf32, #hivm.address_space<ub>>
  %subview_33 = memref.subview %reinterpret_cast_9[0] [%62] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_34 = memref.subview %87[0] [%62] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  scf.if %71 {
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%87 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_33 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_34 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
  %88 = hivm.hir.pointer_cast(%c6176_i64) : memref<32xf32, #hivm.address_space<ub>>
  %subview_35 = memref.subview %reinterpret_cast_10[0] [%62] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_36 = memref.subview %88[0] [%62] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  scf.if %71 {
    hivm.hir.vbrc ins(%cst_2 : f32) outs(%88 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_35 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_36 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %89 = hivm.hir.pointer_cast(%c6304_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %90 = hivm.hir.pointer_cast(%c8352_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.vmul ins(%85, %expand_shape : memref<1x16xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%89 : memref<32x16xf32, #hivm.address_space<ub>>) temp_buffer(%90 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %91 = hivm.hir.pointer_cast(%c2848_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %collapse_shape_37 = memref.collapse_shape %58 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  %collapse_shape_38 = memref.collapse_shape %91 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%collapse_shape_37, %collapse_shape_24 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_38 : memref<512xf32, #hivm.address_space<ub>>)
  %92 = hivm.hir.pointer_cast(%c6304_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %93 = hivm.hir.pointer_cast(%c9376_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vmul ins(%89, %70 : memref<32x16xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%92 : memref<32x16xf32, #hivm.address_space<ub>>) temp_buffer(%93 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
  %94 = hivm.hir.pointer_cast(%c2848_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  %collapse_shape_39 = memref.collapse_shape %92 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  %collapse_shape_40 = memref.collapse_shape %94 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%collapse_shape_38, %collapse_shape_39 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<512xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %subview_41 = memref.subview %94[0, 0] [%66, %65] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_41 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  %95 = hivm.hir.pointer_cast(%c2848_i64) : memref<32x16xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vmul ins(%94, %86 : memref<32x16xf32, #hivm.address_space<ub>>, memref<1x16xf32, #hivm.address_space<ub>>) outs(%95 : memref<32x16xf32, #hivm.address_space<ub>>) broadcast = [0]
  %96 = hivm.hir.pointer_cast(%c10400_i64) : memref<32x1xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vreduce <sum> ins(%95 : memref<32x16xf32, #hivm.address_space<ub>>) outs(%96 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
  %collapse_shape_42 = memref.collapse_shape %96 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
  %97 = hivm.hir.pointer_cast(%c2176_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.vmul ins(%collapse_shape_16, %87 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%97 : memref<32xf32, #hivm.address_space<ub>>)
  %98 = hivm.hir.pointer_cast(%c2176_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%collapse_shape_42, %97 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%98 : memref<32xf32, #hivm.address_space<ub>>)
  %99 = hivm.hir.pointer_cast(%c10528_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vmul ins(%88, %cst : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%99 : memref<32xf32, #hivm.address_space<ub>>)
  %100 = hivm.hir.pointer_cast(%c10528_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%99, %cst_2 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<32xf32, #hivm.address_space<ub>>)
  %101 = hivm.hir.pointer_cast(%c10528_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vexp ins(%100 : memref<32xf32, #hivm.address_space<ub>>) outs(%101 : memref<32xf32, #hivm.address_space<ub>>)
  %102 = hivm.hir.pointer_cast(%c10528_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%101, %cst_1 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%102 : memref<32xf32, #hivm.address_space<ub>>)
  %103 = hivm.hir.pointer_cast(%c10528_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vdiv ins(%88, %102 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%103 : memref<32xf32, #hivm.address_space<ub>>)
  %104 = hivm.hir.pointer_cast(%c10528_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%98, %103 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%104 : memref<32xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %subview_43 = memref.subview %104[0] [%62] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_44 = memref.subview %reinterpret_cast_11[0] [%62] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_43 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_44 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
