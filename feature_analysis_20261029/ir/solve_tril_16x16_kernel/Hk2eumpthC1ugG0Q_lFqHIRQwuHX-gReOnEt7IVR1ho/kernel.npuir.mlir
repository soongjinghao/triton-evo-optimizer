// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c11008_i64 = arith.constant 11008 : i64
  %c9984_i64 = arith.constant 9984 : i64
  %c8960_i64 = arith.constant 8960 : i64
  %c7936_i64 = arith.constant 7936 : i64
  %c7872_i64 = arith.constant 7872 : i64
  %c7360_i64 = arith.constant 7360 : i64
  %c6336_i64 = arith.constant 6336 : i64
  %c11104_i64 = arith.constant 11104 : i64
  %c6272_i64 = arith.constant 6272 : i64
  %c6208_i64 = arith.constant 6208 : i64
  %c5184_i64 = arith.constant 5184 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant -1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg8 : i32
  %5 = arith.remsi %4, %arg7 : i32
  %6 = arith.muli %arg8, %arg7 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg6 : i32
  %9 = arith.divsi %5, %c8_i32 : i32
  %10 = arith.remsi %5, %c8_i32 : i32
  %11 = arith.muli %9, %arg5 : i32
  %12 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%12 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %12 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %13 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %14 = hivm.hir.pointer_cast(%c1088_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%13 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%14 : memref<128xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %expand_shape_1 = memref.expand_shape %12 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x16xi32, #hivm.address_space<ub>>) outs(%15 : memref<16x16xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %16 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %17 = hivm.hir.pointer_cast(%c3648_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x16xi32, #hivm.address_space<ub>>) outs(%16 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%17 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %18 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %15 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %18 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>)
  %19 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %19 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>)
  %20 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %13 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %20 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
  %21 = arith.muli %11, %c8_i32 : i32
  %22 = arith.addi %21, %10 : i32
  %23 = arith.muli %22, %c16_i32 : i32
  %24 = arith.index_cast %23 : i32 to index
  %25 = arith.muli %8, %c16_i32 : i32
  %26 = arith.remsi %25, %c16_i32 : i32
  %27 = arith.maxsi %25, %c0_i32 : i32
  %28 = arith.index_cast %27 : i32 to index
  %29 = arith.maxsi %26, %c0_i32 : i32
  %30 = arith.index_cast %29 : i32 to index
  %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%24, %28]
  %32 = arith.index_cast %arg5 : i32 to index
  %33 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%30, %24, %28]
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  %34 = hivm.hir.pointer_cast(%c4160_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %35 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>()[%32, %30, %28]
  %36 = affine.min affine_map<()[s0] -> (16, s0)>()[%35]
  %37 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 16, 0)>()[%30]
  %38 = affine.min affine_map<()[s0] -> (16, s0)>()[%37]
  %39 = arith.subi %c0_i32, %25 : i32
  %40 = arith.maxsi %39, %c0_i32 : i32
  %41 = arith.index_cast %40 : i32 to index
  %42 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%41, %35]
  %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%36, %42]
  %44 = arith.subi %c0_i32, %26 : i32
  %45 = arith.maxsi %44, %c0_i32 : i32
  %46 = arith.index_cast %45 : i32 to index
  %47 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%46, %37]
  %48 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %47]
  %49 = arith.cmpi slt, %43, %c16 : index
  %50 = arith.cmpi slt, %48, %c16 : index
  %51 = arith.ori %49, %50 : i1
  %subview = memref.subview %reinterpret_cast[0, 0] [%43, %48] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_7 = memref.subview %34[%42, %47] [%43, %48] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  scf.if %51 {
    %collapse_shape_17 = memref.collapse_shape %34 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_17 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %47 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %52 = hivm.hir.pointer_cast(%c5184_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %34 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %collapse_shape_9 = memref.collapse_shape %52 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %53 = hivm.hir.pointer_cast(%c6208_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_8, %cst_0 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_9 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<16xf32, #hivm.address_space<ub>>)
  %54 = hivm.hir.pointer_cast(%c5184_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_10 = memref.collapse_shape %54 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%collapse_shape_9, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_10 : memref<256xf32, #hivm.address_space<ub>>)
  %55 = hivm.hir.pointer_cast(%c5184_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_11 = memref.collapse_shape %55 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%collapse_shape_10, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_11 : memref<256xf32, #hivm.address_space<ub>>)
  %56 = arith.subi %arg5, %25 : i32
  %57 = arith.minsi %56, %c16_i32 : i32
  %58 = arith.index_cast %26 : i32 to index
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg9 = %c2_i32 to %57 step %c1_i32  : i32 {
    %66 = arith.index_cast %arg9 : i32 to index
    %67 = arith.index_cast %c2_i32 : i32 to index
    %68 = arith.index_cast %57 : i32 to index
    %69 = arith.index_cast %c1_i32 : i32 to index
    %70 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%66, %67, %69]
    %71 = arith.index_cast %70 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %72 = arith.select %71, %c1_i64, %c2_i64 : i64
    %73 = hivm.hir.pointer_cast(%c6272_i64, %c11104_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %73 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %74 = arith.addi %25, %arg9 : i32
    %75 = arith.muli %74, %c128_i32 : i32
    %76 = arith.index_cast %75 : i32 to index
    %77 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%58, %24, %76]
    %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%77], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %72]
    hivm.hir.load ins(%reinterpret_cast_17 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%73 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%73, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%73 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%73, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%73 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_18 = memref.expand_shape %73 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %78 = hivm.hir.pointer_cast(%c6336_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %79 = hivm.hir.pointer_cast(%c7360_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_18, %55 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%78 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%79 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %80 = hivm.hir.pointer_cast(%c7872_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %81 = hivm.hir.pointer_cast(%c7936_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%78 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%80 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%81 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_19 = memref.collapse_shape %80 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %82 = hivm.hir.pointer_cast(%c7872_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%73, %collapse_shape_19 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%82 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %72]
    %expand_shape_20 = memref.expand_shape %82 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %83 = hivm.hir.pointer_cast(%c8960_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_20 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%83 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %84 = arith.index_cast %arg9 : i32 to index
    %85 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%84]
    %86 = arith.cmpi slt, %84, %c0 : index
    %87 = arith.cmpi sge, %84, %c16 : index
    %88 = arith.ori %86, %87 : i1
    scf.if %88 {
    } else {
      %subview_21 = memref.subview %55[0, 0] [%84, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_22 = memref.subview %83[0, 0] [%84, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %subview_21 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_24 = memref.collapse_shape %subview_22 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%collapse_shape_23 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %subview_25 = memref.subview %83[0, 0] [%85, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_26 = memref.subview %55[0, 0] [%85, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_27 = memref.collapse_shape %subview_25 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_28 = memref.collapse_shape %subview_26 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%collapse_shape_27 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    }
  }
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  %59 = hivm.hir.pointer_cast(%c9984_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_12 = memref.collapse_shape %59 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %60 = hivm.hir.pointer_cast(%c11008_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  %61 = hivm.hir.pointer_cast(%c5184_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_13 = memref.collapse_shape %61 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%collapse_shape_11, %collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  %62 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%32, %28]
  %63 = affine.min affine_map<()[s0] -> (16, s0)>()[%62]
  %64 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%41, %62]
  %65 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%63, %64]
  %subview_15 = memref.subview %61[%64, 0] [%65, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%65, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_15 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_16 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
