// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c10080_i64 = arith.constant 10080 : i64
  %c9056_i64 = arith.constant 9056 : i64
  %c8992_i64 = arith.constant 8992 : i64
  %c8480_i64 = arith.constant 8480 : i64
  %c7456_i64 = arith.constant 7456 : i64
  %c13152_i64 = arith.constant 13152 : i64
  %c7392_i64 = arith.constant 7392 : i64
  %c7328_i64 = arith.constant 7328 : i64
  %c12128_i64 = arith.constant 12128 : i64
  %c6304_i64 = arith.constant 6304 : i64
  %c11104_i64 = arith.constant 11104 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c5184_i64 = arith.constant 5184 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c2_i32 = arith.constant 2 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
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
  %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c1088_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%6 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%7 : memref<128xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %expand_shape_1 = memref.expand_shape %5 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
  %8 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x16xi32, #hivm.address_space<ub>>) outs(%8 : memref<16x16xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %9 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %10 = hivm.hir.pointer_cast(%c3648_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x16xi32, #hivm.address_space<ub>>) outs(%9 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%10 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %11 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %9 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %8 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %11 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %12 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %6 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %13 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
  %14 = arith.index_cast %arg5 : i32 to index
  %15 = hivm.hir.pointer_cast(%c4160_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %15 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %16 = hivm.hir.pointer_cast(%c5184_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%16 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %17 = arith.index_cast %arg9 : i32 to index
    %18 = arith.index_cast %3 : i32 to index
    %19 = arith.index_cast %1 : i32 to index
    %20 = arith.index_cast %c48_i32 : i32 to index
    %21 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%17, %18, %20]
    %22 = arith.index_cast %21 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %23 = arith.select %22, %c3_i64, %c4_i64 : i64
    %24 = arith.index_cast %21 : index to i1
    %c0_i64_8 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %25 = arith.select %24, %c0_i64_8, %c1_i64 : i64
    %26 = hivm.hir.pointer_cast(%c6304_i64, %c12128_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c5280_i64, %c11104_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %28 = arith.divsi %arg9, %arg8 : i32
    %29 = arith.remsi %28, %arg7 : i32
    %30 = arith.divsi %arg9, %4 : i32
    %31 = arith.remsi %30, %arg6 : i32
    %32 = arith.divsi %29, %c8_i32 : i32
    %33 = arith.remsi %29, %c8_i32 : i32
    %34 = arith.muli %32, %arg5 : i32
    %35 = arith.muli %34, %c8_i32 : i32
    %36 = arith.addi %35, %33 : i32
    %37 = arith.muli %36, %c16_i32 : i32
    %38 = arith.index_cast %37 : i32 to index
    %39 = arith.muli %31, %c16_i32 : i32
    %40 = arith.remsi %39, %c16_i32 : i32
    %41 = arith.maxsi %39, %c0_i32 : i32
    %42 = arith.index_cast %41 : i32 to index
    %43 = arith.maxsi %40, %c0_i32 : i32
    %44 = arith.index_cast %43 : i32 to index
    %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%38, %42]
    %46 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%44, %38, %42]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%46], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %47 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>()[%14, %44, %42]
    %48 = affine.min affine_map<()[s0] -> (16, s0)>()[%47]
    %49 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 16, 0)>()[%44]
    %50 = affine.min affine_map<()[s0] -> (16, s0)>()[%49]
    %51 = arith.subi %c0_i32, %39 : i32
    %52 = arith.maxsi %51, %c0_i32 : i32
    %53 = arith.index_cast %52 : i32 to index
    %54 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%53, %47]
    %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%48, %54]
    %56 = arith.subi %c0_i32, %40 : i32
    %57 = arith.maxsi %56, %c0_i32 : i32
    %58 = arith.index_cast %57 : i32 to index
    %59 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%58, %49]
    %60 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %59]
    %61 = arith.cmpi slt, %55, %c16 : index
    %62 = arith.cmpi slt, %60, %c16 : index
    %63 = arith.ori %61, %62 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%55, %60] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %27[%54, %59] [%55, %60] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %63 {
      %collapse_shape_18 = memref.collapse_shape %27 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %59 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %collapse_shape_10 = memref.collapse_shape %27 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %26 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %64 = hivm.hir.pointer_cast(%c7328_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %25]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_10, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_11 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%64 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
    %collapse_shape_12 = memref.collapse_shape %26 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_11, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>)
    %collapse_shape_13 = memref.collapse_shape %26 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_12, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>>)
    %65 = arith.subi %arg5, %39 : i32
    %66 = arith.minsi %65, %c16_i32 : i32
    %67 = arith.index_cast %40 : i32 to index
    scf.for %arg10 = %c2_i32 to %66 step %c1_i32  : i32 {
      %72 = arith.index_cast %arg10 : i32 to index
      %73 = arith.index_cast %c2_i32 : i32 to index
      %74 = arith.index_cast %66 : i32 to index
      %75 = arith.index_cast %c1_i32 : i32 to index
      %76 = arith.index_cast %arg9 : i32 to index
      %77 = arith.index_cast %3 : i32 to index
      %78 = arith.index_cast %1 : i32 to index
      %79 = arith.index_cast %c48_i32 : i32 to index
      %80 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%72, %73, %74, %75, %76, %77, %79]
      %81 = arith.index_cast %80 : index to i1
      %c1_i64_18 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %82 = arith.select %81, %c1_i64_18, %c2_i64 : i64
      %83 = hivm.hir.pointer_cast(%c7392_i64, %c13152_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %83 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %84 = arith.addi %39, %arg10 : i32
      %85 = arith.muli %84, %c128_i32 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%67, %38, %86]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %82]
      hivm.hir.load ins(%reinterpret_cast_19 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%83 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%83, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%83 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%83, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%83 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_20 = memref.expand_shape %83 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %88 = hivm.hir.pointer_cast(%c7456_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %89 = hivm.hir.pointer_cast(%c8480_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_20, %26 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%88 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%89 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %90 = hivm.hir.pointer_cast(%c8992_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %91 = hivm.hir.pointer_cast(%c9056_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%88 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%90 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%91 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_21 = memref.collapse_shape %90 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %92 = hivm.hir.pointer_cast(%c8992_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%83, %collapse_shape_21 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%92 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %82]
      %expand_shape_22 = memref.expand_shape %92 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %93 = hivm.hir.pointer_cast(%c10080_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_22 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%93 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %94 = arith.index_cast %arg10 : i32 to index
      %95 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%94]
      %96 = arith.cmpi slt, %94, %c0 : index
      %97 = arith.cmpi sge, %94, %c16 : index
      %98 = arith.ori %96, %97 : i1
      scf.if %98 {
      } else {
        %subview_23 = memref.subview %26[0, 0] [%94, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_24 = memref.subview %93[0, 0] [%94, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_25 = memref.collapse_shape %subview_23 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_26 = memref.collapse_shape %subview_24 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_25 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_27 = memref.subview %93[0, 0] [%95, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_28 = memref.subview %26[0, 0] [%95, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_29 = memref.collapse_shape %subview_27 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_30 = memref.collapse_shape %subview_28 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_29 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %collapse_shape_14 = memref.collapse_shape %26 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_13, %collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_15 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %68 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%14, %42]
    %69 = affine.min affine_map<()[s0] -> (16, s0)>()[%68]
    %70 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%53, %68]
    %71 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%69, %70]
    %subview_16 = memref.subview %26[%70, 0] [%71, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_17 = memref.subview %reinterpret_cast_15[0, 0] [%71, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_16 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_17 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %25]
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
