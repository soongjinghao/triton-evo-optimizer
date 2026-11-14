// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_state_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<35xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c2240_i64 = arith.constant 2240 : i64
  %c1216_i64 = arith.constant 1216 : i64
  %c1184_i64 = arith.constant 1184 : i64
  %c3840_i64 = arith.constant 3840 : i64
  %c1120_i64 = arith.constant 1120 : i64
  %c3776_i64 = arith.constant 3776 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c3904_i64 = arith.constant 3904 : i64
  %c2752_i64 = arith.constant 2752 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c15_i32 = arith.constant 15 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i32 = arith.constant 16 : i32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg32, %arg33 : i32
  %1 = arith.muli %0, %arg34 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg34, %arg33 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<ub>>)
  %6 = arith.extsi %arg11 : i32 to i64
  %7 = arith.addi %arg9, %c15_i32 : i32
  %8 = arith.divsi %7, %c16_i32 : i32
  %9 = arith.extsi %arg14 : i32 to i64
  %10 = arith.extsi %arg10 : i32 to i64
  %11 = arith.extsi %arg15 : i32 to i64
  %12 = arith.extsi %arg17 : i32 to i64
  %13 = arith.extsi %arg18 : i32 to i64
  %14 = arith.extsi %arg24 : i32 to i64
  %15 = arith.extsi %arg25 : i32 to i64
  %16 = arith.extsi %arg27 : i32 to i64
  %17 = arith.extsi %arg28 : i32 to i64
  %18 = arith.extsi %arg12 : i32 to i64
  %19 = arith.subi %arg10, %c1_i32 : i32
  %20 = arith.index_cast %19 : i32 to index
  %21 = arith.index_cast %arg8 : i32 to index
  %22 = arith.index_cast %arg9 : i32 to index
  %23 = arith.extsi %arg20 : i32 to i64
  %24 = arith.extsi %arg21 : i32 to i64
  %25 = arith.index_cast %arg23 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg35 = %3 to %1 step %c48_i32  : i32 {
    %26 = arith.index_cast %arg35 : i32 to index
    %27 = arith.index_cast %3 : i32 to index
    %28 = arith.index_cast %1 : i32 to index
    %29 = arith.index_cast %c48_i32 : i32 to index
    %30 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%26, %27, %29]
    %31 = arith.index_cast %30 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %32 = arith.select %31, %c0_i64_0, %c1_i64_1 : i64
    %33 = hivm.hir.pointer_cast(%c2752_i64, %c3904_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %34 = arith.remsi %arg35, %arg34 : i32
    %35 = arith.divsi %arg35, %arg34 : i32
    %36 = arith.remsi %35, %arg33 : i32
    %37 = arith.divsi %arg35, %4 : i32
    %38 = arith.remsi %37, %arg32 : i32
    %39 = arith.extsi %36 : i32 to i64
    %40 = arith.divsi %39, %6 : i64
    %41 = arith.muli %40, %6 : i64
    %42 = arith.subi %39, %41 : i64
    %43 = arith.divsi %38, %8 : i32
    %44 = arith.remsi %38, %8 : i32
    %45 = arith.muli %42, %9 : i64
    %46 = arith.muli %40, %10 : i64
    %47 = arith.muli %46, %11 : i64
    %48 = arith.addi %45, %47 : i64
    %49 = arith.muli %34, %arg16 : i32
    %50 = arith.extsi %49 : i32 to i64
    %51 = arith.addi %48, %50 : i64
    %52 = arith.muli %42, %12 : i64
    %53 = arith.muli %46, %13 : i64
    %54 = arith.addi %52, %53 : i64
    %55 = arith.divsi %34, %arg13 : i32
    %56 = arith.muli %55, %arg19 : i32
    %57 = arith.extsi %56 : i32 to i64
    %58 = arith.addi %54, %57 : i64
    %59 = arith.muli %42, %14 : i64
    %60 = arith.muli %40, %15 : i64
    %61 = arith.addi %59, %60 : i64
    %62 = arith.muli %34, %arg26 : i32
    %63 = arith.extsi %62 : i32 to i64
    %64 = arith.addi %61, %63 : i64
    %65 = arith.muli %42, %16 : i64
    %66 = arith.muli %40, %17 : i64
    %67 = arith.addi %65, %66 : i64
    %68 = arith.muli %34, %arg29 : i32
    %69 = arith.extsi %68 : i32 to i64
    %70 = arith.addi %67, %69 : i64
    %71 = arith.muli %43, %c16_i32 : i32
    %72 = arith.muli %44, %c16_i32 : i32
    %73 = arith.subi %18, %46 : i64
    %74 = arith.minsi %10, %73 : i64
    %75 = arith.index_cast %70 : i64 to index
    %76 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%75, %20]
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%76], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %77 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %78 = arith.index_cast %71 : i32 to index
    %79 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%78, %21]
    %80 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%79, %78]
    %81 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%80, %78]
    %82 = arith.cmpi slt, %81, %c16 : index
    %83 = arith.index_cast %72 : i32 to index
    %84 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%83, %22]
    %85 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%84, %83]
    %86 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%85, %83]
    %87 = arith.cmpi slt, %86, %c16 : index
    %88 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %77, %88[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %collapse_shape_2 = memref.collapse_shape %33 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %32]
    hivm.hir.copy ins(%collapse_shape : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<256xf32, #hivm.address_space<ub>>)
    %89:5 = scf.for %arg36 = %c0_i64 to %74 step %c1_i64 iter_args(%arg37 = %33, %arg38 = %51, %arg39 = %58, %arg40 = %64, %arg41 = %70) -> (memref<16x16xf32, #hivm.address_space<ub>>, i64, i64, i64, i64)  : i64 {
      %102 = arith.index_cast %arg36 : i64 to index
      %103 = arith.index_cast %c0_i64 : i64 to index
      %104 = arith.index_cast %74 : i64 to index
      %105 = arith.index_cast %c1_i64 : i64 to index
      %106 = arith.index_cast %arg35 : i32 to index
      %107 = arith.index_cast %3 : i32 to index
      %108 = arith.index_cast %1 : i32 to index
      %109 = arith.index_cast %c48_i32 : i32 to index
      %110 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%102, %103, %104, %105, %106, %107, %109]
      %111 = arith.index_cast %110 : index to i1
      %c1_i64_5 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %112 = arith.select %111, %c1_i64_5, %c2_i64 : i64
      %113 = hivm.hir.pointer_cast(%c1120_i64, %c3840_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %113 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %114 = hivm.hir.pointer_cast(%c1056_i64, %c3776_i64) : memref<16x1xf32, #hivm.address_space<ub>>
      annotation.mark %114 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
      %115 = arith.index_cast %arg41 : i64 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%115], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %116 = arith.index_cast %arg40 : i64 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%116], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %117 = arith.index_cast %arg39 : i64 to index
      %118 = arith.index_cast %arg38 : i64 to index
      %119 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%118, %78]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%119], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [%81, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %114[0, 0] [%81, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %82 {
        %collapse_shape_19 = memref.collapse_shape %114 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_19 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_12 = memref.collapse_shape %subview_10 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %112]
      hivm.hir.load ins(%collapse_shape_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %120 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%117, %83]
      %reinterpret_cast_13 = memref.reinterpret_cast %arg4 to offset: [%120], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_14 = memref.subview %reinterpret_cast_13[0] [%86] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_15 = memref.subview %113[0] [%86] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %87 {
        hivm.hir.vbrc ins(%cst : f32) outs(%113 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_14 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %121 = memref.load %reinterpret_cast_6[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %122 = memref.load %reinterpret_cast_7[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %123 = memref.load %88[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %124 = arith.subf %123, %121 : f32
      %125 = hivm.hir.pointer_cast(%c1184_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %124, %125[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %126 = hivm.hir.pointer_cast(%c1184_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp ins(%125 : memref<1xf32, #hivm.address_space<ub>>) outs(%126 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %127 = memref.load %126[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %128 = arith.mulf %127, %122 : f32
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%113, %128 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%113 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape = memref.expand_shape %113 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %129 = hivm.hir.pointer_cast(%c1216_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %130 = hivm.hir.pointer_cast(%c2240_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%114, %expand_shape : memref<16x1xf32, #hivm.address_space<ub>>, memref<1x16xf32, #hivm.address_space<ub>>) outs(%129 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%130 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [0, 1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %112]
      %collapse_shape_16 = memref.collapse_shape %arg37 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %129 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %33 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_16, %collapse_shape_17 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>>)
      %131 = arith.addi %arg38, %11 : i64
      %132 = arith.addi %arg39, %13 : i64
      %133 = arith.addi %arg40, %c1_i64 : i64
      %134 = arith.addi %arg41, %c1_i64 : i64
      scf.yield %33, %131, %132, %133, %134 : memref<16x16xf32, #hivm.address_space<ub>>, i64, i64, i64, i64
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %90 = arith.muli %42, %23 : i64
    %91 = arith.muli %40, %24 : i64
    %92 = arith.addi %90, %91 : i64
    %93 = arith.muli %34, %arg22 : i32
    %94 = arith.extsi %93 : i32 to i64
    %95 = arith.addi %92, %94 : i64
    %96 = arith.index_cast %95 : i64 to index
    %97 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%83, %96, %78, %25]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%97], sizes: [16, 16], strides: [%25, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %98 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%80, %78]
    %99 = affine.max affine_map<()[s0] -> (0, s0)>()[%98]
    %100 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%85, %83]
    %101 = affine.max affine_map<()[s0] -> (0, s0)>()[%100]
    %subview = memref.subview %89#0[0, 0] [%99, %101] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%99, %101] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %32]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
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
