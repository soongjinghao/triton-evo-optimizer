// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_cumsum_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: f32, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c9728_i64 = arith.constant 9728 : i64
  %c11392_i64 = arith.constant 11392 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c8704_i64 = arith.constant 8704 : i64
  %c11328_i64 = arith.constant 11328 : i64
  %c8640_i64 = arith.constant 8640 : i64
  %c10816_i64 = arith.constant 10816 : i64
  %c8128_i64 = arith.constant 8128 : i64
  %c8096_i64 = arith.constant 8096 : i64
  %c7584_i64 = arith.constant 7584 : i64
  %c7520_i64 = arith.constant 7520 : i64
  %c7008_i64 = arith.constant 7008 : i64
  %c6496_i64 = arith.constant 6496 : i64
  %c6464_i64 = arith.constant 6464 : i64
  %c6432_i64 = arith.constant 6432 : i64
  %c5920_i64 = arith.constant 5920 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c5344_i64 = arith.constant 5344 : i64
  %c4832_i64 = arith.constant 4832 : i64
  %c4800_i64 = arith.constant 4800 : i64
  %c4768_i64 = arith.constant 4768 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c3712_i64 = arith.constant 3712 : i64
  %c3200_i64 = arith.constant 3200 : i64
  %c10752_i64 = arith.constant 10752 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst_0 = arith.constant 2.000000e+01 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg22, %arg23 : i32
  %1 = arith.muli %0, %arg24 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg24, %arg23 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %6 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_2 : memref<128xf32, #hivm.address_space<ub>>)
  %7 = arith.extsi %arg11 : i32 to i64
  %8 = arith.extsi %arg15 : i32 to i64
  %9 = arith.extsi %arg17 : i32 to i64
  %10 = arith.extsi %arg20 : i32 to i64
  %11 = arith.index_cast %arg18 : i32 to index
  %12 = arith.index_cast %arg21 : i32 to index
  %13 = arith.extsi %arg9 : i32 to i64
  %14 = arith.index_cast %arg15 : i32 to index
  %15 = arith.index_cast %arg10 : i32 to index
  %16 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg12 : f32) outs(%collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %17 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %18 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<128xi1, #hivm.address_space<ub>>)
  %19 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %19 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg13 : f32) outs(%collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>)
  %20 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %20 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<128xi1, #hivm.address_space<ub>>)
  %21 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %21 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_7 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<128xi1, #hivm.address_space<ub>>)
  %22 = arith.index_cast %arg11 : i32 to index
  %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
  %24 = affine.min affine_map<()[s0] -> (8, s0)>()[%23]
  %25 = affine.max affine_map<()[s0] -> (0, s0)>()[%24]
  %26 = hivm.hir.pointer_cast(%c2048_i64) : memref<16x1xf32, #hivm.address_space<ub>>
  hivm.hir.vreduce <sum> ins(%5 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%26 : memref<16x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
  scf.for %arg25 = %3 to %1 step %c48_i32  : i32 {
    %27 = arith.index_cast %arg25 : i32 to index
    %28 = arith.index_cast %3 : i32 to index
    %29 = arith.index_cast %1 : i32 to index
    %30 = arith.index_cast %c48_i32 : i32 to index
    %31 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%27, %28, %30]
    %32 = arith.index_cast %31 : index to i1
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %33 = arith.select %32, %c5_i64, %c6_i64 : i64
    %34 = arith.index_cast %31 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %35 = arith.select %34, %c3_i64, %c4_i64 : i64
    %36 = arith.index_cast %31 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %37 = arith.select %36, %c1_i64, %c2_i64 : i64
    %38 = hivm.hir.pointer_cast(%c9216_i64, %c11392_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c8640_i64, %c11328_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
    %40 = hivm.hir.pointer_cast(%c8128_i64, %c10816_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %41 = hivm.hir.pointer_cast(%c3136_i64, %c10752_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c2112_i64, %c10240_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<8x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %43 = arith.remsi %arg25, %arg24 : i32
    %44 = arith.divsi %arg25, %arg24 : i32
    %45 = arith.remsi %44, %arg23 : i32
    %46 = arith.divsi %arg25, %4 : i32
    %47 = arith.remsi %46, %arg22 : i32
    %48 = arith.extsi %45 : i32 to i64
    %49 = arith.muli %47, %arg14 : i32
    %50 = arith.muli %48, %7 : i64
    %51 = arith.muli %50, %8 : i64
    %52 = arith.extsi %49 : i32 to i64
    %53 = arith.addi %52, %51 : i64
    %54 = arith.index_cast %53 : i64 to index
    %55 = arith.muli %47, %arg16 : i32
    %56 = arith.muli %48, %9 : i64
    %57 = arith.extsi %55 : i32 to i64
    %58 = arith.addi %57, %56 : i64
    %59 = arith.muli %47, %arg19 : i32
    %60 = arith.muli %48, %10 : i64
    %61 = arith.extsi %59 : i32 to i64
    %62 = arith.addi %61, %60 : i64
    %63 = arith.muli %43, %c16_i32 : i32
    %64 = arith.index_cast %63 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %65 = arith.index_cast %58 : i64 to index
    %66 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%65, %64, %11]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%66], sizes: [16, 8], strides: [%11, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %67 = arith.index_cast %62 : i64 to index
    %68 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%67, %64, %12]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%68], sizes: [16, 8], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %69 = arith.subi %13, %50 : i64
    %70 = arith.minsi %7, %69 : i64
    %71 = arith.index_cast %70 : i64 to index
    %72 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%54, %64]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [8, 16], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %73 = affine.max affine_map<()[s0] -> (0, s0)>()[%71]
    %74 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%64, %15]
    %75 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%74, %64]
    %76 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%75, %64]
    %77 = affine.min affine_map<()[s0] -> (8, s0)>()[%73]
    %78 = affine.max affine_map<()[s0] -> (0, s0)>()[%77]
    %79 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%75, %64]
    %80 = affine.max affine_map<()[s0] -> (0, s0)>()[%79]
    %81 = arith.cmpi slt, %78, %c8 : index
    %82 = arith.cmpi slt, %80, %c16 : index
    %83 = arith.ori %81, %82 : i1
    %subview = memref.subview %reinterpret_cast_11[0, 0] [%78, %80] [1, 1] : memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %42[0, 0] [%78, %80] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %83 {
      %collapse_shape_55 = memref.collapse_shape %42 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_55 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %37]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_12 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %84 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%42 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%84 : memref<16x8xf32, #hivm.address_space<ub>>) permutation = [1, 0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %37]
    %reinterpret_cast_13 = memref.reinterpret_cast %arg5 to offset: [%64], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %85 = arith.cmpi slt, %76, %c16 : index
    %subview_14 = memref.subview %reinterpret_cast_13[0, 0] [%76, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_15 = memref.subview %41[0, 0] [%76, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %85 {
      %collapse_shape_55 = memref.collapse_shape %41 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_55 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_16 = memref.collapse_shape %subview_14 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_17 = memref.collapse_shape %subview_15 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %35]
    hivm.hir.load ins(%collapse_shape_16 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %86 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %87 = hivm.hir.pointer_cast(%c3200_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%84, %41 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%86 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%87 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %35]
    %88 = hivm.hir.pointer_cast(%c3712_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %86 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %88 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_18, %collapse_shape_2 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<128xi1, #hivm.address_space<ub>>) compare_mode = <le>
    %89 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %89 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_18 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
    %90 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %90 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_20, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
    %91 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %91 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>)
    %92 = hivm.hir.pointer_cast(%c4256_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %92 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %93 = hivm.hir.pointer_cast(%c4768_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_19, %collapse_shape_22, %collapse_shape_18 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%93 : memref<8xf32, #hivm.address_space<ub>>)
    %94 = hivm.hir.pointer_cast(%c4800_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %94 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_23, %collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<128xi1, #hivm.address_space<ub>>)
    %95 = hivm.hir.pointer_cast(%c4800_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %95 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_24 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xi1, #hivm.address_space<ub>>)
    %96 = hivm.hir.pointer_cast(%c4832_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %96 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_23, %arg12 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_26 : memref<128xf32, #hivm.address_space<ub>>)
    %97 = hivm.hir.pointer_cast(%c5344_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %97 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %98 = hivm.hir.pointer_cast(%c5856_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_25, %arg12, %collapse_shape_26 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%98 : memref<16xf32, #hivm.address_space<ub>>)
    %99 = hivm.hir.pointer_cast(%c5920_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %99 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %100 = hivm.hir.pointer_cast(%c6432_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_5, %collapse_shape_23, %collapse_shape_27 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%100 : memref<8xf32, #hivm.address_space<ub>>)
    %101 = hivm.hir.pointer_cast(%c6464_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %101 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_28, %collapse_shape_28 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<128xi1, #hivm.address_space<ub>>)
    %102 = hivm.hir.pointer_cast(%c6464_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %102 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_29 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<128xi1, #hivm.address_space<ub>>)
    %103 = hivm.hir.pointer_cast(%c6496_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %103 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_28, %arg13 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>)
    %104 = hivm.hir.pointer_cast(%c7008_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %104 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %105 = hivm.hir.pointer_cast(%c7520_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_30, %arg13, %collapse_shape_31 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%105 : memref<16xf32, #hivm.address_space<ub>>)
    %106 = hivm.hir.pointer_cast(%c7584_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_33 = memref.collapse_shape %106 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %107 = hivm.hir.pointer_cast(%c8096_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_8, %collapse_shape_28, %collapse_shape_32 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%107 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_34 = memref.subview %106[0, 0] [%80, %78] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_35 = memref.collapse_shape %40 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<128xf32, #hivm.address_space<ub>>)
    %subview_36 = memref.subview %40[0, 0] [%80, %78] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_34 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_36 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_37 = memref.subview %40[0, 0] [%80, %25] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_38 = memref.subview %reinterpret_cast_9[0, 0] [%80, %25] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_37 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_38 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_39 = memref.subview %reinterpret_cast[0, 0] [%76, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_40 = memref.subview %39[0, 0] [%76, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %85 {
      %collapse_shape_55 = memref.collapse_shape %39 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_55 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_41 = memref.collapse_shape %subview_39 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_42 = memref.collapse_shape %subview_40 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %33]
    hivm.hir.load ins(%collapse_shape_41 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_42 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %108 = hivm.hir.pointer_cast(%c8704_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%40, %39 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%40 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%108 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %33]
    %109 = hivm.hir.pointer_cast(%c9728_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%40, %26 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%38 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%109 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %subview_43 = memref.subview %40[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_44 = memref.subview %38[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %subview_43 [[0, 1]] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
    %collapse_shape_46 = memref.collapse_shape %subview_44 [[0, 1]] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %collapse_shape_45 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_47 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_48, %offset_49, %sizes_50, %strides_51 = memref.extract_strided_metadata %collapse_shape_46 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_52 = memref.reinterpret_cast %base_buffer_48 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%reinterpret_cast_47 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_52 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_53 = memref.subview %38[0, 0] [%80, %25] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_54 = memref.subview %reinterpret_cast_10[0, 0] [%80, %25] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_53 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_54 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
