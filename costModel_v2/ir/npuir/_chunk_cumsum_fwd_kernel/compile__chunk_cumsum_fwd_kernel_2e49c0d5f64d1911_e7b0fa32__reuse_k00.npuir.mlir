// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_cumsum_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: f32, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c10816_i64 = arith.constant 10816 : i64
  %c9152_i64 = arith.constant 9152 : i64
  %c8640_i64 = arith.constant 8640 : i64
  %c10752_i64 = arith.constant 10752 : i64
  %c8576_i64 = arith.constant 8576 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c8064_i64 = arith.constant 8064 : i64
  %c8032_i64 = arith.constant 8032 : i64
  %c7520_i64 = arith.constant 7520 : i64
  %c7456_i64 = arith.constant 7456 : i64
  %c6944_i64 = arith.constant 6944 : i64
  %c6432_i64 = arith.constant 6432 : i64
  %c6400_i64 = arith.constant 6400 : i64
  %c6368_i64 = arith.constant 6368 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c5792_i64 = arith.constant 5792 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c4768_i64 = arith.constant 4768 : i64
  %c4736_i64 = arith.constant 4736 : i64
  %c4704_i64 = arith.constant 4704 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c3680_i64 = arith.constant 3680 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c10176_i64 = arith.constant 10176 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c9664_i64 = arith.constant 9664 : i64
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
  %c1_i32 = arith.constant 1 : i32
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
  %7 = arith.addi %arg9, %arg11 : i32
  %8 = arith.subi %7, %c1_i32 : i32
  %9 = arith.index_cast %arg18 : i32 to index
  %10 = arith.index_cast %arg21 : i32 to index
  %11 = arith.index_cast %arg15 : i32 to index
  %12 = arith.index_cast %arg10 : i32 to index
  %13 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %13 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg12 : f32) outs(%collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %14 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_3 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %15 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<128xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg13 : f32) outs(%collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %17 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<128xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c1536_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %18 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_7 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<128xi1, #hivm.address_space<ub>>)
  %19 = arith.index_cast %arg11 : i32 to index
  %20 = affine.max affine_map<()[s0] -> (0, s0)>()[%19]
  %21 = affine.min affine_map<()[s0] -> (8, s0)>()[%20]
  %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg25 = %3 to %1 step %c48_i32  : i32 {
    %23 = arith.index_cast %arg25 : i32 to index
    %24 = arith.index_cast %3 : i32 to index
    %25 = arith.index_cast %1 : i32 to index
    %26 = arith.index_cast %c48_i32 : i32 to index
    %27 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%23, %24, %26]
    %28 = arith.index_cast %27 : index to i1
    %c0_i64_9 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %29 = arith.select %28, %c0_i64_9, %c1_i64 : i64
    %30 = hivm.hir.pointer_cast(%c9152_i64, %c10816_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c8576_i64, %c10752_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c8064_i64, %c10240_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c3072_i64, %c10176_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<16x1xf32, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c2048_i64, %c9664_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<8x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %35 = arith.divsi %arg25, %arg24 : i32
    %36 = arith.remsi %35, %arg23 : i32
    %37 = arith.divsi %arg25, %4 : i32
    %38 = arith.remsi %37, %arg22 : i32
    %39 = arith.divsi %8, %arg11 : i32
    %40 = arith.divsi %36, %39 : i32
    %41 = arith.remsi %36, %39 : i32
    %42 = arith.muli %38, %arg14 : i32
    %43 = arith.muli %40, %arg11 : i32
    %44 = arith.muli %43, %arg15 : i32
    %45 = arith.addi %42, %44 : i32
    %46 = arith.index_cast %45 : i32 to index
    %47 = arith.muli %38, %arg16 : i32
    %48 = arith.muli %40, %arg17 : i32
    %49 = arith.addi %47, %48 : i32
    %50 = arith.muli %38, %arg19 : i32
    %51 = arith.muli %40, %arg20 : i32
    %52 = arith.addi %50, %51 : i32
    %53 = arith.muli %41, %c16_i32 : i32
    %54 = arith.index_cast %53 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %55 = arith.index_cast %49 : i32 to index
    %56 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%55, %54, %9]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%56], sizes: [16, 8], strides: [%9, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %57 = arith.index_cast %52 : i32 to index
    %58 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%57, %54, %10]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%58], sizes: [16, 8], strides: [%10, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %59 = arith.subi %arg9, %43 : i32
    %60 = arith.minsi %arg11, %59 : i32
    %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%46, %54]
    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%61], sizes: [8, 16], strides: [%11, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %62 = arith.index_cast %60 : i32 to index
    %63 = affine.max affine_map<()[s0] -> (0, s0)>()[%62]
    %64 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%54, %12]
    %65 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%64, %54]
    %66 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%65, %54]
    %67 = affine.min affine_map<()[s0] -> (8, s0)>()[%63]
    %68 = affine.max affine_map<()[s0] -> (0, s0)>()[%67]
    %69 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%65, %54]
    %70 = affine.max affine_map<()[s0] -> (0, s0)>()[%69]
    %71 = arith.cmpi slt, %68, %c8 : index
    %72 = arith.cmpi slt, %70, %c16 : index
    %73 = arith.ori %71, %72 : i1
    %subview = memref.subview %reinterpret_cast_12[0, 0] [%68, %70] [1, 1] : memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_13 = memref.subview %34[0, 0] [%68, %70] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %73 {
      %collapse_shape_46 = memref.collapse_shape %34 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_46 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %74 = hivm.hir.pointer_cast(%c2560_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%34 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%74 : memref<16x8xf32, #hivm.address_space<ub>>) permutation = [1, 0]
    %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [16, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %75 = arith.cmpi slt, %66, %c16 : index
    %subview_15 = memref.subview %reinterpret_cast_14[0, 0] [%66, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_16 = memref.subview %33[0, 0] [%66, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %75 {
      %collapse_shape_46 = memref.collapse_shape %33 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_46 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_17 = memref.collapse_shape %subview_15 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_18 = memref.collapse_shape %subview_16 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %76 = hivm.hir.pointer_cast(%c2560_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %77 = hivm.hir.pointer_cast(%c3136_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%74, %33 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%76 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%77 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %78 = hivm.hir.pointer_cast(%c3648_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %76 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %78 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_19, %collapse_shape_2 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<128xi1, #hivm.address_space<ub>>) compare_mode = <le>
    %79 = hivm.hir.pointer_cast(%c3680_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %79 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_19 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
    %80 = hivm.hir.pointer_cast(%c3680_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %80 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_21, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>)
    %81 = hivm.hir.pointer_cast(%c3680_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %81 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
    %82 = hivm.hir.pointer_cast(%c4192_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %82 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %83 = hivm.hir.pointer_cast(%c4704_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_20, %collapse_shape_23, %collapse_shape_19 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%83 : memref<8xf32, #hivm.address_space<ub>>)
    %84 = hivm.hir.pointer_cast(%c4736_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %84 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_24, %collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<128xi1, #hivm.address_space<ub>>)
    %85 = hivm.hir.pointer_cast(%c4736_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %85 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_25 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<128xi1, #hivm.address_space<ub>>)
    %86 = hivm.hir.pointer_cast(%c4768_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %86 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_24, %arg12 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_27 : memref<128xf32, #hivm.address_space<ub>>)
    %87 = hivm.hir.pointer_cast(%c5280_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %87 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %88 = hivm.hir.pointer_cast(%c5792_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_26, %arg12, %collapse_shape_27 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%88 : memref<16xf32, #hivm.address_space<ub>>)
    %89 = hivm.hir.pointer_cast(%c5856_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %89 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %90 = hivm.hir.pointer_cast(%c6368_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_5, %collapse_shape_24, %collapse_shape_28 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%90 : memref<8xf32, #hivm.address_space<ub>>)
    %91 = hivm.hir.pointer_cast(%c6400_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %91 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_29, %collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<128xi1, #hivm.address_space<ub>>)
    %92 = hivm.hir.pointer_cast(%c6400_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %92 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_30 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<128xi1, #hivm.address_space<ub>>)
    %93 = hivm.hir.pointer_cast(%c6432_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %93 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_29, %arg13 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_32 : memref<128xf32, #hivm.address_space<ub>>)
    %94 = hivm.hir.pointer_cast(%c6944_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_33 = memref.collapse_shape %94 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %95 = hivm.hir.pointer_cast(%c7456_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_31, %arg13, %collapse_shape_32 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%95 : memref<16xf32, #hivm.address_space<ub>>)
    %96 = hivm.hir.pointer_cast(%c7520_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %96 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %97 = hivm.hir.pointer_cast(%c8032_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_8, %collapse_shape_29, %collapse_shape_33 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%97 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_35 = memref.subview %96[0, 0] [%70, %68] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %32 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, %29]
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_36 : memref<128xf32, #hivm.address_space<ub>>)
    %subview_37 = memref.subview %32[0, 0] [%70, %68] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_35 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_37 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_38 = memref.subview %32[0, 0] [%70, %22] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_39 = memref.subview %reinterpret_cast_10[0, 0] [%70, %22] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_38 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_39 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_40 = memref.subview %reinterpret_cast[0, 0] [%66, 1] [1, 1] : memref<16x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_41 = memref.subview %31[0, 0] [%66, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %75 {
      %collapse_shape_46 = memref.collapse_shape %31 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_46 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_42 = memref.collapse_shape %subview_40 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_43 = memref.collapse_shape %subview_41 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_42 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_43 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %98 = hivm.hir.pointer_cast(%c8640_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%32, %31 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%32 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%98 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg26 = %c0 to %c16 step %c1 {
      %99 = memref.load %32[%arg26, %c0] : memref<16x8xf32, #hivm.address_space<ub>>
      memref.store %99, %30[%arg26, %c0] : memref<16x8xf32, #hivm.address_space<ub>>
      scf.for %arg27 = %c1 to %c8 step %c1 {
        %100 = memref.load %32[%arg26, %arg27] : memref<16x8xf32, #hivm.address_space<ub>>
        %101 = arith.subi %arg27, %c1 : index
        %102 = memref.load %30[%arg26, %101] : memref<16x8xf32, #hivm.address_space<ub>>
        %103 = arith.addf %100, %102 : f32
        memref.store %103, %30[%arg26, %arg27] : memref<16x8xf32, #hivm.address_space<ub>>
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, %29]
    %subview_44 = memref.subview %30[0, 0] [%70, %22] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_45 = memref.subview %reinterpret_cast_11[0, 0] [%70, %22] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_44 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_45 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
