// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_quantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c27456_i64 = arith.constant 27456 : i64
  %c23328_i64 = arith.constant 23328 : i64
  %c23264_i64 = arith.constant 23264 : i64
  %c25408_i64 = arith.constant 25408 : i64
  %c21216_i64 = arith.constant 21216 : i64
  %c21184_i64 = arith.constant 21184 : i64
  %c19136_i64 = arith.constant 19136 : i64
  %c17088_i64 = arith.constant 17088 : i64
  %c17024_i64 = arith.constant 17024 : i64
  %c16960_i64 = arith.constant 16960 : i64
  %c16928_i64 = arith.constant 16928 : i64
  %c14880_i64 = arith.constant 14880 : i64
  %c14816_i64 = arith.constant 14816 : i64
  %c12768_i64 = arith.constant 12768 : i64
  %c10720_i64 = arith.constant 10720 : i64
  %c10656_i64 = arith.constant 10656 : i64
  %c10592_i64 = arith.constant 10592 : i64
  %c10336_i64 = arith.constant 10336 : i64
  %c10304_i64 = arith.constant 10304 : i64
  %c10272_i64 = arith.constant 10272 : i64
  %c9248_i64 = arith.constant 9248 : i64
  %c25376_i64 = arith.constant 25376 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c24352_i64 = arith.constant 24352 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c512 = arith.constant 512 : index
  %c512_i32 = arith.constant 512 : i32
  %cst_0 = arith.constant -4.480000e+02 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c4_i32 = arith.constant 4 : i32
  %cst_2 = arith.constant 4.480000e+02 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg13, %arg14 : i32
  %1 = arith.muli %0, %arg15 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg15, %arg14 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4x128xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<512xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c2048_i64) : memref<4x128xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %6 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_3 : memref<512xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg16 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg16 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c0_i64_4 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_4, %c1_i64 : i64
    %14 = arith.index_cast %11 : index to i1
    %c1_i64_5 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %15 = arith.select %14, %c1_i64_5, %c2_i64 : i64
    %16 = hivm.hir.pointer_cast(%c23328_i64, %c27456_i64) : memref<512xbf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<512xbf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c9216_i64, %c25376_i64) : memref<4xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<4xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c21216_i64, %c25408_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c4096_i64, %c24352_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4x128xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg16, %arg15 : i32
    %21 = arith.remsi %20, %arg14 : i32
    %22 = arith.divsi %arg16, %4 : i32
    %23 = arith.remsi %22, %arg13 : i32
    %24 = arith.cmpi slt, %21, %c1_i32 : i32
    scf.if %24 {
      %25 = arith.muli %21, %c4_i32 : i32
      %26 = arith.muli %21, %c512_i32 : i32
      %27 = arith.muli %23, %arg11 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %26 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%30], sizes: [512], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_6 = memref.collapse_shape %19 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
      %31 = affine.max affine_map<()[s0] -> (512, s0)>()[%29]
      %32 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%31, %29]
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %29]
      %34 = arith.cmpi slt, %33, %c512 : index
      %subview = memref.subview %reinterpret_cast[0] [%33] [1] : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %collapse_shape_6[0] [%33] [1] : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %34 {
        hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape_6 : memref<512xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %35 = hivm.hir.pointer_cast(%c5120_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_8 = memref.collapse_shape %35 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%collapse_shape_6 : memref<512xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
      %36 = hivm.hir.pointer_cast(%c7168_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_9 = memref.collapse_shape %36 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vabs ins(%collapse_shape_8 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<512xf32, #hivm.address_space<ub>>)
      %37 = hivm.hir.pointer_cast(%c9216_i64, %c25376_i64) : memref<4x1xf32, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c9248_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <max> ins(%36 : memref<4x128xf32, #hivm.address_space<ub>>) outs(%37 : memref<4x1xf32, #hivm.address_space<ub>>) temp_buffer(%38 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      %collapse_shape_10 = memref.collapse_shape %37 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
      %39 = hivm.hir.pointer_cast(%c9216_i64, %c25376_i64) : memref<4xf32, #hivm.address_space<ub>>
      %40 = hivm.hir.pointer_cast(%c10272_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vdiv ins(%collapse_shape_10, %cst_2 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%39 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%40 : memref<8xf32, #hivm.address_space<ub>>)
      %41 = hivm.hir.pointer_cast(%c10304_i64) : memref<4xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%41 : memref<4xf32, #hivm.address_space<ub>>)
      %42 = hivm.hir.pointer_cast(%c10304_i64) : memref<4xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vdiv ins(%41, %39 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%42 : memref<4xf32, #hivm.address_space<ub>>)
      %expand_shape = memref.expand_shape %42 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
      %43 = hivm.hir.pointer_cast(%c5120_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %44 = hivm.hir.pointer_cast(%c10336_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%35, %expand_shape : memref<4x128xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%43 : memref<4x128xf32, #hivm.address_space<ub>>) temp_buffer(%44 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
      %45 = hivm.hir.pointer_cast(%c10592_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_11 = memref.collapse_shape %43 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_12 = memref.collapse_shape %45 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_11, %collapse_shape_11 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<512xi1, #hivm.address_space<ub>>)
      %46 = hivm.hir.pointer_cast(%c10592_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_13 = memref.collapse_shape %46 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_12 : memref<512xi1, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<512xi1, #hivm.address_space<ub>>)
      %47 = hivm.hir.pointer_cast(%c10656_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %47 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_3 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<512xi1, #hivm.address_space<ub>>)
      %48 = hivm.hir.pointer_cast(%c10656_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %48 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_14 : memref<512xi1, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<512xi1, #hivm.address_space<ub>>)
      %49 = hivm.hir.pointer_cast(%c10720_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %49 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.vmin ins(%collapse_shape_11, %cst_2 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_16 : memref<512xf32, #hivm.address_space<ub>>)
      %50 = hivm.hir.pointer_cast(%c12768_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %50 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %51 = hivm.hir.pointer_cast(%c14816_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_13, %cst_2, %collapse_shape_16 : memref<512xi1, #hivm.address_space<ub>>, f32, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<512xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<16xf32, #hivm.address_space<ub>>)
      %52 = hivm.hir.pointer_cast(%c14880_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %52 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %53 = hivm.hir.pointer_cast(%c16928_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_15, %collapse_shape_11, %collapse_shape_17 : memref<512xi1, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<512xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<8xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c16960_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %54 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<512xi1, #hivm.address_space<ub>>)
      %55 = hivm.hir.pointer_cast(%c16960_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %55 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_19 : memref<512xi1, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<512xi1, #hivm.address_space<ub>>)
      %56 = hivm.hir.pointer_cast(%c17024_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %56 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.vcmp ins(%collapse_shape_18, %collapse_shape_18 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<512xi1, #hivm.address_space<ub>>)
      %57 = hivm.hir.pointer_cast(%c17024_i64) : memref<4x128xi1, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %57 [[0, 1]] : memref<4x128xi1, #hivm.address_space<ub>> into memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_21 : memref<512xi1, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<512xi1, #hivm.address_space<ub>>)
      %58 = hivm.hir.pointer_cast(%c17088_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %58 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.vmax ins(%collapse_shape_18, %cst_0 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_23 : memref<512xf32, #hivm.address_space<ub>>)
      %59 = hivm.hir.pointer_cast(%c19136_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_24 = memref.collapse_shape %59 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %60 = hivm.hir.pointer_cast(%c21184_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_20, %collapse_shape_18, %collapse_shape_23 : memref<512xi1, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<512xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<8xf32, #hivm.address_space<ub>>)
      %61 = hivm.hir.pointer_cast(%c21216_i64, %c25408_i64) : memref<4x128xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %61 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c23264_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_22, %cst_0, %collapse_shape_24 : memref<512xi1, #hivm.address_space<ub>>, f32, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<512xf32, #hivm.address_space<ub>>) temp_buffer(%62 : memref<16xf32, #hivm.address_space<ub>>)
      %collapse_shape_26 = memref.collapse_shape %18 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%collapse_shape_25 : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<512xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %63 = arith.muli %23, %arg8 : i32
      %64 = arith.index_cast %63 : i32 to index
      %65 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%64, %29]
      %reinterpret_cast_27 = memref.reinterpret_cast %arg3 to offset: [%65], sizes: [512], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %collapse_shape_26[0] [%33] [1] : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      %subview_29 = memref.subview %reinterpret_cast_27[0] [%33] [1] : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_28 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_29 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      %66 = arith.muli %23, %arg9 : i32
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %25 : i32 to index
      %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%67, %68]
      %reinterpret_cast_30 = memref.reinterpret_cast %arg4 to offset: [%69], sizes: [4], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.vcast ins(%39 : memref<4xf32, #hivm.address_space<ub>>) outs(%17 : memref<4xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %70 = affine.max affine_map<()[s0] -> (4, s0)>()[%68]
      %71 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%70, %68]
      %72 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%71, %68]
      %subview_31 = memref.subview %17[0] [%72] [1] : memref<4xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      %subview_32 = memref.subview %reinterpret_cast_30[0] [%72] [1] : memref<4xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_31 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_32 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    } else {
      %25 = arith.subi %21, %c1_i32 : i32
      %26 = arith.muli %25, %c512_i32 : i32
      %27 = arith.muli %23, %arg12 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %26 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%30], sizes: [512], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.muli %23, %arg10 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%32, %29]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [512], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max affine_map<()[s0] -> (64, s0)>()[%29]
      %35 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%34, %29]
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %29]
      %37 = arith.cmpi slt, %36, %c512 : index
      %subview = memref.subview %reinterpret_cast[0] [%36] [1] : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %16[0] [%36] [1] : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %37 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : bf16) outs(%16 : memref<512xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_8 = memref.subview %reinterpret_cast_6[0] [%36] [1] : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_7 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
