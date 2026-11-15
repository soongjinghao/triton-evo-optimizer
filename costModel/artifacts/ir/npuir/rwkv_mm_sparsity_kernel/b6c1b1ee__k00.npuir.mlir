// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @rwkv_mm_sparsity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c124256_i64 = arith.constant 124256 : i64
  %c36864_i64 = arith.constant 36864 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c107872_i64 = arith.constant 107872 : i64
  %c91488_i64 = arith.constant 91488 : i64
  %c90464_i64 = arith.constant 90464 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c92448_i64 = arith.constant 92448 : i64
  %c90400_i64 = arith.constant 90400 : i64
  %c69696_i64 = arith.constant 69696 : i64
  %c69632_i64 = arith.constant 69632 : i64
  %c74016_i64 = arith.constant 74016 : i64
  %c74112_i64 = arith.constant 74112 : i64
  %c74048_i64 = arith.constant 74048 : i64
  %c73984_i64 = arith.constant 73984 : i64
  %c71936_i64 = arith.constant 71936 : i64
  %c71872_i64 = arith.constant 71872 : i64
  %c71808_i64 = arith.constant 71808 : i64
  %c69760_i64 = arith.constant 69760 : i64
  %c512_i32 = arith.constant 512 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c69760_i64) : memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<512xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c71808_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%6 : memref<16xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c71872_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%7 : memref<16xi32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c71936_i64) : memref<512xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%8 : memref<512xi32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %9 = arith.index_cast %arg9 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %1 : i32 to index
    %12 = arith.index_cast %c48_i32 : i32 to index
    %13 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%9, %10, %12]
    %14 = arith.index_cast %13 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_1, %c1_i64 : i64
    %16 = hivm.hir.pointer_cast(%c69632_i64, %c69696_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg9, %4 : i32
    %18 = arith.remsi %17, %arg6 : i32
    %19 = arith.muli %18, %c16_i32 : i32
    %20 = hivm.hir.pointer_cast(%c73984_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%7, %19 : memref<16xi32, #hivm.address_space<ub>>, i32) outs(%20 : memref<16xi32, #hivm.address_space<ub>>)
    %21 = hivm.hir.pointer_cast(%c74048_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmax ins(%20, %c512_i32 : memref<16xi32, #hivm.address_space<ub>>, i32) outs(%21 : memref<16xi32, #hivm.address_space<ub>>)
    %22 = hivm.hir.pointer_cast(%c74048_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%21, %20 : memref<16xi32, #hivm.address_space<ub>>, memref<16xi32, #hivm.address_space<ub>>) outs(%22 : memref<16xi1, #hivm.address_space<ub>>)
    %23 = hivm.hir.pointer_cast(%c74048_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%22 : memref<16xi1, #hivm.address_space<ub>>) outs(%23 : memref<16xi1, #hivm.address_space<ub>>)
    %24 = hivm.hir.pointer_cast(%c73984_i64) : memref<16xf16, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c74112_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%23 : memref<16xi1, #hivm.address_space<ub>>) outs(%24 : memref<16xf16, #hivm.address_space<ub>>) temp_buffer(%25 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %expand_shape = memref.expand_shape %24 [[0, 1]] output_shape [1, 16] : memref<16xf16, #hivm.address_space<ub>> into memref<1x16xf16, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c74016_i64) : memref<512x16xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape : memref<1x16xf16, #hivm.address_space<ub>>) outs(%26 : memref<512x16xf16, #hivm.address_space<ub>>) broadcast_dims = [0]
    %27 = hivm.hir.pointer_cast(%c74016_i64) : memref<512x16xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %26 [[0, 1]] : memref<512x16xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
    %collapse_shape_2 = memref.collapse_shape %27 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_2 : memref<8192xi1, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c74016_i64) : memref<512x16xi1, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %28 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_2 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<8192xi1, #hivm.address_space<ub>>)
    %29 = arith.index_cast %19 : i32 to index
    %30 = affine.max affine_map<()[s0] -> (512, s0)>()[%29]
    %31 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%30, %29]
    %32 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%31, %29]
    %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%32]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    hivm.hir.copy ins(%6 : memref<16xf32, #hivm.address_space<ub>>) outs(%16 : memref<16xf32, #hivm.address_space<ub>>)
    %34 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %16) -> (memref<16xf32, #hivm.address_space<ub>>)  : i32 {
      %36 = arith.index_cast %arg10 : i32 to index
      %37 = arith.index_cast %c0_i32 : i32 to index
      %38 = arith.index_cast %c2_i32 : i32 to index
      %39 = arith.index_cast %c1_i32 : i32 to index
      %40 = arith.index_cast %arg9 : i32 to index
      %41 = arith.index_cast %3 : i32 to index
      %42 = arith.index_cast %1 : i32 to index
      %43 = arith.index_cast %c48_i32 : i32 to index
      %44 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%36, %37, %38, %39, %40, %41, %43]
      %45 = arith.index_cast %44 : index to i1
      %c1_i64_5 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %46 = arith.select %45, %c1_i64_5, %c2_i64 : i64
      %47 = hivm.hir.pointer_cast(%c4096_i64, %c36864_i64) : memref<512x16xf32, #hivm.address_space<ub>>
      annotation.mark %47 {hivm.multi_buffer = 2 : i32} : memref<512x16xf32, #hivm.address_space<ub>>
      %48 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<512x1xf32, #hivm.address_space<ub>>
      annotation.mark %48 {hivm.multi_buffer = 2 : i32} : memref<512x1xf32, #hivm.address_space<ub>>
      %49 = arith.muli %arg10, %c512_i32 : i32
      %50 = hivm.hir.pointer_cast(%c90400_i64) : memref<512xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%8, %49 : memref<512xi32, #hivm.address_space<ub>>, i32) outs(%50 : memref<512xi32, #hivm.address_space<ub>>)
      %51 = hivm.hir.pointer_cast(%c92448_i64) : memref<512xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmax ins(%50, %c1024_i32 : memref<512xi32, #hivm.address_space<ub>>, i32) outs(%51 : memref<512xi32, #hivm.address_space<ub>>)
      %52 = hivm.hir.pointer_cast(%c92448_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%51, %50 : memref<512xi32, #hivm.address_space<ub>>, memref<512xi32, #hivm.address_space<ub>>) outs(%52 : memref<512xi1, #hivm.address_space<ub>>)
      %53 = hivm.hir.pointer_cast(%c92448_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%52 : memref<512xi1, #hivm.address_space<ub>>) outs(%53 : memref<512xi1, #hivm.address_space<ub>>)
      %54 = arith.index_cast %49 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [512, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<512x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %55 = affine.max affine_map<()[s0] -> (1024, s0)>()[%54]
      %56 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%55, %54]
      %57 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%56, %54]
      %58 = arith.cmpi slt, %57, %c512 : index
      %subview_7 = memref.subview %reinterpret_cast_6[0, 0] [%57, 1] [1, 1] : memref<512x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %48[0, 0] [%57, 1] [1, 1] : memref<512x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %58 {
        %collapse_shape_28 = memref.collapse_shape %48 [[0, 1]] : memref<512x1xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_28 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_9 = memref.collapse_shape %subview_7 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_10 = memref.collapse_shape %subview_8 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %46]
      hivm.hir.load ins(%collapse_shape_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_11 = memref.collapse_shape %48 [[0, 1]] : memref<512x1xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %59 = hivm.hir.pointer_cast(%c90400_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcmp ins(%collapse_shape_11, %5 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%59 : memref<512xi1, #hivm.address_space<ub>>)
      %60 = hivm.hir.pointer_cast(%c90400_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%59 : memref<512xi1, #hivm.address_space<ub>>) outs(%60 : memref<512xi1, #hivm.address_space<ub>>)
      %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%29, %54]
      %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [512, 16], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<512x16xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %62 = hivm.hir.pointer_cast(%c90464_i64) : memref<512xf16, #hivm.address_space<ub>>
      %63 = hivm.hir.pointer_cast(%c91488_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%53 : memref<512xi1, #hivm.address_space<ub>>) outs(%62 : memref<512xf16, #hivm.address_space<ub>>) temp_buffer(%63 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %expand_shape_13 = memref.expand_shape %62 [[0, 1]] output_shape [512, 1] : memref<512xf16, #hivm.address_space<ub>> into memref<512x1xf16, #hivm.address_space<ub>>
      %64 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xf16, #hivm.address_space<ub>>
      %65 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_13 : memref<512x1xf16, #hivm.address_space<ub>>) outs(%64 : memref<512x16xf16, #hivm.address_space<ub>>) temp_buffer(%65 : memref<0xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
      %66 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %64 [[0, 1]] : memref<512x16xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %66 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_14, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>)
      %67 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %67 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8192xi1, #hivm.address_space<ub>>)
      %68 = hivm.hir.pointer_cast(%c90464_i64) : memref<512xf16, #hivm.address_space<ub>>
      %69 = hivm.hir.pointer_cast(%c107872_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%60 : memref<512xi1, #hivm.address_space<ub>>) outs(%68 : memref<512xf16, #hivm.address_space<ub>>) temp_buffer(%69 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %expand_shape_17 = memref.expand_shape %68 [[0, 1]] output_shape [512, 1] : memref<512xf16, #hivm.address_space<ub>> into memref<512x1xf16, #hivm.address_space<ub>>
      %70 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xf16, #hivm.address_space<ub>>
      %71 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_17 : memref<512x1xf16, #hivm.address_space<ub>>) outs(%70 : memref<512x16xf16, #hivm.address_space<ub>>) temp_buffer(%71 : memref<0xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
      %72 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %70 [[0, 1]] : memref<512x16xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %72 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_18, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_19 : memref<8192xi1, #hivm.address_space<ub>>)
      %73 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %73 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_19 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<8192xi1, #hivm.address_space<ub>>)
      %74 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %74 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.vand ins(%collapse_shape_16, %collapse_shape_3 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<8192xi1, #hivm.address_space<ub>>)
      %75 = affine.min affine_map<()[s0, s1] -> (512, s0 - s1)>()[%56, %54]
      %76 = affine.max affine_map<()[s0] -> (0, s0)>()[%75]
      %subview_22 = memref.subview %reinterpret_cast_12[0, 0] [%76, %33] [1, 1] : memref<512x16xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_23 = memref.subview %47[0, 0] [%76, %33] [1, 1] : memref<512x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_22 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_23 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %77 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_24 = memref.collapse_shape %77 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vand ins(%collapse_shape_21, %collapse_shape_20 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<8192xi1, #hivm.address_space<ub>>)
      %78 = hivm.hir.pointer_cast(%c124256_i64) : memref<512x16xf32, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %47 [[0, 1]] : memref<512x16xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %collapse_shape_26 = memref.collapse_shape %78 [[0, 1]] : memref<512x16xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %79 = hivm.hir.pointer_cast(%c90400_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_24, %collapse_shape_25, %cst : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_26 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%79 : memref<16xf32, #hivm.address_space<ub>>)
      %80 = hivm.hir.pointer_cast(%c124256_i64) : memref<512x16xf32, #hivm.address_space<ub>>
      %81 = hivm.hir.pointer_cast(%c90400_i64) : memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%48, %78 : memref<512x1xf32, #hivm.address_space<ub>>, memref<512x16xf32, #hivm.address_space<ub>>) outs(%80 : memref<512x16xf32, #hivm.address_space<ub>>) temp_buffer(%81 : memref<4096xf32, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %46]
      %82 = hivm.hir.pointer_cast(%c90400_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %83 = hivm.hir.pointer_cast(%c90464_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%80 : memref<512x16xf32, #hivm.address_space<ub>>) outs(%82 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%83 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_27 = memref.collapse_shape %82 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg11, %collapse_shape_27 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%16 : memref<16xf32, #hivm.address_space<ub>>)
      scf.yield %16 : memref<16xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%31, %29]
    %subview = memref.subview %34[0] [%35] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %reinterpret_cast[0] [%35] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
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
