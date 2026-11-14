// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c8640_i64 = arith.constant 8640 : i64
  %c7616_i64 = arith.constant 7616 : i64
  %c7552_i64 = arith.constant 7552 : i64
  %c7040_i64 = arith.constant 7040 : i64
  %c6016_i64 = arith.constant 6016 : i64
  %c11712_i64 = arith.constant 11712 : i64
  %c5952_i64 = arith.constant 5952 : i64
  %c5888_i64 = arith.constant 5888 : i64
  %c10688_i64 = arith.constant 10688 : i64
  %c4864_i64 = arith.constant 4864 : i64
  %c9664_i64 = arith.constant 9664 : i64
  %c3840_i64 = arith.constant 3840 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2688_i64 = arith.constant 2688 : i64
  %c1664_i64 = arith.constant 1664 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32
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
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x1xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %6 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%5 : memref<16xi32, #hivm.address_space<ub>>) outs(%collapse_shape : memref<16xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c0_i64) : memref<1x16xf32, #hivm.address_space<ub>>
  %collapse_shape_1 = memref.collapse_shape %7 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%5 : memref<16xi32, #hivm.address_space<ub>>) outs(%collapse_shape_1 : memref<16xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c128_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %9 = hivm.hir.pointer_cast(%c1152_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%6 : memref<16x1xf32, #hivm.address_space<ub>>) outs(%8 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%9 : memref<128xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %10 = hivm.hir.pointer_cast(%c1664_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%7 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%10 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %11 = hivm.hir.pointer_cast(%c2688_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %8 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %10 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %11 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_2, %collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>) compare_mode = <gt>
  %12 = hivm.hir.pointer_cast(%c1664_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %12 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_2, %collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<256xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %13 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %14 = hivm.hir.pointer_cast(%c3744_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_5 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%14 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %15 = arith.index_cast %arg9 : i32 to index
    %16 = arith.index_cast %3 : i32 to index
    %17 = arith.index_cast %1 : i32 to index
    %18 = arith.index_cast %c48_i32 : i32 to index
    %19 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%15, %16, %18]
    %20 = arith.index_cast %19 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %21 = arith.select %20, %c3_i64, %c4_i64 : i64
    %22 = arith.index_cast %19 : index to i1
    %c0_i64_7 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %23 = arith.select %22, %c0_i64_7, %c1_i64 : i64
    %24 = hivm.hir.pointer_cast(%c4864_i64, %c10688_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c3840_i64, %c9664_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %26 = arith.divsi %arg9, %arg8 : i32
    %27 = arith.remsi %26, %arg7 : i32
    %28 = arith.divsi %arg9, %4 : i32
    %29 = arith.remsi %28, %arg6 : i32
    %30 = arith.divsi %29, %c8_i32 : i32
    %31 = arith.remsi %29, %c8_i32 : i32
    %32 = arith.muli %30, %arg5 : i32
    %33 = arith.muli %32, %c8_i32 : i32
    %34 = arith.addi %33, %31 : i32
    %35 = arith.muli %34, %c16_i32 : i32
    %36 = arith.index_cast %35 : i32 to index
    %37 = arith.muli %27, %c16_i32 : i32
    %38 = arith.remsi %37, %c16_i32 : i32
    %39 = arith.maxsi %37, %c0_i32 : i32
    %40 = arith.index_cast %39 : i32 to index
    %41 = arith.maxsi %38, %c0_i32 : i32
    %42 = arith.index_cast %41 : i32 to index
    %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%36, %40]
    %44 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%42, %36, %40]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %21]
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%25 : memref<16x16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %collapse_shape_8 = memref.collapse_shape %25 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_9 = memref.collapse_shape %24 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c5888_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %23]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_8, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_9 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%45 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %21]
    %collapse_shape_10 = memref.collapse_shape %24 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_9, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_10 : memref<256xf32, #hivm.address_space<ub>>)
    %collapse_shape_11 = memref.collapse_shape %24 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_10, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_11 : memref<256xf32, #hivm.address_space<ub>>)
    %46 = arith.subi %arg5, %37 : i32
    %47 = arith.minsi %46, %c16_i32 : i32
    %48 = arith.index_cast %38 : i32 to index
    %49 = arith.index_cast %46 : i32 to index
    %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
    %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
    %52 = arith.cmpi slt, %51, %c16 : index
    scf.for %arg10 = %c2_i32 to %47 step %c1_i32  : i32 {
      %53 = arith.index_cast %arg10 : i32 to index
      %54 = arith.index_cast %c2_i32 : i32 to index
      %55 = arith.index_cast %47 : i32 to index
      %56 = arith.index_cast %c1_i32 : i32 to index
      %57 = arith.index_cast %arg9 : i32 to index
      %58 = arith.index_cast %3 : i32 to index
      %59 = arith.index_cast %1 : i32 to index
      %60 = arith.index_cast %c48_i32 : i32 to index
      %61 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%53, %54, %55, %56, %57, %58, %60]
      %62 = arith.index_cast %61 : index to i1
      %c1_i64_14 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %63 = arith.select %62, %c1_i64_14, %c2_i64 : i64
      %64 = hivm.hir.pointer_cast(%c5952_i64, %c11712_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %64 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %65 = arith.addi %37, %arg10 : i32
      %66 = arith.muli %65, %c128_i32 : i32
      %67 = arith.index_cast %66 : i32 to index
      %68 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%48, %36, %67]
      %reinterpret_cast_15 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast_15[0] [%51] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_16 = memref.subview %64[0] [%51] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %52 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%64 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %63]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%64, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%64 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%64, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%64 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape = memref.expand_shape %64 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %69 = hivm.hir.pointer_cast(%c6016_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %70 = hivm.hir.pointer_cast(%c7040_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape, %24 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%69 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %71 = hivm.hir.pointer_cast(%c7552_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %72 = hivm.hir.pointer_cast(%c7616_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%69 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%71 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%72 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_17 = memref.collapse_shape %71 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%64, %collapse_shape_17 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%64 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_18 = memref.expand_shape %64 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %73 = hivm.hir.pointer_cast(%c8640_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_18 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%73 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %63]
      %74 = arith.index_cast %arg10 : i32 to index
      %75 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%74]
      %76 = arith.cmpi slt, %74, %c0 : index
      %77 = arith.cmpi sge, %74, %c16 : index
      %78 = arith.ori %76, %77 : i1
      scf.if %78 {
      } else {
        %subview_19 = memref.subview %24[0, 0] [%74, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_20 = memref.subview %73[0, 0] [%74, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_21 = memref.collapse_shape %subview_19 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_22 = memref.collapse_shape %subview_20 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_23 = memref.subview %73[0, 0] [%75, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %subview_24 = memref.subview %24[0, 0] [%75, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %collapse_shape_25 = memref.collapse_shape %subview_23 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_26 = memref.collapse_shape %subview_24 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_25 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %collapse_shape_12 = memref.collapse_shape %24 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_11, %collapse_shape_6 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_13 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%24 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_13 : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %23]
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
