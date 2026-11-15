// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_64x64_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c8544_i64 = arith.constant 8544 : i64
  %c8032_i64 = arith.constant 8032 : i64
  %c7520_i64 = arith.constant 7520 : i64
  %c7008_i64 = arith.constant 7008 : i64
  %c6496_i64 = arith.constant 6496 : i64
  %c5984_i64 = arith.constant 5984 : i64
  %c5472_i64 = arith.constant 5472 : i64
  %c5440_i64 = arith.constant 5440 : i64
  %c4928_i64 = arith.constant 4928 : i64
  %c4896_i64 = arith.constant 4896 : i64
  %c4384_i64 = arith.constant 4384 : i64
  %c3872_i64 = arith.constant 3872 : i64
  %c3808_i64 = arith.constant 3808 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c3232_i64 = arith.constant 3232 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c576_i64 = arith.constant 576 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c15360 = arith.constant 15360 : index
  %c13312 = arith.constant 13312 : index
  %c11264 = arith.constant 11264 : index
  %c10240 = arith.constant 10240 : index
  %c8192 = arith.constant 8192 : index
  %c6144 = arith.constant 6144 : index
  %c5120 = arith.constant 5120 : index
  %c3072 = arith.constant 3072 : index
  %c2048 = arith.constant 2048 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %c32_i32 = arith.constant 32 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %c256_i32 = arith.constant 256 : i32
  %c-16_i32 = arith.constant -16 : i32
  %c-32_i32 = arith.constant -32 : i32
  %c-48_i32 = arith.constant -48 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c576_i64) : memref<0xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%6 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%7 : memref<0xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %8 = arith.index_cast %arg5 : i32 to index
  %9 = hivm.hir.get_sub_block_idx -> i64
  %10 = arith.index_cast %9 : i64 to index
  %11 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%10]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
  %subview = memref.subview %5[%11] [8] [1] {to_be_bubbled_slice} : memref<16xi32, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %expand_shape_1 = memref.expand_shape %subview [[0, 1]] output_shape [1, 8] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
  %12 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%12 : memref<16x8xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %13 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %14 = hivm.hir.pointer_cast(%c1600_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%13 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%14 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %15 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %13 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %12 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %15 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<128xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %6 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %17 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c2112_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %18 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  %19 = hivm.hir.pointer_cast(%c2624_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_6 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%19 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %20 = arith.remsi %arg9, %arg8 : i32
    %21 = arith.divsi %arg9, %arg8 : i32
    %22 = arith.remsi %21, %arg7 : i32
    %23 = arith.divsi %arg9, %4 : i32
    %24 = arith.remsi %23, %arg6 : i32
    %25 = arith.divsi %22, %c4_i32 : i32
    %26 = arith.remsi %22, %c4_i32 : i32
    %27 = arith.muli %25, %arg5 : i32
    %28 = arith.muli %27, %c4_i32 : i32
    %29 = arith.addi %28, %26 : i32
    %30 = arith.muli %29, %c64_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = arith.cmpi eq, %20, %c0_i32 : i32
    %33 = scf.if %32 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %76 = arith.cmpi eq, %20, %c1_i32 : i32
      %77 = scf.if %76 -> (i32) {
        scf.yield %c16_i32 : i32
      } else {
        %78 = arith.cmpi eq, %20, %c2_i32 : i32
        %79 = arith.select %78, %c32_i32, %c48_i32 : i32
        scf.yield %79 : i32
      }
      scf.yield %77 : i32
    }
    %34 = arith.muli %24, %c64_i32 : i32
    %35 = arith.addi %34, %33 : i32
    %36 = arith.maxsi %35, %c0_i32 : i32
    %37 = arith.index_cast %36 : i32 to index
    %38 = arith.maxsi %33, %c0_i32 : i32
    %39 = arith.index_cast %38 : i32 to index
    %40 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 256) * 256 + 64, 0)>()[%39]
    %41 = affine.min affine_map<()[s0] -> (16, s0)>()[%40]
    %42 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 256)>()[%39, %31, %37]
    %43 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 256, 0)>()[%8, %39, %37]
    %44 = affine.min affine_map<()[s0] -> (16, s0)>()[%43]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %45 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %46 = arith.subi %c0_i32, %35 : i32
    %47 = arith.maxsi %46, %c0_i32 : i32
    %48 = arith.index_cast %47 : i32 to index
    %49 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%48, %43]
    %50 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%44, %49]
    %51 = arith.subi %c0_i32, %33 : i32
    %52 = arith.maxsi %51, %c0_i32 : i32
    %53 = arith.index_cast %52 : i32 to index
    %54 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%53, %40]
    %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%41, %54]
    %56 = arith.cmpi slt, %50, %c16 : index
    %57 = arith.cmpi slt, %55, %c16 : index
    %58 = arith.ori %56, %57 : i1
    %59 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %41, %54]
    %60 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%59, %41, %54]
    %subview_8 = memref.subview %reinterpret_cast[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %subview_8[0, 0] [%50, %60] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %61 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%54, %10]
    %62 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%40, %10]
    %63 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%62, %54, %10]
    %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%63, %61]
    %65 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%61, %10]
    %subview_10 = memref.subview %45[%49, %65] [%50, %64] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %58 {
      %collapse_shape_20 = memref.collapse_shape %45 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_20 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%subview_9 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %54 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %66 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %45 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %66 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c3744_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_11, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_12 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %68 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_12, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_13 : memref<128xf32, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_13, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_14 : memref<128xf32, #hivm.address_space<ub>>)
    %70 = arith.addi %33, %c2_i32 : i32
    %71 = arith.addi %33, %c16_i32 : i32
    %72 = arith.subi %arg5, %34 : i32
    %73 = arith.minsi %71, %72 : i32
    %74 = arith.index_cast %33 : i32 to index
    scf.for %arg10 = %70 to %73 step %c1_i32  : i32 {
      %76 = arith.addi %34, %arg10 : i32
      %77 = arith.muli %76, %c256_i32 : i32
      %78 = arith.index_cast %77 : i32 to index
      %79 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%74, %31, %78]
      %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%79], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %80 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_20 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%80 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %81 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%80, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%81 : memref<16xf32, #hivm.address_space<ub>>)
      %82 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%81, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%82 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_21 = memref.expand_shape %82 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %83 = hivm.hir.pointer_cast(%c3872_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %84 = hivm.hir.pointer_cast(%c4384_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_21, %69 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%83 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%84 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %85 = hivm.hir.pointer_cast(%c4896_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %86 = hivm.hir.pointer_cast(%c4928_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%83 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%85 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%86 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_22 = memref.collapse_shape %85 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_23 = memref.subview %82[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %87 = hivm.hir.pointer_cast(%c5440_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_23, %collapse_shape_22 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%87 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %88 = arith.subi %arg10, %33 : i32
      %expand_shape_24 = memref.expand_shape %87 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %89 = hivm.hir.pointer_cast(%c5472_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_24 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%89 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %90 = arith.index_cast %88 : i32 to index
      %91 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%90]
      %92 = arith.cmpi slt, %90, %c0 : index
      %93 = arith.cmpi sge, %90, %c16 : index
      %94 = arith.ori %92, %93 : i1
      scf.if %94 {
      } else {
        %subview_25 = memref.subview %69[0, 0] [%90, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_26 = memref.subview %89[0, 0] [%90, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_27 = memref.collapse_shape %subview_25 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_28 = memref.collapse_shape %subview_26 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_27 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_29 = memref.subview %89[0, 0] [%91, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_30 = memref.subview %69[0, 0] [%91, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_31 = memref.collapse_shape %subview_29 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_32 = memref.collapse_shape %subview_30 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_31 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %75 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %75 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_14, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %reinterpret_cast_16[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %75[%49, %65] [%50, %64] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_19 = memref.subview %subview_17[0, 0] [%50, %60] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_18 : memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_19 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10
    scf.if %32 {
      %76 = arith.addi %34, %c16_i32 : i32
      %77 = arith.maxsi %76, %c0_i32 : i32
      %78 = arith.index_cast %77 : i32 to index
      %79 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%31, %78]
      %80 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %78]
      %81 = affine.min affine_map<()[s0] -> (16, s0)>()[%80]
      %82 = arith.addi %34, %c32_i32 : i32
      %83 = arith.maxsi %82, %c0_i32 : i32
      %84 = arith.index_cast %83 : i32 to index
      %85 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%31, %84]
      %86 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%31, %84]
      %87 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %84]
      %88 = affine.min affine_map<()[s0] -> (16, s0)>()[%87]
      %89 = arith.addi %34, %c48_i32 : i32
      %90 = arith.maxsi %89, %c0_i32 : i32
      %91 = arith.index_cast %90 : i32 to index
      %92 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%31, %91]
      %93 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%31, %91]
      %94 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%31, %91]
      %95 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %91]
      %96 = affine.min affine_map<()[s0] -> (16, s0)>()[%95]
      %97 = arith.subi %c-16_i32, %34 : i32
      %98 = arith.maxsi %97, %c0_i32 : i32
      %99 = arith.index_cast %98 : i32 to index
      %100 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%99, %80]
      %101 = arith.subi %c-32_i32, %34 : i32
      %102 = arith.maxsi %101, %c0_i32 : i32
      %103 = arith.index_cast %102 : i32 to index
      %104 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%103, %87]
      %105 = arith.subi %c-48_i32, %34 : i32
      %106 = arith.maxsi %105, %c0_i32 : i32
      %107 = arith.index_cast %106 : i32 to index
      %108 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%107, %95]
      %109 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_20 = memref.subview %109[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %110 = hivm.hir.pointer_cast(%c5984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_22 = memref.collapse_shape %110 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%collapse_shape_21 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
      %111 = hivm.hir.pointer_cast(%c5984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %111 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_22, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
      %112 = hivm.hir.pointer_cast(%c5984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_24 = memref.collapse_shape %112 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_23, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %113 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_25 = memref.subview %113[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
      %collapse_shape_26 = memref.collapse_shape %subview_25 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_25 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %114 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_27 = memref.subview %114[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
      %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_27 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %115 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_29 = memref.subview %115[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %116 = hivm.hir.pointer_cast(%c6496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %subview_29 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_31 = memref.collapse_shape %116 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%collapse_shape_30 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_31 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
      %117 = hivm.hir.pointer_cast(%c6496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_32 = memref.collapse_shape %117 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_31, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_32 : memref<128xf32, #hivm.address_space<ub>>)
      %118 = hivm.hir.pointer_cast(%c6496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_33 = memref.collapse_shape %118 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_32, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_33 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %119 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_34 = memref.subview %119[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_33 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_34 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %120 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_36 = memref.subview %120[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %121 = hivm.hir.pointer_cast(%c7008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_38 = memref.collapse_shape %121 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%collapse_shape_37 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_38 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      %122 = hivm.hir.pointer_cast(%c7008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_39 = memref.collapse_shape %122 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_38, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_39 : memref<128xf32, #hivm.address_space<ub>>)
      %123 = hivm.hir.pointer_cast(%c7008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_40 = memref.collapse_shape %123 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_39, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_40 : memref<128xf32, #hivm.address_space<ub>>)
      %124 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_41 = memref.subview %124[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %125 = hivm.hir.pointer_cast(%c7520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_42 = memref.collapse_shape %subview_41 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_43 = memref.collapse_shape %125 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%collapse_shape_42 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_43 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      %126 = hivm.hir.pointer_cast(%c7520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_44 = memref.collapse_shape %126 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_43, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>)
      %127 = hivm.hir.pointer_cast(%c7520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_45 = memref.collapse_shape %127 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_44, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_45 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %128 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      %subview_46 = memref.subview %128[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      %collapse_shape_47 = memref.collapse_shape %subview_46 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%collapse_shape_45 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_47 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
      annotation.mark %subview_46 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %129 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_48 = memref.subview %129[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %130 = hivm.hir.pointer_cast(%c8032_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_49 = memref.collapse_shape %subview_48 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_50 = memref.collapse_shape %130 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%collapse_shape_49 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_50 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      %131 = hivm.hir.pointer_cast(%c8032_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_51 = memref.collapse_shape %131 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_50, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_51 : memref<128xf32, #hivm.address_space<ub>>)
      %132 = hivm.hir.pointer_cast(%c8032_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_52 = memref.collapse_shape %132 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_51, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_52 : memref<128xf32, #hivm.address_space<ub>>)
      %133 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %subview_53 = memref.subview %133[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %134 = hivm.hir.pointer_cast(%c8544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_54 = memref.collapse_shape %subview_53 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_55 = memref.collapse_shape %134 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%collapse_shape_54 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_55 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      %135 = hivm.hir.pointer_cast(%c8544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_56 = memref.collapse_shape %135 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_55, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_56 : memref<128xf32, #hivm.address_space<ub>>)
      %136 = hivm.hir.pointer_cast(%c8544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      %collapse_shape_57 = memref.collapse_shape %136 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_56, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_57 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_58 = memref.reinterpret_cast %arg4 to offset: [%79], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_59 = memref.subview %reinterpret_cast_58[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_60 = memref.reinterpret_cast %arg4 to offset: [%85], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_61 = memref.subview %reinterpret_cast_60[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_62 = memref.reinterpret_cast %arg4 to offset: [%86], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_63 = memref.subview %reinterpret_cast_62[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_64 = memref.reinterpret_cast %arg4 to offset: [%92], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_65 = memref.subview %reinterpret_cast_64[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_66 = memref.reinterpret_cast %arg4 to offset: [%93], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_67 = memref.subview %reinterpret_cast_66[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_68 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_69 = memref.subview %reinterpret_cast_68[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %137 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%100, %10]
      %138 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%80, %10]
      %139 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%138, %100, %10]
      %140 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%139, %137]
      %141 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%137, %10]
      %subview_70 = memref.subview %112[%141, 0] [%140, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %142 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %81, %100]
      %143 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%142, %81, %100]
      %subview_71 = memref.subview %subview_59[0, 0] [%143, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_70 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_71 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      %144 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%104, %10]
      %145 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%87, %10]
      %146 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%145, %104, %10]
      %147 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%146, %144]
      %148 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%144, %10]
      %subview_72 = memref.subview %127[%148, 0] [%147, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %149 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %88, %104]
      %150 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%149, %88, %104]
      %subview_73 = memref.subview %subview_61[0, 0] [%150, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_72 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_73 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      %subview_74 = memref.subview %118[%148, 0] [%147, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_75 = memref.subview %subview_63[0, 0] [%150, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_74 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_75 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      %151 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%108, %10]
      %152 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%95, %10]
      %153 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%152, %108, %10]
      %154 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%153, %151]
      %155 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%151, %10]
      %subview_76 = memref.subview %136[%155, 0] [%154, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %156 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %96, %108]
      %157 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%156, %96, %108]
      %subview_77 = memref.subview %subview_65[0, 0] [%157, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_76 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_77 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      %subview_78 = memref.subview %132[%155, 0] [%154, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_79 = memref.subview %subview_67[0, 0] [%157, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_78 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_79 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      %subview_80 = memref.subview %123[%155, 0] [%154, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_81 = memref.subview %subview_69[0, 0] [%157, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_80 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_81 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    }
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_64x64_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c0_i64_2 = arith.constant 0 : i64
  %c0_i64_3 = arith.constant 0 : i64
  %c0_i64_4 = arith.constant 0 : i64
  %c0_i64_5 = arith.constant 0 : i64
  %c39936_i64 = arith.constant 39936 : i64
  %c19456_i64 = arith.constant 19456 : i64
  %c38912_i64 = arith.constant 38912 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c37888_i64 = arith.constant 37888 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c36864_i64 = arith.constant 36864 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c35840_i64 = arith.constant 35840 : i64
  %c15360_i64 = arith.constant 15360 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c14336_i64 = arith.constant 14336 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c13312_i64 = arith.constant 13312 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c31744_i64 = arith.constant 31744 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c30720_i64 = arith.constant 30720 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c29696_i64 = arith.constant 29696 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c28672_i64 = arith.constant 28672 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c27648_i64 = arith.constant 27648 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c26624_i64 = arith.constant 26624 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c25600_i64 = arith.constant 25600 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c23552_i64 = arith.constant 23552 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c22528_i64 = arith.constant 22528 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c21504_i64 = arith.constant 21504 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c0_i64_6 = arith.constant 0 : i64
  %c24_i32 = arith.constant 24 : i32
  %false = arith.constant false
  %true = arith.constant true
  %c-48_i32 = arith.constant -48 : i32
  %c-32_i32 = arith.constant -32 : i32
  %c-16_i32 = arith.constant -16 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c48_i32 = arith.constant 48 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  %c3072 = arith.constant 3072 : index
  %c4096 = arith.constant 4096 : index
  %c5120 = arith.constant 5120 : index
  %c6144 = arith.constant 6144 : index
  %c7168 = arith.constant 7168 : index
  %c8192 = arith.constant 8192 : index
  %c9216 = arith.constant 9216 : index
  %c10240 = arith.constant 10240 : index
  %c11264 = arith.constant 11264 : index
  %c12288 = arith.constant 12288 : index
  %c13312 = arith.constant 13312 : index
  %c14336 = arith.constant 14336 : index
  %c15360 = arith.constant 15360 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = arith.index_cast %arg5 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
    %6 = arith.index_cast %arg9 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c24_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = arith.index_cast %3 : i32 to index
    %14 = arith.index_cast %1 : i32 to index
    %15 = arith.index_cast %c24_i32 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%12, %13, %15]
    %17 = arith.index_cast %16 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %18 = arith.select %17, %c2_i64, %c3_i64 : i64
    %19 = arith.index_cast %16 : index to i1
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %20 = arith.select %19, %c5_i64, %c6_i64 : i64
    %21 = arith.index_cast %16 : index to i1
    %c0_i64_7 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %22 = arith.select %21, %c0_i64_7, %c1_i64 : i64
    %23 = hivm.hir.pointer_cast(%c19456_i64, %c39936_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %24 = hivm.hir.pointer_cast(%c18432_i64, %c38912_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %25 = hivm.hir.pointer_cast(%c17408_i64, %c37888_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %26 = hivm.hir.pointer_cast(%c16384_i64, %c36864_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %27 = hivm.hir.pointer_cast(%c15360_i64, %c35840_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %28 = hivm.hir.pointer_cast(%c14336_i64, %c34816_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %29 = hivm.hir.pointer_cast(%c13312_i64, %c33792_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %30 = hivm.hir.pointer_cast(%c12288_i64, %c32768_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %31 = hivm.hir.pointer_cast(%c11264_i64, %c31744_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %32 = hivm.hir.pointer_cast(%c10240_i64, %c30720_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %33 = hivm.hir.pointer_cast(%c9216_i64, %c29696_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %34 = hivm.hir.pointer_cast(%c8192_i64, %c28672_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %35 = hivm.hir.pointer_cast(%c7168_i64, %c27648_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %36 = hivm.hir.pointer_cast(%c6144_i64, %c26624_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %37 = hivm.hir.pointer_cast(%c5120_i64, %c25600_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %38 = hivm.hir.pointer_cast(%c4096_i64, %c24576_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %39 = hivm.hir.pointer_cast(%c3072_i64, %c23552_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %40 = hivm.hir.pointer_cast(%c2048_i64, %c22528_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %41 = hivm.hir.pointer_cast(%c1024_i64, %c21504_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %42 = hivm.hir.pointer_cast(%c0_i64_6, %c20480_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_mask_norm
    %43 = arith.remsi %arg9, %arg8 : i32
    %44 = arith.divsi %arg9, %arg8 : i32
    %45 = arith.remsi %44, %arg7 : i32
    %46 = arith.divsi %arg9, %4 : i32
    %47 = arith.remsi %46, %arg6 : i32
    %48 = arith.divsi %45, %c4_i32 : i32
    %49 = arith.remsi %45, %c4_i32 : i32
    %50 = arith.muli %48, %arg5 : i32
    %51 = arith.muli %50, %c4_i32 : i32
    %52 = arith.addi %51, %49 : i32
    %53 = arith.muli %52, %c64_i32 : i32
    %54 = arith.index_cast %53 : i32 to index
    %55 = arith.cmpi eq, %43, %c0_i32 : i32
    %56 = arith.muli %47, %c64_i32 : i32
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
    scf.if %55 {
      %57 = arith.addi %56, %c16_i32 : i32
      %58 = arith.maxsi %57, %c0_i32 : i32
      %59 = arith.index_cast %58 : i32 to index
      %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%54, %59]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%60], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %61 = arith.addi %56, %c32_i32 : i32
      %62 = arith.maxsi %61, %c0_i32 : i32
      %63 = arith.index_cast %62 : i32 to index
      %64 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%54, %63]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%64], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %65 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%54, %63]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%65], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %66 = arith.addi %56, %c48_i32 : i32
      %67 = arith.maxsi %66, %c0_i32 : i32
      %68 = arith.index_cast %67 : i32 to index
      %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%54, %68]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %70 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%54, %68]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%70], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %71 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%54, %68]
      %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%71], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %cast = memref.cast %42 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %72 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %59]
      %73 = affine.min affine_map<()[s0] -> (16, s0)>()[%72]
      %74 = arith.subi %c-16_i32, %56 : i32
      %75 = arith.maxsi %74, %c0_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%76, %72]
      %78 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%73, %77]
      %79 = arith.cmpi slt, %78, %c16 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%78, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %80 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%73, %77]
      %81 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%77]
      %82 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%77]
      %subview_13 = memref.subview %42[0, %81, %82, 0] [2, %80, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_14 = memref.cast %subview_13 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %79 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %42 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %20]
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_14 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_15 = memref.cast %41 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %83 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %63]
      %84 = affine.min affine_map<()[s0] -> (16, s0)>()[%83]
      %85 = arith.subi %c-32_i32, %56 : i32
      %86 = arith.maxsi %85, %c0_i32 : i32
      %87 = arith.index_cast %86 : i32 to index
      %88 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%87, %83]
      %89 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%84, %88]
      %90 = arith.cmpi slt, %89, %c16 : index
      %subview_16 = memref.subview %reinterpret_cast_8[0, 0] [%89, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %91 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%84, %88]
      %92 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%88]
      %93 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%88]
      %subview_17 = memref.subview %41[0, %92, %93, 0] [2, %91, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_18 = memref.cast %subview_17 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %90 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %41 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_16 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_19 = memref.cast %40 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview_20 = memref.subview %reinterpret_cast_9[0, 0] [%89, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_21 = memref.subview %40[0, %92, %93, 0] [2, %91, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_22 = memref.cast %subview_21 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %90 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %40 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_20 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_22 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_23 = memref.cast %39 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %94 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %68]
      %95 = affine.min affine_map<()[s0] -> (16, s0)>()[%94]
      %96 = arith.subi %c-48_i32, %56 : i32
      %97 = arith.maxsi %96, %c0_i32 : i32
      %98 = arith.index_cast %97 : i32 to index
      %99 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%98, %94]
      %100 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%95, %99]
      %101 = arith.cmpi slt, %100, %c16 : index
      %subview_24 = memref.subview %reinterpret_cast_10[0, 0] [%100, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %102 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%95, %99]
      %103 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%99]
      %104 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%99]
      %subview_25 = memref.subview %39[0, %103, %104, 0] [2, %102, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_26 = memref.cast %subview_25 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %101 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %39 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_24 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_26 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_27 = memref.cast %38 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview_28 = memref.subview %reinterpret_cast_11[0, 0] [%100, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_29 = memref.subview %38[0, %103, %104, 0] [2, %102, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_30 = memref.cast %subview_29 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %101 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %38 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_28 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_30 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_31 = memref.cast %37 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview_32 = memref.subview %reinterpret_cast_12[0, 0] [%100, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_33 = memref.subview %37[0, %103, %104, 0] [2, %102, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_34 = memref.cast %subview_33 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %101 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %37 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_32 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_34 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %105 = arith.maxsi %56, %c0_i32 : i32
      %106 = arith.index_cast %105 : i32 to index
      %107 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%54, %106]
      %reinterpret_cast_35 = memref.reinterpret_cast %arg4 to offset: [%107], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %108 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%54, %59]
      %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%108], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %109 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%54, %63]
      %reinterpret_cast_37 = memref.reinterpret_cast %arg4 to offset: [%109], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %110 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 48)>()[%54, %68]
      %reinterpret_cast_38 = memref.reinterpret_cast %arg4 to offset: [%110], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_39 = memref.cast %36 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %111 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %106]
      %112 = affine.min affine_map<()[s0] -> (16, s0)>()[%111]
      %113 = arith.subi %c0_i32, %56 : i32
      %114 = arith.maxsi %113, %c0_i32 : i32
      %115 = arith.index_cast %114 : i32 to index
      %116 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%115, %111]
      %117 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%112, %116]
      %118 = arith.cmpi slt, %117, %c16 : index
      %subview_40 = memref.subview %reinterpret_cast_35[0, 0] [%117, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %119 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%112, %116]
      %120 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%116]
      %121 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%116]
      %subview_41 = memref.subview %36[0, %120, %121, 0] [2, %119, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_42 = memref.cast %subview_41 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %118 {
        %collapse_shape = memref.collapse_shape %36 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_40 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_42 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_43 = memref.cast %35 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview_44 = memref.subview %reinterpret_cast_36[0, 0] [%78, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_45 = memref.subview %35[0, %81, %82, 0] [2, %80, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_46 = memref.cast %subview_45 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %79 {
        %collapse_shape = memref.collapse_shape %35 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_44 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_46 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %cast_47 = memref.cast %34 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview_48 = memref.subview %reinterpret_cast_37[0, 0] [%89, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_49 = memref.subview %34[0, %92, %93, 0] [2, %91, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_50 = memref.cast %subview_49 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %90 {
        %collapse_shape = memref.collapse_shape %34 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_48 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_50 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_51 = memref.cast %33 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %subview_52 = memref.subview %reinterpret_cast_38[0, 0] [%100, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_53 = memref.subview %33[0, %103, %104, 0] [2, %102, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      %cast_54 = memref.cast %subview_53 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %101 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        %collapse_shape = memref.collapse_shape %33 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_52 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_54 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11
      %122 = hivm.hir.pointer_cast(%c0_i64_6) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_55 = memref.cast %122 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_43, %cast, %true, %78, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_55 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_5, %c-1_i64, %c-1_i64, %c-1_i64, %11, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %123 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_55 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%123 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_56 = memref.cast %32 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%123 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_56 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
      %124 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_57 = memref.cast %124 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      %c-1_i64_58 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_56, %cast_39, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_57 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_4, %c-1_i64_58, %c-1_i64_58, %c-1_i64_58, %11, %c-1_i64_58, %c-1_i64_58 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %125 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_57 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%125 : memref<16x16xf32, #hivm.address_space<gm>>)
      annotation.mark %125 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %126 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %cast_59 = memref.cast %31 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%126 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_59 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
      %127 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %cast_60 = memref.cast %30 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%127 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_60 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
      %128 = hivm.hir.pointer_cast(%c2048_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_61 = memref.cast %128 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_47, %cast_19, %true, %89, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_61 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %129 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_61 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%129 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_62 = memref.cast %29 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%129 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_62 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
      %130 = hivm.hir.pointer_cast(%c3072_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_63 = memref.cast %130 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      %c-1_i64_64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_62, %cast_43, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_63 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_3, %c-1_i64_64, %c-1_i64_64, %c-1_i64_64, %11, %c-1_i64_64, %c-1_i64_64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %131 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_63 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%131 : memref<16x16xf32, #hivm.address_space<gm>>)
      annotation.mark %131 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %132 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %cast_65 = memref.cast %28 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%132 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_65 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      %133 = hivm.hir.pointer_cast(%c4096_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_66 = memref.cast %133 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_51, %cast_31, %true, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_66 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %134 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_66 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%134 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_67 = memref.cast %27 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%134 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_67 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
      %135 = hivm.hir.pointer_cast(%c5120_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_68 = memref.cast %135 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      %c-1_i64_69 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_67, %cast_47, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_68 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_2, %c-1_i64_69, %c-1_i64_69, %c-1_i64_69, %11, %c-1_i64_69, %c-1_i64_69 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %136 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_68 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%136 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
      annotation.mark %136 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %137 = hivm.hir.pointer_cast(%c6144_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_70 = memref.cast %137 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
      hivm.hir.mmadL1 ins(%cast_15, %cast_39, %true, %89, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_70 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_19, %cast_59, %false, %89, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_70 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %138 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_70 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%138 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
      %cast_71 = memref.cast %26 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%138 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_71 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
      %139 = hivm.hir.pointer_cast(%c7168_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_72 = memref.cast %139 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
      %c-1_i64_73 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_47, %cast_71, %true, %89, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_72 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_73, %c0_i64_1, %c-1_i64_73, %c-1_i64_73, %11, %c-1_i64_73, %c-1_i64_73 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %140 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_72 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%140 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
      annotation.mark %140 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %141 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      %cast_74 = memref.cast %25 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%141 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_74 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      %142 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_75 = memref.cast %142 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
      hivm.hir.mmadL1 ins(%cast_27, %cast_43, %true, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_75 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_31, %cast_65, %false, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_75 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %143 = memref_ext.alloc_workspace() from %arg2 offset = [%c12288] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_75 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%143 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
      %cast_76 = memref.cast %24 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%143 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_76 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
      %144 = hivm.hir.pointer_cast(%c9216_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_77 = memref.cast %144 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
      %c-1_i64_78 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_51, %cast_76, %true, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_77 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_78, %c0_i64_0, %c-1_i64_78, %c-1_i64_78, %11, %c-1_i64_78, %c-1_i64_78 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %145 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_77 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%145 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
      annotation.mark %145 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
      %146 = hivm.hir.pointer_cast(%c10240_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_79 = memref.cast %146 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      hivm.hir.mmadL1 ins(%cast_23, %cast_39, %true, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_79 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.mmadL1 ins(%cast_27, %cast_60, %false, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_79 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      %c-1_i64_80 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_31, %cast_74, %false, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_79 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_80, %c-1_i64_80, %20, %c-1_i64_80, %11, %c-1_i64_80, %c-1_i64_80 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %147 = memref_ext.alloc_workspace() from %arg2 offset = [%c14336] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_79 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%147 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %cast_81 = memref.cast %23 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %22]
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.nd2nz {dst_continuous} ins(%147 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_81 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
      %148 = hivm.hir.pointer_cast(%c11264_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      %cast_82 = memref.cast %148 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %c-1_i64_83 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_51, %cast_81, %true, %100, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_82 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_83, %c0_i64, %18, %22, %11, %c-1_i64_83, %c-1_i64_83 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %149 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_82 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%149 : memref<16x16xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      annotation.mark %149 : memref<16x16xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12
    }
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
