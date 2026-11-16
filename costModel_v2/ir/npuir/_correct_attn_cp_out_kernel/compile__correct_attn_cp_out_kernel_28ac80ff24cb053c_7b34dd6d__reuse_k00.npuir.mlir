// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c6528_i64 = arith.constant 6528 : i64
  %c7424_i64 = arith.constant 7424 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c2368_i64 = arith.constant 2368 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c2272_i64 = arith.constant 2272 : i64
  %c2240_i64 = arith.constant 2240 : i64
  %c7360_i64 = arith.constant 7360 : i64
  %c2176_i64 = arith.constant 2176 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1984_i64 = arith.constant 1984 : i64
  %c1952_i64 = arith.constant 1952 : i64
  %c1696_i64 = arith.constant 1696 : i64
  %c7296_i64 = arith.constant 7296 : i64
  %c1632_i64 = arith.constant 1632 : i64
  %c1568_i64 = arith.constant 1568 : i64
  %c1504_i64 = arith.constant 1504 : i64
  %c1472_i64 = arith.constant 1472 : i64
  %c1216_i64 = arith.constant 1216 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c832_i64 = arith.constant 832 : i64
  %c800_i64 = arith.constant 800 : i64
  %c768_i64 = arith.constant 768 : i64
  %c7040_i64 = arith.constant 7040 : i64
  %c512_i64 = arith.constant 512 : i64
  %c448_i64 = arith.constant 448 : i64
  %c384_i64 = arith.constant 384 : i64
  %c320_i64 = arith.constant 320 : i64
  %c256_i64 = arith.constant 256 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %c16_i64 = arith.constant 16 : i64
  %cst_0 = arith.constant 0x7F800000 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg13, %arg14 : i32
  %1 = arith.muli %0, %arg15 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg15, %arg14 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4x16xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<4x16xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<64xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c256_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%6 : memref<16xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c320_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%7 : memref<16xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c384_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%8 : memref<16xf32, #hivm.address_space<ub>>)
  %9 = arith.extsi %arg10 : i32 to i64
  %10 = arith.index_cast %arg9 : i32 to index
  %11 = arith.index_cast %arg12 : i32 to index
  %12 = hivm.hir.pointer_cast(%c448_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%12 : memref<16xf32, #hivm.address_space<ub>>)
  %13 = arith.muli %arg11, %arg9 : i32
  %14 = arith.extsi %13 : i32 to i64
  %15 = arith.extsi %arg7 : i32 to i64
  %16 = arith.index_cast %arg8 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg16 = %3 to %1 step %c48_i32  : i32 {
    %17 = arith.index_cast %arg16 : i32 to index
    %18 = arith.index_cast %3 : i32 to index
    %19 = arith.index_cast %1 : i32 to index
    %20 = arith.index_cast %c48_i32 : i32 to index
    %21 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%17, %18, %20]
    %22 = arith.index_cast %21 : index to i1
    %c0_i64_3 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %23 = arith.select %22, %c0_i64_3, %c1_i64 : i64
    %24 = arith.index_cast %21 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %25 = arith.select %24, %c3_i64, %c4_i64 : i64
    %26 = arith.index_cast %21 : index to i1
    %c1_i64_4 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %27 = arith.select %26, %c1_i64_4, %c2_i64 : i64
    %28 = hivm.hir.pointer_cast(%c2432_i64, %c7424_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c2432_i64, %c7424_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c2176_i64, %c7360_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c1632_i64, %c7296_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c512_i64, %c7040_i64) : memref<4x16xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<4x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %33 = arith.divsi %arg16, %arg15 : i32
    %34 = arith.remsi %33, %arg14 : i32
    %35 = arith.divsi %arg16, %4 : i32
    %36 = arith.remsi %35, %arg13 : i32
    %37 = arith.extsi %36 : i32 to i64
    %38 = arith.extsi %34 : i32 to i64
    %39 = arith.muli %38, %c16_i64 : i64
    %40 = arith.muli %37, %9 : i64
    %41 = arith.index_cast %40 : i64 to index
    %42 = arith.index_cast %39 : i64 to index
    %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%41, %42]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%43], sizes: [4, 16], strides: [%10, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %44 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%42, %11]
    %45 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%44, %42]
    %46 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%45, %42]
    %47 = arith.cmpi slt, %46, %c16 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [4, %46] [1, 1] : memref<4x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<4x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %32[0, 0] [4, %46] [1, 1] : memref<4x16xf32, #hivm.address_space<ub>> to memref<4x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %47 {
      %collapse_shape_29 = memref.collapse_shape %32 [[0, 1]] : memref<4x16xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_29 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %27]
    hivm.hir.load ins(%subview : memref<4x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<4x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %48 = hivm.hir.pointer_cast(%c768_i64) : memref<4x16xi1, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %32 [[0, 1]] : memref<4x16xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %48 [[0, 1]] : memref<4x16xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape_6 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<64xi1, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c768_i64) : memref<4x16xi1, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %49 [[0, 1]] : memref<4x16xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_7 : memref<64xi1, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<64xi1, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c800_i64) : memref<4x16xi1, #hivm.address_space<ub>>
    %collapse_shape_9 = memref.collapse_shape %50 [[0, 1]] : memref<4x16xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<64xi1, #hivm.address_space<ub>>)
    %51 = hivm.hir.pointer_cast(%c800_i64) : memref<4x16xi1, #hivm.address_space<ub>>
    %collapse_shape_10 = memref.collapse_shape %51 [[0, 1]] : memref<4x16xi1, #hivm.address_space<ub>> into memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%collapse_shape_8, %collapse_shape_9 : memref<64xi1, #hivm.address_space<ub>>, memref<64xi1, #hivm.address_space<ub>>) outs(%collapse_shape_10 : memref<64xi1, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c832_i64) : memref<4x16xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %52 [[0, 1]] : memref<4x16xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c1088_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_10, %cst, %collapse_shape_6 : memref<64xi1, #hivm.address_space<ub>>, f32, memref<64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<64xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %27]
    %54 = hivm.hir.pointer_cast(%c1152_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %55 = hivm.hir.pointer_cast(%c1216_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%52 : memref<4x16xf32, #hivm.address_space<ub>>) outs(%54 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_12 = memref.collapse_shape %54 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%c1472_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_12, %6 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%56 : memref<16xi1, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c1504_i64) : memref<16xf32, #hivm.address_space<ub>>
    %58 = hivm.hir.pointer_cast(%c1568_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%56, %cst_1, %collapse_shape_12 : memref<16xi1, #hivm.address_space<ub>>, f32, memref<16xf32, #hivm.address_space<ub>>) outs(%57 : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%58 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %57 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c832_i64) : memref<4x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%52, %expand_shape : memref<4x16xf32, #hivm.address_space<ub>>, memref<1x16xf32, #hivm.address_space<ub>>) outs(%59 : memref<4x16xf32, #hivm.address_space<ub>>) broadcast = [0]
    %60 = hivm.hir.pointer_cast(%c832_i64) : memref<4x16xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %59 [[0, 1]] : memref<4x16xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %60 [[0, 1]] : memref<4x16xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_13 : memref<64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<64xf32, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c1632_i64, %c7296_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c1696_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %27]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%60 : memref<4x16xf32, #hivm.address_space<ub>>) outs(%61 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%62 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_15 = memref.collapse_shape %61 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c1952_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_15, %7 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%63 : memref<16xi1, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c1984_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vdiv ins(%12, %collapse_shape_15 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%64 : memref<16xf32, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
    %66 = hivm.hir.pointer_cast(%c2112_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%63, %cst_1, %64 : memref<16xi1, #hivm.address_space<ub>>, f32, memref<16xf32, #hivm.address_space<ub>>) outs(%65 : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%66 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.vln ins(%collapse_shape_15 : memref<16xf32, #hivm.address_space<ub>>) outs(%31 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%31, %57 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%31 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg6 to offset: [%43], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %31[0] [%46] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %reinterpret_cast_16[0] [%46] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_18 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %27]
    %67 = arith.addi %14, %40 : i64
    %68 = arith.index_cast %67 : i64 to index
    %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%68, %42]
    %reinterpret_cast_19 = memref.reinterpret_cast %arg5 to offset: [%69], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_20 = memref.subview %reinterpret_cast_19[0] [%46] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_21 = memref.subview %30[0] [%46] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %47 {
      hivm.hir.vbrc ins(%cst : f32) outs(%30 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %25]
    hivm.hir.load ins(%subview_20 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %70 = hivm.hir.pointer_cast(%c2240_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%30, %30 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%70 : memref<16xi1, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c2240_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%70 : memref<16xi1, #hivm.address_space<ub>>) outs(%71 : memref<16xi1, #hivm.address_space<ub>>)
    %72 = hivm.hir.pointer_cast(%c2272_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%30, %8 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%72 : memref<16xi1, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c2272_i64) : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%71, %72 : memref<16xi1, #hivm.address_space<ub>>, memref<16xi1, #hivm.address_space<ub>>) outs(%73 : memref<16xi1, #hivm.address_space<ub>>)
    %74 = hivm.hir.pointer_cast(%c2304_i64) : memref<16xf32, #hivm.address_space<ub>>
    %75 = hivm.hir.pointer_cast(%c2368_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%73, %cst, %30 : memref<16xi1, #hivm.address_space<ub>>, f32, memref<16xf32, #hivm.address_space<ub>>) outs(%74 : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%75 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %25]
    %76 = hivm.hir.pointer_cast(%c2304_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%74, %57 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%76 : memref<16xf32, #hivm.address_space<ub>>)
    %77 = hivm.hir.pointer_cast(%c2304_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%76 : memref<16xf32, #hivm.address_space<ub>>) outs(%77 : memref<16xf32, #hivm.address_space<ub>>)
    %78 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%77, %65 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%78 : memref<16xf32, #hivm.address_space<ub>>)
    %79 = arith.muli %37, %15 : i64
    %80 = arith.index_cast %79 : i64 to index
    %81 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%80, %42, %16]
    %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%81], sizes: [16, 64], strides: [%16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_23 = memref.subview %reinterpret_cast_22[0, 0] [%46, 64] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_24 = memref.subview %29[0, 0] [%46, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    scf.if %47 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_29 = memref.collapse_shape %29 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_29 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %23]
    hivm.hir.load ins(%subview_23 : memref<?x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_24 : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape_25 = memref.expand_shape %78 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %82 = hivm.hir.pointer_cast(%c6528_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%29, %expand_shape_25 : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%28 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%82 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_26 = memref.reinterpret_cast %arg4 to offset: [%81], sizes: [16, 64], strides: [%16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_27 = memref.subview %28[0, 0] [%46, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_28 = memref.subview %reinterpret_cast_26[0, 0] [%46, 64] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_27 : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_28 : memref<?x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %23]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
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
