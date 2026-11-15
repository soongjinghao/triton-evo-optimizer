// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3456_i64 = arith.constant 3456 : i64
  %c4928_i64 = arith.constant 4928 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c3904_i64 = arith.constant 3904 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c1344_i64 = arith.constant 1344 : i64
  %c1312_i64 = arith.constant 1312 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c1248_i64 = arith.constant 1248 : i64
  %c3872_i64 = arith.constant 3872 : i64
  %c1216_i64 = arith.constant 1216 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c992_i64 = arith.constant 992 : i64
  %c3840_i64 = arith.constant 3840 : i64
  %c960_i64 = arith.constant 960 : i64
  %c928_i64 = arith.constant 928 : i64
  %c800_i64 = arith.constant 800 : i64
  %c768_i64 = arith.constant 768 : i64
  %c704_i64 = arith.constant 704 : i64
  %c576_i64 = arith.constant 576 : i64
  %c448_i64 = arith.constant 448 : i64
  %c320_i64 = arith.constant 320 : i64
  %c3712_i64 = arith.constant 3712 : i64
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0x7F800000 : f32
  %c4_i64 = arith.constant 4 : i64
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
  %subview = memref.subview %5[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%subview : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c128_i64) : memref<4xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%6 : memref<4xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c160_i64) : memref<4xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%7 : memref<4xf32, #hivm.address_space<ub>>)
  %8 = arith.extsi %arg10 : i32 to i64
  %9 = arith.index_cast %arg9 : i32 to index
  %10 = arith.extsi %arg7 : i32 to i64
  %11 = arith.muli %arg11, %arg9 : i32
  %12 = arith.extsi %11 : i32 to i64
  %13 = arith.index_cast %arg8 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  scf.for %arg15 = %3 to %1 step %c48_i32  : i32 {
    %14 = arith.index_cast %arg15 : i32 to index
    %15 = arith.index_cast %3 : i32 to index
    %16 = arith.index_cast %1 : i32 to index
    %17 = arith.index_cast %c48_i32 : i32 to index
    %18 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%14, %15, %17]
    %19 = arith.index_cast %18 : index to i1
    %c4_i64_2 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %20 = arith.select %19, %c4_i64_2, %c5_i64 : i64
    %21 = arith.index_cast %18 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %22 = arith.select %21, %c2_i64, %c3_i64 : i64
    %23 = arith.index_cast %18 : index to i1
    %c0_i64_3 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %24 = arith.select %23, %c0_i64_3, %c1_i64 : i64
    %25 = hivm.hir.pointer_cast(%c2432_i64, %c4928_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<4x64xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c1408_i64, %c3904_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<4x64xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c1216_i64, %c3872_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c960_i64, %c3840_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c192_i64, %c3712_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<4x8x1xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %30 = arith.divsi %arg15, %arg14 : i32
    %31 = arith.remsi %30, %arg13 : i32
    %32 = arith.divsi %arg15, %4 : i32
    %33 = arith.remsi %32, %arg12 : i32
    %34 = arith.extsi %33 : i32 to i64
    %35 = arith.extsi %31 : i32 to i64
    %36 = arith.muli %35, %c4_i64 : i64
    %37 = arith.muli %34, %8 : i64
    %38 = arith.index_cast %37 : i64 to index
    %39 = arith.index_cast %36 : i64 to index
    %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%38, %39]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [4, 4], strides: [%9, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x4xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %29[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %24]
    hivm.hir.load ins(%reinterpret_cast : memref<4x4xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %41 = hivm.hir.pointer_cast(%c320_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
    %subview_5 = memref.subview %41[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%subview_4, %subview_4 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c320_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
    %subview_6 = memref.subview %42[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%subview_5 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %43 = hivm.hir.pointer_cast(%c448_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
    %subview_7 = memref.subview %43[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%subview_4, %subview : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %44 = hivm.hir.pointer_cast(%c320_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
    %subview_8 = memref.subview %44[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%subview_6, %subview_7 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c576_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %subview_9 = memref.subview %45[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %46 = hivm.hir.pointer_cast(%c704_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%subview_8, %cst_0, %subview_4 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%46 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %24]
    %47 = hivm.hir.pointer_cast(%c768_i64) : memref<1x4xf32, #hivm.address_space<ub>>
    %48 = hivm.hir.pointer_cast(%c800_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%subview_9 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%47 : memref<1x4xf32, #hivm.address_space<ub>>) temp_buffer(%48 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape = memref.collapse_shape %47 [[0, 1]] : memref<1x4xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c928_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape, %6 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%49 : memref<4xi1, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c992_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %24]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%49, %cst_1, %collapse_shape : memref<4xi1, #hivm.address_space<ub>>, f32, memref<4xf32, #hivm.address_space<ub>>) outs(%28 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %28 [[0, 1]] output_shape [1, 4] : memref<4xf32, #hivm.address_space<ub>> into memref<1x4xf32, #hivm.address_space<ub>>
    %51 = hivm.hir.pointer_cast(%c576_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %subview_10 = memref.subview %51[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%subview_9, %expand_shape : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<1x4xf32, #hivm.address_space<ub>>) outs(%subview_10 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) broadcast = [0]
    %52 = hivm.hir.pointer_cast(%c576_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %subview_11 = memref.subview %52[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%subview_10 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %53 = hivm.hir.pointer_cast(%c1056_i64) : memref<1x4xf32, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c1088_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%subview_11 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%53 : memref<1x4xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_12 = memref.collapse_shape %53 [[0, 1]] : memref<1x4xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %55 = hivm.hir.pointer_cast(%c1056_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_12 : memref<4xf32, #hivm.address_space<ub>>) outs(%55 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%55, %28 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%28 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%40], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%28 : memref<4xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_13 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %24]
    %56 = arith.muli %34, %10 : i64
    %57 = arith.addi %12, %37 : i64
    %58 = arith.index_cast %57 : i64 to index
    %59 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%58, %39]
    %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %22]
    hivm.hir.load ins(%reinterpret_cast_14 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%27 : memref<4xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%27, %28 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%27 : memref<4xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c1248_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%27, %27 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%60 : memref<4xi1, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c1248_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%60 : memref<4xi1, #hivm.address_space<ub>>) outs(%61 : memref<4xi1, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c1280_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%27, %7 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%62 : memref<4xi1, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c1248_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%61, %62 : memref<4xi1, #hivm.address_space<ub>>, memref<4xi1, #hivm.address_space<ub>>) outs(%63 : memref<4xi1, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c1312_i64) : memref<4xf32, #hivm.address_space<ub>>
    %65 = hivm.hir.pointer_cast(%c1344_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%63, %cst_0, %27 : memref<4xi1, #hivm.address_space<ub>>, f32, memref<4xf32, #hivm.address_space<ub>>) outs(%64 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%65 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %22]
    %66 = hivm.hir.pointer_cast(%c1312_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%64 : memref<4xf32, #hivm.address_space<ub>>) outs(%66 : memref<4xf32, #hivm.address_space<ub>>)
    %67 = arith.index_cast %56 : i64 to index
    %68 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%67, %39, %13]
    %reinterpret_cast_15 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [4, 64], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %20]
    hivm.hir.load ins(%reinterpret_cast_15 : memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%26 : memref<4x64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape_16 = memref.expand_shape %66 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %69 = hivm.hir.pointer_cast(%c3456_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %22]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%26, %expand_shape_16 : memref<4x64xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%25 : memref<4x64xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %20]
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [4, 64], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%25 : memref<4x64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_17 : memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %22]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
