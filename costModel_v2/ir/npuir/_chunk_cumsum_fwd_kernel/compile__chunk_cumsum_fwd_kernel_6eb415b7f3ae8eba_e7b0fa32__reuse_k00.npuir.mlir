// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_cumsum_fwd_kernel_aligned(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: f32, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4160_i64 = arith.constant 4160 : i64
  %c3328_i64 = arith.constant 3328 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c3040_i64 = arith.constant 3040 : i64
  %c4032_i64 = arith.constant 4032 : i64
  %c2912_i64 = arith.constant 2912 : i64
  %c2848_i64 = arith.constant 2848 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c2528_i64 = arith.constant 2528 : i64
  %c2400_i64 = arith.constant 2400 : i64
  %c2336_i64 = arith.constant 2336 : i64
  %c2208_i64 = arith.constant 2208 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c2016_i64 = arith.constant 2016 : i64
  %c1888_i64 = arith.constant 1888 : i64
  %c1760_i64 = arith.constant 1760 : i64
  %c1728_i64 = arith.constant 1728 : i64
  %c1472_i64 = arith.constant 1472 : i64
  %c4000_i64 = arith.constant 4000 : i64
  %c1440_i64 = arith.constant 1440 : i64
  %c3968_i64 = arith.constant 3968 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c896_i64 = arith.constant 896 : i64
  %c3456_i64 = arith.constant 3456 : i64
  %c384_i64 = arith.constant 384 : i64
  %c256_i64 = arith.constant 256 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 2.000000e+01 : f32
  %c4_i32 = arith.constant 4 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg22, %arg23 : i32
  %1 = arith.muli %0, %arg24 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg24, %arg23 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<32xf32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg15 : i32 to index
  %7 = hivm.hir.pointer_cast(%c128_i64) : memref<4x8xf32, #hivm.address_space<ub>>
  %collapse_shape_1 = memref.collapse_shape %7 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg12 : f32) outs(%collapse_shape_1 : memref<32xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c128_i64) : memref<4x8xi1, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %8 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_1, %collapse_shape_1 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<32xi1, #hivm.address_space<ub>>)
  %9 = hivm.hir.pointer_cast(%c128_i64) : memref<4x8xi1, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %9 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_2 : memref<32xi1, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<32xi1, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c256_i64) : memref<4x8xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %10 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg13 : f32) outs(%collapse_shape_4 : memref<32xf32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c256_i64) : memref<4x8xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %11 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_4, %collapse_shape_4 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<32xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c256_i64) : memref<4x8xi1, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %12 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_5 : memref<32xi1, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<32xi1, #hivm.address_space<ub>>)
  %13 = arith.index_cast %arg18 : i32 to index
  %14 = arith.index_cast %arg21 : i32 to index
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg25 = %3 to %1 step %c48_i32  : i32 {
    %15 = arith.index_cast %arg25 : i32 to index
    %16 = arith.index_cast %3 : i32 to index
    %17 = arith.index_cast %1 : i32 to index
    %18 = arith.index_cast %c48_i32 : i32 to index
    %19 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%15, %16, %18]
    %20 = arith.index_cast %19 : index to i1
    %c0_i64_7 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %21 = arith.select %20, %c0_i64_7, %c1_i64 : i64
    %22 = hivm.hir.pointer_cast(%c3328_i64, %c4160_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<4x8xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c2912_i64, %c4032_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<4x8xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c1440_i64, %c4000_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<4x1xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c1408_i64, %c3968_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<4x1xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c384_i64, %c3456_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<8x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %27 = arith.remsi %arg25, %arg24 : i32
    %28 = arith.divsi %arg25, %arg24 : i32
    %29 = arith.remsi %28, %arg23 : i32
    %30 = arith.divsi %arg25, %4 : i32
    %31 = arith.remsi %30, %arg22 : i32
    %32 = arith.muli %31, %arg14 : i32
    %33 = arith.muli %29, %arg11 : i32
    %34 = arith.muli %33, %arg15 : i32
    %35 = arith.addi %32, %34 : i32
    %36 = arith.index_cast %35 : i32 to index
    %37 = arith.muli %31, %arg16 : i32
    %38 = arith.muli %29, %arg17 : i32
    %39 = arith.addi %37, %38 : i32
    %40 = arith.muli %31, %arg19 : i32
    %41 = arith.muli %29, %arg20 : i32
    %42 = arith.addi %40, %41 : i32
    %43 = arith.muli %27, %c4_i32 : i32
    %44 = arith.index_cast %43 : i32 to index
    %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%36, %44]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%45], sizes: [8, 4], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x4xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %26[0, 0] [8, 4] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<8x4xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<8x4xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview : memref<8x4xf32, strided<[16, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %46 = hivm.hir.pointer_cast(%c896_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %subview_8 = memref.subview %46[0, 0] [4, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%subview : memref<8x4xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) permutation = [1, 0]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [4, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_10 = memref.collapse_shape %reinterpret_cast_9 [[0, 1]] : memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_11 = memref.collapse_shape %25 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_10 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_11 : memref<4xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [4, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_13 = memref.collapse_shape %reinterpret_cast_12 [[0, 1]] : memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_14 = memref.collapse_shape %24 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_13 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_14 : memref<4xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %47 = hivm.hir.pointer_cast(%c896_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %48 = hivm.hir.pointer_cast(%c1472_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%subview_8, %24 : memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%47 : memref<4x8xf32, #hivm.address_space<ub>>) temp_buffer(%48 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %49 = hivm.hir.pointer_cast(%c1728_i64) : memref<4x8xi1, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %47 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %49 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_15, %collapse_shape : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <le>
    %50 = hivm.hir.pointer_cast(%c1760_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %50 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_15 : memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<32xf32, #hivm.address_space<ub>>)
    %51 = hivm.hir.pointer_cast(%c1760_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %51 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_17, %cst_0 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_18 : memref<32xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c1760_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %52 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_18 : memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<32xf32, #hivm.address_space<ub>>)
    %53 = hivm.hir.pointer_cast(%c1888_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %53 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c2016_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_16, %collapse_shape_19, %collapse_shape_15 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<8xf32, #hivm.address_space<ub>>)
    %55 = hivm.hir.pointer_cast(%c2048_i64) : memref<4x8xi1, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %55 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_20, %collapse_shape_20 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<32xi1, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c2048_i64) : memref<4x8xi1, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %56 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_21 : memref<32xi1, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<32xi1, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c2080_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %57 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_20, %arg12 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_23 : memref<32xf32, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c2208_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %58 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c2336_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_22, %arg12, %collapse_shape_23 : memref<32xi1, #hivm.address_space<ub>>, f32, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c2400_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %60 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c2528_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_3, %collapse_shape_20, %collapse_shape_24 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<8xf32, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c2560_i64) : memref<4x8xi1, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %62 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_25, %collapse_shape_25 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<32xi1, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c2560_i64) : memref<4x8xi1, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %63 [[0, 1]] : memref<4x8xi1, #hivm.address_space<ub>> into memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_26 : memref<32xi1, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<32xi1, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c2592_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %64 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_25, %arg13 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_28 : memref<32xf32, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c2720_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %65 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %66 = hivm.hir.pointer_cast(%c2848_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_27, %arg13, %collapse_shape_28 : memref<32xi1, #hivm.address_space<ub>>, f32, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%66 : memref<16xf32, #hivm.address_space<ub>>)
    %collapse_shape_30 = memref.collapse_shape %23 [[0, 1]] : memref<4x8xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c3040_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, %21]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_6, %collapse_shape_25, %collapse_shape_29 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %68 = arith.index_cast %39 : i32 to index
    %69 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%68, %44, %13]
    %reinterpret_cast_31 = memref.reinterpret_cast %arg6 to offset: [%69], sizes: [4, 8], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%23 : memref<4x8xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_31 : memref<4x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %70 = hivm.hir.pointer_cast(%c3072_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%23, %25 : memref<4x8xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%23 : memref<4x8xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg26 = %c0 to %c4 step %c1 {
      %73 = memref.load %23[%arg26, %c0] : memref<4x8xf32, #hivm.address_space<ub>>
      memref.store %73, %22[%arg26, %c0] : memref<4x8xf32, #hivm.address_space<ub>>
      scf.for %arg27 = %c1 to %c8 step %c1 {
        %74 = memref.load %23[%arg26, %arg27] : memref<4x8xf32, #hivm.address_space<ub>>
        %75 = arith.subi %arg27, %c1 : index
        %76 = memref.load %22[%arg26, %75] : memref<4x8xf32, #hivm.address_space<ub>>
        %77 = arith.addf %74, %76 : f32
        memref.store %77, %22[%arg26, %arg27] : memref<4x8xf32, #hivm.address_space<ub>>
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, %21]
    %71 = arith.index_cast %42 : i32 to index
    %72 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%71, %44, %14]
    %reinterpret_cast_32 = memref.reinterpret_cast %arg7 to offset: [%72], sizes: [4, 8], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%22 : memref<4x8xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_32 : memref<4x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
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
