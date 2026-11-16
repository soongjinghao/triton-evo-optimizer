// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_token_group_quant_8bit_colmajor_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16224_i64 = arith.constant 16224 : i64
  %c25472_i64 = arith.constant 25472 : i64
  %c15200_i64 = arith.constant 15200 : i64
  %c11104_i64 = arith.constant 11104 : i64
  %c11040_i64 = arith.constant 11040 : i64
  %c10016_i64 = arith.constant 10016 : i64
  %c9984_i64 = arith.constant 9984 : i64
  %c8960_i64 = arith.constant 8960 : i64
  %c7936_i64 = arith.constant 7936 : i64
  %c7904_i64 = arith.constant 7904 : i64
  %c7872_i64 = arith.constant 7872 : i64
  %c6848_i64 = arith.constant 6848 : i64
  %c6784_i64 = arith.constant 6784 : i64
  %c5760_i64 = arith.constant 5760 : i64
  %c4736_i64 = arith.constant 4736 : i64
  %c4704_i64 = arith.constant 4704 : i64
  %c4448_i64 = arith.constant 4448 : i64
  %c4416_i64 = arith.constant 4416 : i64
  %c4384_i64 = arith.constant 4384 : i64
  %c25440_i64 = arith.constant 25440 : i64
  %c4352_i64 = arith.constant 4352 : i64
  %c4288_i64 = arith.constant 4288 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c4224_i64 = arith.constant 4224 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c24416_i64 = arith.constant 24416 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.00787401571 : f32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c8_i32 = arith.constant 8 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant -1.280000e+02 : f32
  %cst_2 = arith.constant 1.270000e+02 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<8x32xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x32xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %6 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg6 : f32) outs(%7 : memref<8xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%7, %7 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%8 : memref<8xi1, #hivm.address_space<ub>>)
  %9 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%8 : memref<8xi1, #hivm.address_space<ub>>) outs(%9 : memref<8xi1, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c2080_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : memref<8xf32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x32xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %11 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x32xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %12 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<256xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c0_i64) : memref<8x32xi1, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %13 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c0_i64) : memref<8x32xi1, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %14 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<256xi1, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %15 = arith.index_cast %arg10 : i32 to index
    %16 = arith.index_cast %3 : i32 to index
    %17 = arith.index_cast %1 : i32 to index
    %18 = arith.index_cast %c48_i32 : i32 to index
    %19 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%15, %16, %18]
    %20 = arith.index_cast %19 : index to i1
    %c0_i64_8 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %21 = arith.select %20, %c0_i64_8, %c1_i64 : i64
    %22 = hivm.hir.pointer_cast(%c15200_i64, %c25472_i64) : memref<32x32xi8, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<32x32xi8, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c4352_i64, %c25440_i64) : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c2112_i64, %c24416_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8x32xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %25 = arith.divsi %arg10, %arg9 : i32
    %26 = arith.remsi %25, %arg8 : i32
    %27 = arith.divsi %arg10, %4 : i32
    %28 = arith.remsi %27, %arg7 : i32
    %29 = arith.muli %28, %c8_i32 : i32
    %30 = arith.muli %26, %c32_i32 : i32
    %31 = arith.index_cast %29 : i32 to index
    %32 = arith.index_cast %30 : i32 to index
    %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%32, %31]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [8, 32], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [8, 32], strides: [128, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<8x32xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %34 = arith.muli %26, %c16_i32 : i32
    %35 = arith.index_cast %34 : i32 to index
    %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%35, %31]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%36], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %21]
    hivm.hir.load ins(%reinterpret_cast : memref<8x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%24 : memref<8x32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %37 = hivm.hir.pointer_cast(%c3136_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %24 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %37 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%collapse_shape_11 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c4160_i64) : memref<8x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%37 : memref<8x32xf32, #hivm.address_space<ub>>) outs(%38 : memref<8x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_13 = memref.collapse_shape %38 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c4192_i64) : memref<8xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_13, %collapse_shape_13 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%39 : memref<8xi1, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c4192_i64) : memref<8xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%39 : memref<8xi1, #hivm.address_space<ub>>) outs(%40 : memref<8xi1, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c4224_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_13, %arg6 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%41 : memref<8xf32, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c4256_i64) : memref<8xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c4288_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%40, %arg6, %41 : memref<8xi1, #hivm.address_space<ub>>, f32, memref<8xf32, #hivm.address_space<ub>>) outs(%42 : memref<8xf32, #hivm.address_space<ub>>) temp_buffer(%43 : memref<16xf32, #hivm.address_space<ub>>)
    %44 = hivm.hir.pointer_cast(%c4384_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %21]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%9, %collapse_shape_13, %42 : memref<8xi1, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%23 : memref<8xf32, #hivm.address_space<ub>>) temp_buffer(%44 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%23, %cst : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%23 : memref<8xf32, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c4416_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%10, %23 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%45 : memref<8xf32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %45 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %46 = hivm.hir.pointer_cast(%c4448_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%24, %expand_shape : memref<8x32xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%24 : memref<8x32xf32, #hivm.address_space<ub>>) temp_buffer(%46 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %47 = hivm.hir.pointer_cast(%c4704_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %24 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %47 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_14, %collapse_shape_14 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<256xi1, #hivm.address_space<ub>>)
    %48 = hivm.hir.pointer_cast(%c4704_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %48 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_15 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c4736_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %49 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_14, %cst_2 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<256xf32, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c5760_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %50 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %51 = hivm.hir.pointer_cast(%c6784_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_16, %cst_2, %collapse_shape_17 : memref<256xi1, #hivm.address_space<ub>>, f32, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<16xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c6848_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %52 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c7872_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_5, %collapse_shape_14, %collapse_shape_18 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %21]
    %54 = hivm.hir.pointer_cast(%c7904_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %54 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_19, %collapse_shape_19 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<256xi1, #hivm.address_space<ub>>)
    %55 = hivm.hir.pointer_cast(%c7904_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %55 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_20 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<256xi1, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c7936_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %56 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_19, %cst_1 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_22 : memref<256xf32, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c8960_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %57 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %58 = hivm.hir.pointer_cast(%c9984_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_7, %collapse_shape_19, %collapse_shape_22 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%58 : memref<8xf32, #hivm.address_space<ub>>)
    %59 = hivm.hir.pointer_cast(%c10016_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %59 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %60 = hivm.hir.pointer_cast(%c11040_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_21, %cst_1, %collapse_shape_23 : memref<256xi1, #hivm.address_space<ub>>, f32, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<16xf32, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c11104_i64) : memref<32x32xi32, #hivm.address_space<ub>>
    %subview = memref.subview %61[0, 0] [8, 32] [1, 1] : memref<32x32xi32, #hivm.address_space<ub>> to memref<8x32xi32, strided<[32, 1]>, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %subview [[0, 1]] : memref<8x32xi32, strided<[32, 1]>, #hivm.address_space<ub>> into memref<256xi32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_24 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<256xi32, strided<[1]>, #hivm.address_space<ub>>) round_mode = <trunc>
    %subview_26 = memref.subview %22[0, 0] [8, 32] [1, 1] : memref<32x32xi8, #hivm.address_space<ub>> to memref<8x32xi8, strided<[32, 1]>, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1]] : memref<8x32xi8, strided<[32, 1]>, #hivm.address_space<ub>> into memref<256xi8, strided<[1]>, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c16224_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_25 : memref<256xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<256xi8, strided<[1]>, #hivm.address_space<ub>>) temp_buffer(%62 : memref<2048xi32, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_26 : memref<8x32xi8, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<8x32xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.store ins(%23 : memref<8xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %21]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
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
