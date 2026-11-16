// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c62176_i64 = arith.constant 62176 : i64
  %c59232_i64 = arith.constant 59232 : i64
  %c58208_i64 = arith.constant 58208 : i64
  %c58144_i64 = arith.constant 58144 : i64
  %c57632_i64 = arith.constant 57632 : i64
  %c62112_i64 = arith.constant 62112 : i64
  %c57568_i64 = arith.constant 57568 : i64
  %c57504_i64 = arith.constant 57504 : i64
  %c56480_i64 = arith.constant 56480 : i64
  %c55456_i64 = arith.constant 55456 : i64
  %c54432_i64 = arith.constant 54432 : i64
  %c54368_i64 = arith.constant 54368 : i64
  %c53856_i64 = arith.constant 53856 : i64
  %c62048_i64 = arith.constant 62048 : i64
  %c53792_i64 = arith.constant 53792 : i64
  %c53728_i64 = arith.constant 53728 : i64
  %c52704_i64 = arith.constant 52704 : i64
  %c51680_i64 = arith.constant 51680 : i64
  %c50656_i64 = arith.constant 50656 : i64
  %c50592_i64 = arith.constant 50592 : i64
  %c50080_i64 = arith.constant 50080 : i64
  %c61984_i64 = arith.constant 61984 : i64
  %c50016_i64 = arith.constant 50016 : i64
  %c49952_i64 = arith.constant 49952 : i64
  %c48928_i64 = arith.constant 48928 : i64
  %c47904_i64 = arith.constant 47904 : i64
  %c46880_i64 = arith.constant 46880 : i64
  %c46816_i64 = arith.constant 46816 : i64
  %c46304_i64 = arith.constant 46304 : i64
  %c61920_i64 = arith.constant 61920 : i64
  %c46240_i64 = arith.constant 46240 : i64
  %c46176_i64 = arith.constant 46176 : i64
  %c45152_i64 = arith.constant 45152 : i64
  %c44128_i64 = arith.constant 44128 : i64
  %c43104_i64 = arith.constant 43104 : i64
  %c43040_i64 = arith.constant 43040 : i64
  %c42528_i64 = arith.constant 42528 : i64
  %c61856_i64 = arith.constant 61856 : i64
  %c42464_i64 = arith.constant 42464 : i64
  %c42400_i64 = arith.constant 42400 : i64
  %c41376_i64 = arith.constant 41376 : i64
  %c40352_i64 = arith.constant 40352 : i64
  %c39328_i64 = arith.constant 39328 : i64
  %c39264_i64 = arith.constant 39264 : i64
  %c38752_i64 = arith.constant 38752 : i64
  %c61792_i64 = arith.constant 61792 : i64
  %c38688_i64 = arith.constant 38688 : i64
  %c38624_i64 = arith.constant 38624 : i64
  %c37600_i64 = arith.constant 37600 : i64
  %c36576_i64 = arith.constant 36576 : i64
  %c35552_i64 = arith.constant 35552 : i64
  %c35488_i64 = arith.constant 35488 : i64
  %c34976_i64 = arith.constant 34976 : i64
  %c61728_i64 = arith.constant 61728 : i64
  %c34912_i64 = arith.constant 34912 : i64
  %c34848_i64 = arith.constant 34848 : i64
  %c33824_i64 = arith.constant 33824 : i64
  %c32800_i64 = arith.constant 32800 : i64
  %c31776_i64 = arith.constant 31776 : i64
  %c31712_i64 = arith.constant 31712 : i64
  %c31200_i64 = arith.constant 31200 : i64
  %c61664_i64 = arith.constant 61664 : i64
  %c31136_i64 = arith.constant 31136 : i64
  %c31072_i64 = arith.constant 31072 : i64
  %c30048_i64 = arith.constant 30048 : i64
  %c29024_i64 = arith.constant 29024 : i64
  %c28000_i64 = arith.constant 28000 : i64
  %c27936_i64 = arith.constant 27936 : i64
  %c27424_i64 = arith.constant 27424 : i64
  %c61600_i64 = arith.constant 61600 : i64
  %c27360_i64 = arith.constant 27360 : i64
  %c27296_i64 = arith.constant 27296 : i64
  %c26272_i64 = arith.constant 26272 : i64
  %c25248_i64 = arith.constant 25248 : i64
  %c24224_i64 = arith.constant 24224 : i64
  %c24160_i64 = arith.constant 24160 : i64
  %c23648_i64 = arith.constant 23648 : i64
  %c61536_i64 = arith.constant 61536 : i64
  %c23584_i64 = arith.constant 23584 : i64
  %c23520_i64 = arith.constant 23520 : i64
  %c22496_i64 = arith.constant 22496 : i64
  %c21472_i64 = arith.constant 21472 : i64
  %c20448_i64 = arith.constant 20448 : i64
  %c20384_i64 = arith.constant 20384 : i64
  %c19872_i64 = arith.constant 19872 : i64
  %c61472_i64 = arith.constant 61472 : i64
  %c19808_i64 = arith.constant 19808 : i64
  %c19744_i64 = arith.constant 19744 : i64
  %c18720_i64 = arith.constant 18720 : i64
  %c17696_i64 = arith.constant 17696 : i64
  %c16672_i64 = arith.constant 16672 : i64
  %c16608_i64 = arith.constant 16608 : i64
  %c16096_i64 = arith.constant 16096 : i64
  %c61408_i64 = arith.constant 61408 : i64
  %c16032_i64 = arith.constant 16032 : i64
  %c15968_i64 = arith.constant 15968 : i64
  %c14944_i64 = arith.constant 14944 : i64
  %c13920_i64 = arith.constant 13920 : i64
  %c12896_i64 = arith.constant 12896 : i64
  %c12832_i64 = arith.constant 12832 : i64
  %c12320_i64 = arith.constant 12320 : i64
  %c61344_i64 = arith.constant 61344 : i64
  %c12256_i64 = arith.constant 12256 : i64
  %c12192_i64 = arith.constant 12192 : i64
  %c11168_i64 = arith.constant 11168 : i64
  %c10144_i64 = arith.constant 10144 : i64
  %c9120_i64 = arith.constant 9120 : i64
  %c9056_i64 = arith.constant 9056 : i64
  %c8544_i64 = arith.constant 8544 : i64
  %c61280_i64 = arith.constant 61280 : i64
  %c8480_i64 = arith.constant 8480 : i64
  %c8416_i64 = arith.constant 8416 : i64
  %c7392_i64 = arith.constant 7392 : i64
  %c7328_i64 = arith.constant 7328 : i64
  %c6304_i64 = arith.constant 6304 : i64
  %c60256_i64 = arith.constant 60256 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c5184_i64 = arith.constant 5184 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c6_i32 = arith.constant 6 : i32
  %c7_i32 = arith.constant 7 : i32
  %c9_i32 = arith.constant 9 : i32
  %c10_i32 = arith.constant 10 : i32
  %c11_i32 = arith.constant 11 : i32
  %c12_i32 = arith.constant 12 : i32
  %c13_i32 = arith.constant 13 : i32
  %c14_i32 = arith.constant 14 : i32
  %c15_i32 = arith.constant 15 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c1088_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%6 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%7 : memref<128xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %expand_shape_1 = memref.expand_shape %5 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
  %8 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x16xi32, #hivm.address_space<ub>>) outs(%8 : memref<16x16xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %9 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi32, #hivm.address_space<ub>>
  %10 = hivm.hir.pointer_cast(%c3648_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape, %expand_shape_1 : memref<16x1xi32, #hivm.address_space<ub>>, memref<1x16xi32, #hivm.address_space<ub>>) outs(%9 : memref<16x16xi32, #hivm.address_space<ub>>) temp_buffer(%10 : memref<128xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %11 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %9 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %8 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %11 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %12 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_3 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<256xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %6 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %13 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_2 : memref<256xi32, #hivm.address_space<ub>>, memref<256xi32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>)
  %14 = arith.index_cast %arg5 : i32 to index
  %15 = hivm.hir.pointer_cast(%c4160_i64) : memref<16x16xf32, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %15 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %16 = hivm.hir.pointer_cast(%c5184_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_6 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%16 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %17 = arith.index_cast %arg9 : i32 to index
    %18 = arith.index_cast %3 : i32 to index
    %19 = arith.index_cast %1 : i32 to index
    %20 = arith.index_cast %c48_i32 : i32 to index
    %21 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%17, %18, %20]
    %22 = arith.index_cast %21 : index to i1
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %23 = arith.select %22, %c5_i64, %c6_i64 : i64
    %24 = arith.index_cast %21 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %25 = arith.select %24, %c3_i64, %c4_i64 : i64
    %26 = arith.index_cast %21 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %27 = arith.select %26, %c1_i64, %c2_i64 : i64
    %28 = arith.index_cast %21 : index to i1
    %c0_i64_8 = arith.constant 0 : i64
    %c1_i64_9 = arith.constant 1 : i64
    %29 = arith.select %28, %c0_i64_8, %c1_i64_9 : i64
    %30 = hivm.hir.pointer_cast(%c59232_i64, %c62176_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c57568_i64, %c62112_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c53792_i64, %c62048_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c50016_i64, %c61984_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c46240_i64, %c61920_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %35 = hivm.hir.pointer_cast(%c42464_i64, %c61856_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %36 = hivm.hir.pointer_cast(%c38688_i64, %c61792_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %37 = hivm.hir.pointer_cast(%c34912_i64, %c61728_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %38 = hivm.hir.pointer_cast(%c31136_i64, %c61664_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c27360_i64, %c61600_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %40 = hivm.hir.pointer_cast(%c23584_i64, %c61536_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %41 = hivm.hir.pointer_cast(%c19808_i64, %c61472_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c16032_i64, %c61408_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c12256_i64, %c61344_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %43 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c8480_i64, %c61280_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %44 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c5280_i64, %c60256_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %46 = arith.divsi %arg9, %arg8 : i32
    %47 = arith.remsi %46, %arg7 : i32
    %48 = arith.divsi %arg9, %4 : i32
    %49 = arith.remsi %48, %arg6 : i32
    %50 = arith.divsi %47, %c8_i32 : i32
    %51 = arith.remsi %47, %c8_i32 : i32
    %52 = arith.muli %50, %arg5 : i32
    %53 = arith.muli %52, %c8_i32 : i32
    %54 = arith.addi %53, %51 : i32
    %55 = arith.muli %54, %c16_i32 : i32
    %56 = arith.index_cast %55 : i32 to index
    %57 = arith.muli %49, %c16_i32 : i32
    %58 = arith.remsi %57, %c16_i32 : i32
    %59 = arith.maxsi %57, %c0_i32 : i32
    %60 = arith.index_cast %59 : i32 to index
    %61 = arith.maxsi %58, %c0_i32 : i32
    %62 = arith.index_cast %61 : i32 to index
    %63 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%56, %60]
    %64 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%62, %56, %60]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%64], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %65 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>()[%14, %62, %60]
    %66 = affine.min affine_map<()[s0] -> (16, s0)>()[%65]
    %67 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 16, 0)>()[%62]
    %68 = affine.min affine_map<()[s0] -> (16, s0)>()[%67]
    %69 = arith.subi %c0_i32, %57 : i32
    %70 = arith.maxsi %69, %c0_i32 : i32
    %71 = arith.index_cast %70 : i32 to index
    %72 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%71, %65]
    %73 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%66, %72]
    %74 = arith.subi %c0_i32, %58 : i32
    %75 = arith.maxsi %74, %c0_i32 : i32
    %76 = arith.index_cast %75 : i32 to index
    %77 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%76, %67]
    %78 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%68, %77]
    %79 = arith.cmpi slt, %73, %c16 : index
    %80 = arith.cmpi slt, %78, %c16 : index
    %81 = arith.ori %79, %80 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%73, %78] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %45[%72, %77] [%73, %78] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %81 {
      %collapse_shape_172 = memref.collapse_shape %45 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_172 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %27]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %77 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %82 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %45 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %82 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %83 = hivm.hir.pointer_cast(%c7328_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_11, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%83 : memref<16xf32, #hivm.address_space<ub>>)
    %84 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %84 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_12, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>>)
    %85 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %85 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_13, %cst : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_14 : memref<256xf32, #hivm.address_space<ub>>)
    %86 = hivm.hir.pointer_cast(%c7392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %86 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %87 = hivm.hir.pointer_cast(%c8416_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_15 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%87 : memref<16xf32, #hivm.address_space<ub>>)
    %88 = arith.addi %57, %c2_i32 : i32
    %89 = arith.muli %88, %c128_i32 : i32
    %90 = arith.index_cast %89 : i32 to index
    %91 = arith.index_cast %58 : i32 to index
    %92 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %90]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_16 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%44 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%44, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%44 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%44, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%44 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_17 = memref.expand_shape %44 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %93 = hivm.hir.pointer_cast(%c7392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %94 = hivm.hir.pointer_cast(%c8544_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_17, %86 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%93 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%94 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %95 = hivm.hir.pointer_cast(%c9056_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %96 = hivm.hir.pointer_cast(%c9120_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%93 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%95 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%96 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_18 = memref.collapse_shape %95 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %97 = hivm.hir.pointer_cast(%c9056_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%44, %collapse_shape_18 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%97 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_19 = memref.expand_shape %97 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %98 = hivm.hir.pointer_cast(%c10144_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_19 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%98 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_20 = memref.subview %85[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_21 = memref.subview %98[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %subview_20 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %subview_21 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_22 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_24 = memref.subview %98[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_25 = memref.subview %85[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %subview_24 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %subview_25 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_26 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>)
    %99 = hivm.hir.pointer_cast(%c11168_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %99 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %100 = hivm.hir.pointer_cast(%c12192_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_28 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%100 : memref<16xf32, #hivm.address_space<ub>>)
    %101 = arith.addi %57, %c3_i32 : i32
    %102 = arith.muli %101, %c128_i32 : i32
    %103 = arith.index_cast %102 : i32 to index
    %104 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %103]
    %reinterpret_cast_29 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_29 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%43 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%43, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%43 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%43, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%43 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_30 = memref.expand_shape %43 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %105 = hivm.hir.pointer_cast(%c11168_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %106 = hivm.hir.pointer_cast(%c12320_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_30, %99 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%105 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%106 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %107 = hivm.hir.pointer_cast(%c12832_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %108 = hivm.hir.pointer_cast(%c12896_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%105 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%107 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%108 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_31 = memref.collapse_shape %107 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%43, %collapse_shape_31 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%43 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_32 = memref.expand_shape %43 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %109 = hivm.hir.pointer_cast(%c13920_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_32 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%109 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_33 = memref.subview %109[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %subview_33 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_26 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_35 = memref.subview %109[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_36 = memref.subview %85[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_37 = memref.collapse_shape %subview_35 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_38 = memref.collapse_shape %subview_36 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_37 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_38 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>)
    %110 = hivm.hir.pointer_cast(%c14944_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_39 = memref.collapse_shape %110 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %111 = hivm.hir.pointer_cast(%c15968_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_39 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%111 : memref<16xf32, #hivm.address_space<ub>>)
    %112 = arith.addi %57, %c4_i32 : i32
    %113 = arith.muli %112, %c128_i32 : i32
    %114 = arith.index_cast %113 : i32 to index
    %115 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %114]
    %reinterpret_cast_40 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_40 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%42 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%42, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%42 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%42, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%42 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_41 = memref.expand_shape %42 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %116 = hivm.hir.pointer_cast(%c14944_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %117 = hivm.hir.pointer_cast(%c16096_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_41, %110 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%116 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%117 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %118 = hivm.hir.pointer_cast(%c16608_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %119 = hivm.hir.pointer_cast(%c16672_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%116 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%118 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%119 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_42 = memref.collapse_shape %118 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%42, %collapse_shape_42 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%42 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_43 = memref.expand_shape %42 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %120 = hivm.hir.pointer_cast(%c17696_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_43 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%120 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_44 = memref.subview %120[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %subview_44 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_37 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_45 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_46 = memref.subview %120[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_47 = memref.subview %85[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_48 = memref.collapse_shape %subview_46 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_49 = memref.collapse_shape %subview_47 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_48 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_49 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>)
    %121 = hivm.hir.pointer_cast(%c18720_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_50 = memref.collapse_shape %121 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %122 = hivm.hir.pointer_cast(%c19744_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_50 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%122 : memref<16xf32, #hivm.address_space<ub>>)
    %123 = arith.addi %57, %c5_i32 : i32
    %124 = arith.muli %123, %c128_i32 : i32
    %125 = arith.index_cast %124 : i32 to index
    %126 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %125]
    %reinterpret_cast_51 = memref.reinterpret_cast %arg3 to offset: [%126], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_51 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%41 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%41, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%41 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%41, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%41 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_52 = memref.expand_shape %41 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %127 = hivm.hir.pointer_cast(%c18720_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %128 = hivm.hir.pointer_cast(%c19872_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_52, %121 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%127 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%128 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %129 = hivm.hir.pointer_cast(%c20384_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %130 = hivm.hir.pointer_cast(%c20448_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%127 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%129 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%130 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_53 = memref.collapse_shape %129 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %131 = hivm.hir.pointer_cast(%c20384_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%41, %collapse_shape_53 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%131 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_54 = memref.expand_shape %131 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %132 = hivm.hir.pointer_cast(%c21472_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_54 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%132 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_55 = memref.subview %132[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_56 = memref.collapse_shape %subview_55 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_48 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_56 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_57 = memref.subview %132[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_58 = memref.subview %85[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_59 = memref.collapse_shape %subview_57 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_60 = memref.collapse_shape %subview_58 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_59 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_60 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>)
    %133 = hivm.hir.pointer_cast(%c22496_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_61 = memref.collapse_shape %133 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %134 = hivm.hir.pointer_cast(%c23520_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_61 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%134 : memref<16xf32, #hivm.address_space<ub>>)
    %135 = arith.addi %57, %c6_i32 : i32
    %136 = arith.muli %135, %c128_i32 : i32
    %137 = arith.index_cast %136 : i32 to index
    %138 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %137]
    %reinterpret_cast_62 = memref.reinterpret_cast %arg3 to offset: [%138], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_62 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%40 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%40, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%40 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%40, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%40 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_63 = memref.expand_shape %40 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %139 = hivm.hir.pointer_cast(%c22496_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %140 = hivm.hir.pointer_cast(%c23648_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_63, %133 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%139 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%140 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %141 = hivm.hir.pointer_cast(%c24160_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %142 = hivm.hir.pointer_cast(%c24224_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%139 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%141 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%142 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_64 = memref.collapse_shape %141 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %143 = hivm.hir.pointer_cast(%c24160_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%40, %collapse_shape_64 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%143 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_65 = memref.expand_shape %143 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %144 = hivm.hir.pointer_cast(%c25248_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_65 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%144 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_66 = memref.subview %144[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_67 = memref.collapse_shape %subview_66 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_59 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_67 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_68 = memref.subview %144[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_69 = memref.subview %85[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_70 = memref.collapse_shape %subview_68 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_71 = memref.collapse_shape %subview_69 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_70 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_71 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>)
    %145 = hivm.hir.pointer_cast(%c26272_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_72 = memref.collapse_shape %145 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %146 = hivm.hir.pointer_cast(%c27296_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_72 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%146 : memref<16xf32, #hivm.address_space<ub>>)
    %147 = arith.addi %57, %c7_i32 : i32
    %148 = arith.muli %147, %c128_i32 : i32
    %149 = arith.index_cast %148 : i32 to index
    %150 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %149]
    %reinterpret_cast_73 = memref.reinterpret_cast %arg3 to offset: [%150], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_73 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%39 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%39, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%39 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%39, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%39 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_74 = memref.expand_shape %39 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %151 = hivm.hir.pointer_cast(%c26272_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %152 = hivm.hir.pointer_cast(%c27424_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_74, %145 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%151 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%152 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %153 = hivm.hir.pointer_cast(%c27936_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %154 = hivm.hir.pointer_cast(%c28000_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%151 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%153 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%154 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_75 = memref.collapse_shape %153 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%39, %collapse_shape_75 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%39 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_76 = memref.expand_shape %39 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %155 = hivm.hir.pointer_cast(%c29024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_76 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%155 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_77 = memref.subview %155[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_78 = memref.collapse_shape %subview_77 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_70 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_78 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_79 = memref.subview %155[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_80 = memref.subview %85[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_81 = memref.collapse_shape %subview_79 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_82 = memref.collapse_shape %subview_80 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_81 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_82 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>)
    %156 = hivm.hir.pointer_cast(%c30048_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_83 = memref.collapse_shape %156 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %157 = hivm.hir.pointer_cast(%c31072_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_83 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%157 : memref<16xf32, #hivm.address_space<ub>>)
    %158 = arith.addi %57, %c8_i32 : i32
    %159 = arith.muli %158, %c128_i32 : i32
    %160 = arith.index_cast %159 : i32 to index
    %161 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %160]
    %reinterpret_cast_84 = memref.reinterpret_cast %arg3 to offset: [%161], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_84 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%38 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%38, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%38, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_85 = memref.expand_shape %38 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %162 = hivm.hir.pointer_cast(%c30048_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %163 = hivm.hir.pointer_cast(%c31200_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_85, %156 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%162 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%163 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %164 = hivm.hir.pointer_cast(%c31712_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %165 = hivm.hir.pointer_cast(%c31776_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%162 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%164 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%165 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_86 = memref.collapse_shape %164 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%38, %collapse_shape_86 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%38 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_87 = memref.expand_shape %38 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %166 = hivm.hir.pointer_cast(%c32800_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_87 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%166 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_88 = memref.subview %166[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_89 = memref.collapse_shape %subview_88 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_81 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_89 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_90 = memref.subview %166[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_91 = memref.subview %85[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_92 = memref.collapse_shape %subview_90 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_93 = memref.collapse_shape %subview_91 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_92 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_93 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>)
    %167 = hivm.hir.pointer_cast(%c33824_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_94 = memref.collapse_shape %167 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %168 = hivm.hir.pointer_cast(%c34848_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_94 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%168 : memref<16xf32, #hivm.address_space<ub>>)
    %169 = arith.addi %57, %c9_i32 : i32
    %170 = arith.muli %169, %c128_i32 : i32
    %171 = arith.index_cast %170 : i32 to index
    %172 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %171]
    %reinterpret_cast_95 = memref.reinterpret_cast %arg3 to offset: [%172], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_95 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%37 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%37, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%37, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_96 = memref.expand_shape %37 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %173 = hivm.hir.pointer_cast(%c33824_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %174 = hivm.hir.pointer_cast(%c34976_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_96, %167 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%173 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%174 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %175 = hivm.hir.pointer_cast(%c35488_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %176 = hivm.hir.pointer_cast(%c35552_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%173 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%175 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%176 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_97 = memref.collapse_shape %175 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%37, %collapse_shape_97 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%37 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_98 = memref.expand_shape %37 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %177 = hivm.hir.pointer_cast(%c36576_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_98 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%177 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_99 = memref.subview %177[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_100 = memref.collapse_shape %subview_99 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_92 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_100 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_101 = memref.subview %177[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_102 = memref.subview %85[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_103 = memref.collapse_shape %subview_101 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_104 = memref.collapse_shape %subview_102 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_103 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_104 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>)
    %178 = hivm.hir.pointer_cast(%c37600_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_105 = memref.collapse_shape %178 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %179 = hivm.hir.pointer_cast(%c38624_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_105 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%179 : memref<16xf32, #hivm.address_space<ub>>)
    %180 = arith.addi %57, %c10_i32 : i32
    %181 = arith.muli %180, %c128_i32 : i32
    %182 = arith.index_cast %181 : i32 to index
    %183 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %182]
    %reinterpret_cast_106 = memref.reinterpret_cast %arg3 to offset: [%183], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_106 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%36 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%36, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%36 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%36, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%36 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_107 = memref.expand_shape %36 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %184 = hivm.hir.pointer_cast(%c37600_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %185 = hivm.hir.pointer_cast(%c38752_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_107, %178 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%184 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%185 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %186 = hivm.hir.pointer_cast(%c39264_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %187 = hivm.hir.pointer_cast(%c39328_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%184 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%186 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%187 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_108 = memref.collapse_shape %186 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %188 = hivm.hir.pointer_cast(%c39264_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%36, %collapse_shape_108 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%188 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_109 = memref.expand_shape %188 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %189 = hivm.hir.pointer_cast(%c40352_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_109 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%189 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_110 = memref.subview %189[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_111 = memref.collapse_shape %subview_110 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_103 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_111 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_112 = memref.subview %189[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_113 = memref.subview %85[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_114 = memref.collapse_shape %subview_112 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_115 = memref.collapse_shape %subview_113 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_114 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_115 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>)
    %190 = hivm.hir.pointer_cast(%c41376_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_116 = memref.collapse_shape %190 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %191 = hivm.hir.pointer_cast(%c42400_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_116 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%191 : memref<16xf32, #hivm.address_space<ub>>)
    %192 = arith.addi %57, %c11_i32 : i32
    %193 = arith.muli %192, %c128_i32 : i32
    %194 = arith.index_cast %193 : i32 to index
    %195 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %194]
    %reinterpret_cast_117 = memref.reinterpret_cast %arg3 to offset: [%195], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_117 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%35 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%35, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%35 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%35, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%35 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_118 = memref.expand_shape %35 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %196 = hivm.hir.pointer_cast(%c41376_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %197 = hivm.hir.pointer_cast(%c42528_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_118, %190 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%196 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%197 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %198 = hivm.hir.pointer_cast(%c43040_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %199 = hivm.hir.pointer_cast(%c43104_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%196 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%198 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%199 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_119 = memref.collapse_shape %198 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%35, %collapse_shape_119 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%35 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_120 = memref.expand_shape %35 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %200 = hivm.hir.pointer_cast(%c44128_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_120 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%200 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_121 = memref.subview %200[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_122 = memref.collapse_shape %subview_121 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_114 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_122 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_123 = memref.subview %200[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_124 = memref.subview %85[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_125 = memref.collapse_shape %subview_123 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_126 = memref.collapse_shape %subview_124 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_125 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_126 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>)
    %201 = hivm.hir.pointer_cast(%c45152_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_127 = memref.collapse_shape %201 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %202 = hivm.hir.pointer_cast(%c46176_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_127 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%202 : memref<16xf32, #hivm.address_space<ub>>)
    %203 = arith.addi %57, %c12_i32 : i32
    %204 = arith.muli %203, %c128_i32 : i32
    %205 = arith.index_cast %204 : i32 to index
    %206 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %205]
    %reinterpret_cast_128 = memref.reinterpret_cast %arg3 to offset: [%206], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_128 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%34 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%34, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%34 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%34, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%34 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_129 = memref.expand_shape %34 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %207 = hivm.hir.pointer_cast(%c45152_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %208 = hivm.hir.pointer_cast(%c46304_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_129, %201 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%207 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%208 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %209 = hivm.hir.pointer_cast(%c46816_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %210 = hivm.hir.pointer_cast(%c46880_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%207 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%209 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%210 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_130 = memref.collapse_shape %209 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%34, %collapse_shape_130 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%34 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_131 = memref.expand_shape %34 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %211 = hivm.hir.pointer_cast(%c47904_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_131 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%211 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_132 = memref.subview %211[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_133 = memref.collapse_shape %subview_132 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_125 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_133 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_134 = memref.subview %211[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_135 = memref.subview %85[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_136 = memref.collapse_shape %subview_134 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_137 = memref.collapse_shape %subview_135 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_136 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_137 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>)
    %212 = hivm.hir.pointer_cast(%c48928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_138 = memref.collapse_shape %212 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %213 = hivm.hir.pointer_cast(%c49952_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_138 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%213 : memref<16xf32, #hivm.address_space<ub>>)
    %214 = arith.addi %57, %c13_i32 : i32
    %215 = arith.muli %214, %c128_i32 : i32
    %216 = arith.index_cast %215 : i32 to index
    %217 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %216]
    %reinterpret_cast_139 = memref.reinterpret_cast %arg3 to offset: [%217], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_139 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%33 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%33, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%33 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%33, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%33 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_140 = memref.expand_shape %33 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %218 = hivm.hir.pointer_cast(%c48928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %219 = hivm.hir.pointer_cast(%c50080_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_140, %212 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%218 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%219 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %220 = hivm.hir.pointer_cast(%c50592_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %221 = hivm.hir.pointer_cast(%c50656_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%218 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%220 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%221 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_141 = memref.collapse_shape %220 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %222 = hivm.hir.pointer_cast(%c50592_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%33, %collapse_shape_141 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%222 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %27]
    %expand_shape_142 = memref.expand_shape %222 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %223 = hivm.hir.pointer_cast(%c51680_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_142 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%223 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_143 = memref.subview %223[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_144 = memref.collapse_shape %subview_143 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_136 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_144 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_145 = memref.subview %223[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_146 = memref.subview %85[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_147 = memref.collapse_shape %subview_145 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_148 = memref.collapse_shape %subview_146 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_147 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_148 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>)
    %224 = hivm.hir.pointer_cast(%c52704_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_149 = memref.collapse_shape %224 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %225 = hivm.hir.pointer_cast(%c53728_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_149 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%225 : memref<16xf32, #hivm.address_space<ub>>)
    %226 = arith.addi %57, %c14_i32 : i32
    %227 = arith.muli %226, %c128_i32 : i32
    %228 = arith.index_cast %227 : i32 to index
    %229 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %228]
    %reinterpret_cast_150 = memref.reinterpret_cast %arg3 to offset: [%229], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %25]
    hivm.hir.load ins(%reinterpret_cast_150 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%32 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%32, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%32 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%32, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%32 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_151 = memref.expand_shape %32 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %230 = hivm.hir.pointer_cast(%c52704_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %231 = hivm.hir.pointer_cast(%c53856_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_151, %224 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%230 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%231 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %232 = hivm.hir.pointer_cast(%c54368_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %233 = hivm.hir.pointer_cast(%c54432_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%230 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%232 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%233 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_152 = memref.collapse_shape %232 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%32, %collapse_shape_152 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%32 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_153 = memref.expand_shape %32 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    %234 = hivm.hir.pointer_cast(%c55456_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_153 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%234 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %25]
    %subview_154 = memref.subview %234[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_155 = memref.collapse_shape %subview_154 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_147 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_155 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_156 = memref.subview %234[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_157 = memref.subview %85[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_158 = memref.collapse_shape %subview_156 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_159 = memref.collapse_shape %subview_157 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_158 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_159 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>)
    %235 = hivm.hir.pointer_cast(%c56480_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_160 = memref.collapse_shape %235 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %236 = hivm.hir.pointer_cast(%c57504_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_14, %cst : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_160 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%236 : memref<16xf32, #hivm.address_space<ub>>)
    %237 = arith.addi %57, %c15_i32 : i32
    %238 = arith.muli %237, %c128_i32 : i32
    %239 = arith.index_cast %238 : i32 to index
    %240 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%91, %56, %239]
    %reinterpret_cast_161 = memref.reinterpret_cast %arg3 to offset: [%240], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
    hivm.hir.load ins(%reinterpret_cast_161 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%31 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%31, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%31, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape_162 = memref.expand_shape %31 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %241 = hivm.hir.pointer_cast(%c56480_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %242 = hivm.hir.pointer_cast(%c57632_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%expand_shape_162, %235 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%241 : memref<16x16xf32, #hivm.address_space<ub>>) temp_buffer(%242 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    %243 = hivm.hir.pointer_cast(%c58144_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %244 = hivm.hir.pointer_cast(%c58208_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%241 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%243 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%244 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_163 = memref.collapse_shape %243 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %245 = hivm.hir.pointer_cast(%c58144_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%31, %collapse_shape_163 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%245 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
    %expand_shape_164 = memref.expand_shape %245 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %29]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_164 : memref<1x16xf32, #hivm.address_space<ub>>) outs(%30 : memref<16x16xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %subview_165 = memref.subview %30[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %collapse_shape_166 = memref.collapse_shape %subview_165 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_158 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_166 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>>)
    %collapse_shape_167 = memref.collapse_shape %30 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_168 = memref.collapse_shape %30 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_167, %collapse_shape_7 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_168 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_169 = memref.reinterpret_cast %arg4 to offset: [%63], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %246 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%14, %60]
    %247 = affine.min affine_map<()[s0] -> (16, s0)>()[%246]
    %248 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%71, %246]
    %249 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%247, %248]
    %subview_170 = memref.subview %30[%248, 0] [%249, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_171 = memref.subview %reinterpret_cast_169[0, 0] [%249, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_170 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_171 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %29]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
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
