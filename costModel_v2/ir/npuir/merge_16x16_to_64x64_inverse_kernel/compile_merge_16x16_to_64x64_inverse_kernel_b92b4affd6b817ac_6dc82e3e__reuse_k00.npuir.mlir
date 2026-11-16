// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_64x64_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c15520_i64 = arith.constant 15520 : i64
  %c15008_i64 = arith.constant 15008 : i64
  %c14496_i64 = arith.constant 14496 : i64
  %c13984_i64 = arith.constant 13984 : i64
  %c13472_i64 = arith.constant 13472 : i64
  %c12960_i64 = arith.constant 12960 : i64
  %c12448_i64 = arith.constant 12448 : i64
  %c11936_i64 = arith.constant 11936 : i64
  %c11424_i64 = arith.constant 11424 : i64
  %c10912_i64 = arith.constant 10912 : i64
  %c10880_i64 = arith.constant 10880 : i64
  %c10368_i64 = arith.constant 10368 : i64
  %c10336_i64 = arith.constant 10336 : i64
  %c9824_i64 = arith.constant 9824 : i64
  %c9312_i64 = arith.constant 9312 : i64
  %c9280_i64 = arith.constant 9280 : i64
  %c8768_i64 = arith.constant 8768 : i64
  %c8736_i64 = arith.constant 8736 : i64
  %c8224_i64 = arith.constant 8224 : i64
  %c7712_i64 = arith.constant 7712 : i64
  %c7680_i64 = arith.constant 7680 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c7136_i64 = arith.constant 7136 : i64
  %c6624_i64 = arith.constant 6624 : i64
  %c6112_i64 = arith.constant 6112 : i64
  %c6080_i64 = arith.constant 6080 : i64
  %c5568_i64 = arith.constant 5568 : i64
  %c5536_i64 = arith.constant 5536 : i64
  %c5024_i64 = arith.constant 5024 : i64
  %c4512_i64 = arith.constant 4512 : i64
  %c4448_i64 = arith.constant 4448 : i64
  %c4384_i64 = arith.constant 4384 : i64
  %c4320_i64 = arith.constant 4320 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c3232_i64 = arith.constant 3232 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2208_i64 = arith.constant 2208 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c576_i64 = arith.constant 576 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c27648 = arith.constant 27648 : index
  %c26624 = arith.constant 26624 : index
  %c25600 = arith.constant 25600 : index
  %c24576 = arith.constant 24576 : index
  %c23552 = arith.constant 23552 : index
  %c22528 = arith.constant 22528 : index
  %c21504 = arith.constant 21504 : index
  %c20480 = arith.constant 20480 : index
  %c19456 = arith.constant 19456 : index
  %c18432 = arith.constant 18432 : index
  %c11264 = arith.constant 11264 : index
  %c10240 = arith.constant 10240 : index
  %c9216 = arith.constant 9216 : index
  %c8192 = arith.constant 8192 : index
  %c7168 = arith.constant 7168 : index
  %c6144 = arith.constant 6144 : index
  %c5120 = arith.constant 5120 : index
  %c4096 = arith.constant 4096 : index
  %c3072 = arith.constant 3072 : index
  %c2048 = arith.constant 2048 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c256_i32 = arith.constant 256 : i32
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %c32_i32 = arith.constant 32 : i32
  %c48_i32 = arith.constant 48 : i32
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c16384_i32 = arith.constant 16384 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg7, %arg6 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c512_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : memref<16xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %6 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  %8 = hivm.hir.pointer_cast(%c1088_i64) : memref<0xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<16x1xi32, #hivm.address_space<ub>>) outs(%7 : memref<16x8xi32, #hivm.address_space<ub>>) temp_buffer(%8 : memref<0xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
  %9 = hivm.hir.get_sub_block_idx -> i64
  %10 = arith.index_cast %9 : i64 to index
  %11 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%10]
  hivm.hir.set_ffts_base_addr %arg0
  annotation.mark %1 {logical_block_num} : i32
  %subview = memref.subview %6[%11] [8] [1] {to_be_bubbled_slice} : memref<16xi32, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %expand_shape_1 = memref.expand_shape %subview [[0, 1]] output_shape [1, 8] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
  %12 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_1 : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%12 : memref<16x8xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %13 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
  %collapse_shape_2 = memref.collapse_shape %7 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %12 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %13 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_2, %collapse_shape_3 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x8xf32, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %14 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c2112_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_4 : memref<128xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%15 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg8 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg8, %arg7 : i32
    %17 = arith.remsi %16, %arg6 : i32
    %18 = arith.divsi %arg8, %4 : i32
    %19 = arith.remsi %18, %arg5 : i32
    %20 = arith.divsi %19, %c4_i32 : i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.subi %19, %21 : i32
    %23 = arith.muli %20, %c512_i32 : i32
    %24 = arith.addi %23, %22 : i32
    %25 = arith.muli %24, %c64_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.muli %17, %c64_i32 : i32
    %28 = arith.muli %17, %c16384_i32 : i32
    %29 = arith.index_cast %28 : i32 to index
    %30 = hivm.hir.pointer_cast(%c2208_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %30 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<128xf32, #hivm.address_space<ub>>)
    %31 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %31 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %32 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<128xf32, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_9 = memref.collapse_shape %33 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.copy ins(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<128xf32, #hivm.address_space<ub>>)
    scf.for %arg9 = %c1_i32 to %c16_i32 step %c1_i32  : i32 {
      %99 = arith.muli %arg9, %c256_i32 : i32
      %100 = arith.index_cast %99 : i32 to index
      %101 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%100, %26, %29]
      %reinterpret_cast_83 = memref.reinterpret_cast %arg3 to offset: [%101], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %102 = hivm.hir.pointer_cast(%c4256_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%reinterpret_cast_83 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%102 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %103 = hivm.hir.pointer_cast(%c4256_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%102, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%103 : memref<16xf32, #hivm.address_space<ub>>)
      %104 = hivm.hir.pointer_cast(%c4256_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%103, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%104 : memref<16xf32, #hivm.address_space<ub>>)
      %105 = arith.addi %arg9, %c16_i32 : i32
      %106 = arith.muli %105, %c256_i32 : i32
      %107 = arith.index_cast %106 : i32 to index
      %108 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 16)>()[%107, %26, %29]
      %reinterpret_cast_84 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %109 = hivm.hir.pointer_cast(%c4320_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_84 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%109 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %110 = hivm.hir.pointer_cast(%c4320_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%109, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%110 : memref<16xf32, #hivm.address_space<ub>>)
      %111 = hivm.hir.pointer_cast(%c4320_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%110, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%111 : memref<16xf32, #hivm.address_space<ub>>)
      %112 = arith.addi %arg9, %c32_i32 : i32
      %113 = arith.muli %112, %c256_i32 : i32
      %114 = arith.index_cast %113 : i32 to index
      %115 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 32)>()[%114, %26, %29]
      %reinterpret_cast_85 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %116 = hivm.hir.pointer_cast(%c4384_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%reinterpret_cast_85 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%116 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %117 = hivm.hir.pointer_cast(%c4384_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%116, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%117 : memref<16xf32, #hivm.address_space<ub>>)
      %118 = hivm.hir.pointer_cast(%c4384_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%117, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%118 : memref<16xf32, #hivm.address_space<ub>>)
      %119 = arith.addi %arg9, %c48_i32 : i32
      %120 = arith.muli %119, %c256_i32 : i32
      %121 = arith.index_cast %120 : i32 to index
      %122 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 48)>()[%121, %26, %29]
      %reinterpret_cast_86 = memref.reinterpret_cast %arg3 to offset: [%122], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %123 = hivm.hir.pointer_cast(%c4448_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.load ins(%reinterpret_cast_86 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%123 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %124 = hivm.hir.pointer_cast(%c4448_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%123, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%124 : memref<16xf32, #hivm.address_space<ub>>)
      %125 = hivm.hir.pointer_cast(%c4448_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%124, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%125 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_87 = memref.expand_shape %104 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %126 = hivm.hir.pointer_cast(%c4512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %127 = hivm.hir.pointer_cast(%c5024_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%expand_shape_87, %30 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%126 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%127 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %128 = hivm.hir.pointer_cast(%c5536_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %129 = hivm.hir.pointer_cast(%c5568_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%126 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%128 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%129 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_88 = memref.collapse_shape %128 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_89 = memref.subview %104[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %130 = hivm.hir.pointer_cast(%c6080_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_89, %collapse_shape_88 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%130 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %expand_shape_90 = memref.expand_shape %111 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %131 = hivm.hir.pointer_cast(%c6112_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %132 = hivm.hir.pointer_cast(%c6624_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%expand_shape_90, %31 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%131 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%132 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %133 = hivm.hir.pointer_cast(%c7136_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %134 = hivm.hir.pointer_cast(%c7168_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%131 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%133 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%134 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_91 = memref.collapse_shape %133 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_92 = memref.subview %111[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %135 = hivm.hir.pointer_cast(%c7680_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_92, %collapse_shape_91 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%135 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %expand_shape_93 = memref.expand_shape %118 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %136 = hivm.hir.pointer_cast(%c7712_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %137 = hivm.hir.pointer_cast(%c8224_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%expand_shape_93, %32 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%136 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%137 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %138 = hivm.hir.pointer_cast(%c8736_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %139 = hivm.hir.pointer_cast(%c8768_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%136 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%138 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%139 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_94 = memref.collapse_shape %138 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_95 = memref.subview %118[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %140 = hivm.hir.pointer_cast(%c9280_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_95, %collapse_shape_94 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%140 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %expand_shape_96 = memref.expand_shape %125 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %141 = hivm.hir.pointer_cast(%c9312_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %142 = hivm.hir.pointer_cast(%c9824_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%expand_shape_96, %33 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%141 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%142 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %143 = hivm.hir.pointer_cast(%c10336_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %144 = hivm.hir.pointer_cast(%c10368_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%141 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%143 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%144 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_97 = memref.collapse_shape %143 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_98 = memref.subview %125[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %145 = hivm.hir.pointer_cast(%c10880_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_98, %collapse_shape_97 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%145 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      %expand_shape_99 = memref.expand_shape %130 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %146 = hivm.hir.pointer_cast(%c10912_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_99 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%146 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %147 = arith.index_cast %arg9 : i32 to index
      %148 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%147]
      %149 = arith.cmpi slt, %147, %c0 : index
      %150 = arith.cmpi sge, %147, %c16 : index
      %151 = arith.ori %149, %150 : i1
      scf.if %151 {
      } else {
        %subview_103 = memref.subview %30[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_104 = memref.subview %146[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_105 = memref.collapse_shape %subview_103 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_106 = memref.collapse_shape %subview_104 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_105 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_106 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_107 = memref.subview %146[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_108 = memref.subview %30[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_109 = memref.collapse_shape %subview_107 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_110 = memref.collapse_shape %subview_108 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_109 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_110 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
      %expand_shape_100 = memref.expand_shape %135 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %152 = hivm.hir.pointer_cast(%c11424_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_100 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%152 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      scf.if %151 {
      } else {
        %subview_103 = memref.subview %31[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_104 = memref.subview %152[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_105 = memref.collapse_shape %subview_103 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_106 = memref.collapse_shape %subview_104 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_105 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_106 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_107 = memref.subview %152[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_108 = memref.subview %31[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_109 = memref.collapse_shape %subview_107 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_110 = memref.collapse_shape %subview_108 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_109 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_110 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
      %expand_shape_101 = memref.expand_shape %140 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %153 = hivm.hir.pointer_cast(%c11936_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%expand_shape_101 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%153 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      scf.if %151 {
      } else {
        %subview_103 = memref.subview %32[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_104 = memref.subview %153[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_105 = memref.collapse_shape %subview_103 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_106 = memref.collapse_shape %subview_104 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_105 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_106 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_107 = memref.subview %153[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_108 = memref.subview %32[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_109 = memref.collapse_shape %subview_107 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_110 = memref.collapse_shape %subview_108 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_109 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_110 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
      %expand_shape_102 = memref.expand_shape %145 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %154 = hivm.hir.pointer_cast(%c12448_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_102 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%154 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      scf.if %151 {
      } else {
        %subview_103 = memref.subview %33[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_104 = memref.subview %154[0, 0] [%147, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_105 = memref.collapse_shape %subview_103 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_106 = memref.collapse_shape %subview_104 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_105 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_106 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_107 = memref.subview %154[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_108 = memref.subview %33[0, 0] [%148, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_109 = memref.collapse_shape %subview_107 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_110 = memref.collapse_shape %subview_108 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_109 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_110 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %34 = hivm.hir.pointer_cast(%c2208_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_10 = memref.collapse_shape %34 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_6, %collapse_shape_5 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_10 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %35 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_11 = memref.subview %35[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%34 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_11 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_11 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %36 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_12 = memref.subview %36[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%34 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_12 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_12 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %37 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_13 = memref.subview %37[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%34 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_13 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_13 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %38 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %38 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_7, %collapse_shape_5 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_14 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %39 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_15 = memref.subview %39[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%38 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_15 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_15 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %40 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_16 = memref.subview %40[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%38 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_16 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_16 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %41 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_17 = memref.subview %41[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%38 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_17 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_17 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %42 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %42 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_8, %collapse_shape_5 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %43 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_19 = memref.subview %43[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%42 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_19 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_19 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %44 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_20 = memref.subview %44[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%42 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_20 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_20 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %45 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_21 = memref.subview %45[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%42 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_21 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_21 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %46 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %46 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_9, %collapse_shape_5 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %47 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_23 = memref.subview %47[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%46 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_23 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_23 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %48 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_24 = memref.subview %48[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%46 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_24 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %49 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_25 = memref.subview %49[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%46 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_25 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_25 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %50 = arith.addi %27, %c16_i32 : i32
    %51 = arith.maxsi %50, %c0_i32 : i32
    %52 = arith.index_cast %51 : i32 to index
    %53 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%26, %52]
    %54 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%26, %52]
    %55 = arith.addi %27, %c32_i32 : i32
    %56 = arith.maxsi %55, %c0_i32 : i32
    %57 = arith.index_cast %56 : i32 to index
    %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%26, %57]
    %59 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%26, %57]
    %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%26, %57]
    %61 = arith.addi %27, %c48_i32 : i32
    %62 = arith.maxsi %61, %c0_i32 : i32
    %63 = arith.index_cast %62 : i32 to index
    %64 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%26, %63]
    %65 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%26, %63]
    %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%26, %63]
    %67 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 48)>()[%26, %63]
    %68 = memref_ext.alloc_workspace() from %arg2 offset = [%c18432] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_26 = memref.subview %68[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %69 = hivm.hir.pointer_cast(%c12960_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_28 = memref.collapse_shape %69 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape_27 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_28 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %70 = hivm.hir.pointer_cast(%c12960_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %70 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_28, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c12960_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %71 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_29, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %72 = memref_ext.alloc_workspace() from %arg2 offset = [%c19456] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_31 = memref.subview %72[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_32 = memref.collapse_shape %subview_31 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_31 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %73 = memref_ext.alloc_workspace() from %arg2 offset = [%c20480] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_33 = memref.subview %73[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_34 = memref.collapse_shape %subview_33 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_33 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %74 = memref_ext.alloc_workspace() from %arg2 offset = [%c21504] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_35 = memref.subview %74[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %75 = hivm.hir.pointer_cast(%c13472_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %subview_35 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_37 = memref.collapse_shape %75 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%collapse_shape_36 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_37 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %76 = hivm.hir.pointer_cast(%c13472_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_38 = memref.collapse_shape %76 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_37, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_38 : memref<128xf32, #hivm.address_space<ub>>)
    %77 = hivm.hir.pointer_cast(%c13472_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_39 = memref.collapse_shape %77 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_38, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_39 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %78 = memref_ext.alloc_workspace() from %arg2 offset = [%c22528] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_40 = memref.subview %78[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_41 = memref.collapse_shape %subview_40 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_39 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_41 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_40 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %79 = memref_ext.alloc_workspace() from %arg2 offset = [%c23552] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_42 = memref.subview %79[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %80 = hivm.hir.pointer_cast(%c13984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_43 = memref.collapse_shape %subview_42 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_44 = memref.collapse_shape %80 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%collapse_shape_43 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %81 = hivm.hir.pointer_cast(%c13984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %81 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_44, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_45 : memref<128xf32, #hivm.address_space<ub>>)
    %82 = hivm.hir.pointer_cast(%c13984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_46 = memref.collapse_shape %82 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_45, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_46 : memref<128xf32, #hivm.address_space<ub>>)
    %83 = memref_ext.alloc_workspace() from %arg2 offset = [%c24576] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_47 = memref.subview %83[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %84 = hivm.hir.pointer_cast(%c14496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_48 = memref.collapse_shape %subview_47 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_49 = memref.collapse_shape %84 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_48 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_49 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %85 = hivm.hir.pointer_cast(%c14496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_50 = memref.collapse_shape %85 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_49, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_50 : memref<128xf32, #hivm.address_space<ub>>)
    %86 = hivm.hir.pointer_cast(%c14496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_51 = memref.collapse_shape %86 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_50, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_51 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %87 = memref_ext.alloc_workspace() from %arg2 offset = [%c25600] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_52 = memref.subview %87[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_53 = memref.collapse_shape %subview_52 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_51 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_52 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %88 = memref_ext.alloc_workspace() from %arg2 offset = [%c26624] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_54 = memref.subview %88[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %89 = hivm.hir.pointer_cast(%c15008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_56 = memref.collapse_shape %89 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_55 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_56 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %90 = hivm.hir.pointer_cast(%c15008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_57 = memref.collapse_shape %90 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_56, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_57 : memref<128xf32, #hivm.address_space<ub>>)
    %91 = hivm.hir.pointer_cast(%c15008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_58 = memref.collapse_shape %91 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_57, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_58 : memref<128xf32, #hivm.address_space<ub>>)
    %92 = memref_ext.alloc_workspace() from %arg2 offset = [%c27648] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_59 = memref.subview %92[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %93 = hivm.hir.pointer_cast(%c15520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_60 = memref.collapse_shape %subview_59 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_61 = memref.collapse_shape %93 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_60 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_61 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %94 = hivm.hir.pointer_cast(%c15520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_62 = memref.collapse_shape %94 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_61, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_62 : memref<128xf32, #hivm.address_space<ub>>)
    %95 = hivm.hir.pointer_cast(%c15520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_63 = memref.collapse_shape %95 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_62, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_63 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %96 = arith.maxsi %27, %c0_i32 : i32
    %97 = arith.index_cast %96 : i32 to index
    %98 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%26, %97]
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%98], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_64 = memref.subview %reinterpret_cast[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_65 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_66 = memref.subview %reinterpret_cast_65[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_67 = memref.reinterpret_cast %arg4 to offset: [%60], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_68 = memref.subview %reinterpret_cast_67[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_69 = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_70 = memref.subview %reinterpret_cast_69[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_71 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_72 = memref.subview %reinterpret_cast_71[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_73 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_74 = memref.subview %reinterpret_cast_73[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_75 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_76 = memref.subview %reinterpret_cast_75[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_77 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_78 = memref.subview %reinterpret_cast_77[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_79 = memref.reinterpret_cast %arg4 to offset: [%65], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_80 = memref.subview %reinterpret_cast_79[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_81 = memref.reinterpret_cast %arg4 to offset: [%66], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_82 = memref.subview %reinterpret_cast_81[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%34 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_64 : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%38 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_66 : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%42 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_68 : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%46 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_70 : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%71 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%subview_72 : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%86 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%subview_74 : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%77 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%subview_76 : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%95 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%subview_78 : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%91 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%subview_80 : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%82 : memref<8x16xf32, #hivm.address_space<ub>>) outs(%subview_82 : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_64x64_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c2_i64 = arith.constant 2 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c3_i64 = arith.constant 3 : i64
  %c5_i64 = arith.constant 5 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c1_i64_2 = arith.constant 1 : i64
  %c2_i64_3 = arith.constant 2 : i64
  %c3_i64_4 = arith.constant 3 : i64
  %c4_i64 = arith.constant 4 : i64
  %c5_i64_5 = arith.constant 5 : i64
  %c6_i64 = arith.constant 6 : i64
  %c56320_i64 = arith.constant 56320 : i64
  %c27648_i64 = arith.constant 27648 : i64
  %c55296_i64 = arith.constant 55296 : i64
  %c26624_i64 = arith.constant 26624 : i64
  %c54272_i64 = arith.constant 54272 : i64
  %c25600_i64 = arith.constant 25600 : i64
  %c53248_i64 = arith.constant 53248 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c52224_i64 = arith.constant 52224 : i64
  %c23552_i64 = arith.constant 23552 : i64
  %c51200_i64 = arith.constant 51200 : i64
  %c22528_i64 = arith.constant 22528 : i64
  %c50176_i64 = arith.constant 50176 : i64
  %c21504_i64 = arith.constant 21504 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c48128_i64 = arith.constant 48128 : i64
  %c19456_i64 = arith.constant 19456 : i64
  %c47104_i64 = arith.constant 47104 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c46080_i64 = arith.constant 46080 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c45056_i64 = arith.constant 45056 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c44032_i64 = arith.constant 44032 : i64
  %c15360_i64 = arith.constant 15360 : i64
  %c43008_i64 = arith.constant 43008 : i64
  %c14336_i64 = arith.constant 14336 : i64
  %c41984_i64 = arith.constant 41984 : i64
  %c13312_i64 = arith.constant 13312 : i64
  %c40960_i64 = arith.constant 40960 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c39936_i64 = arith.constant 39936 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c38912_i64 = arith.constant 38912 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c37888_i64 = arith.constant 37888 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c36864_i64 = arith.constant 36864 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c35840_i64 = arith.constant 35840 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c31744_i64 = arith.constant 31744 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c30720_i64 = arith.constant 30720 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c29696_i64 = arith.constant 29696 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c28672_i64 = arith.constant 28672 : i64
  %c0_i64_6 = arith.constant 0 : i64
  %c24_i32 = arith.constant 24 : i32
  %false = arith.constant false
  %true = arith.constant true
  %c512_i32 = arith.constant 512 : i32
  %c0_i32 = arith.constant 0 : i32
  %c48_i32 = arith.constant 48 : i32
  %c32_i32 = arith.constant 32 : i32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16_i32 = arith.constant 16 : i32
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
  %c16384 = arith.constant 16384 : index
  %c17408 = arith.constant 17408 : index
  %c18432 = arith.constant 18432 : index
  %c19456 = arith.constant 19456 : index
  %c20480 = arith.constant 20480 : index
  %c21504 = arith.constant 21504 : index
  %c22528 = arith.constant 22528 : index
  %c23552 = arith.constant 23552 : index
  %c24576 = arith.constant 24576 : index
  %c25600 = arith.constant 25600 : index
  %c26624 = arith.constant 26624 : index
  %c27648 = arith.constant 27648 : index
  hivm.hir.set_ffts_base_addr %arg0
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg7, %arg6 : i32
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
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  scf.for %arg8 = %3 to %1 step %c24_i32  : i32 {
    %5 = arith.index_cast %arg8 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c24_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i64
    %11 = arith.index_cast %arg8 : i32 to index
    %12 = arith.index_cast %3 : i32 to index
    %13 = arith.index_cast %1 : i32 to index
    %14 = arith.index_cast %c24_i32 : i32 to index
    %15 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%11, %12, %14]
    %16 = arith.index_cast %15 : index to i1
    %c6_i64_7 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %17 = arith.select %16, %c6_i64_7, %c7_i64 : i64
    %18 = arith.index_cast %15 : index to i1
    %c0_i64_8 = arith.constant 0 : i64
    %c1_i64_9 = arith.constant 1 : i64
    %19 = arith.select %18, %c0_i64_8, %c1_i64_9 : i64
    %20 = arith.index_cast %15 : index to i1
    %c4_i64_10 = arith.constant 4 : i64
    %c5_i64_11 = arith.constant 5 : i64
    %21 = arith.select %20, %c4_i64_10, %c5_i64_11 : i64
    %22 = arith.index_cast %15 : index to i1
    %c2_i64_12 = arith.constant 2 : i64
    %c3_i64_13 = arith.constant 3 : i64
    %23 = arith.select %22, %c2_i64_12, %c3_i64_13 : i64
    %24 = hivm.hir.pointer_cast(%c27648_i64, %c56320_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %25 = hivm.hir.pointer_cast(%c26624_i64, %c55296_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %26 = hivm.hir.pointer_cast(%c25600_i64, %c54272_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %27 = hivm.hir.pointer_cast(%c24576_i64, %c53248_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %28 = hivm.hir.pointer_cast(%c23552_i64, %c52224_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %29 = hivm.hir.pointer_cast(%c22528_i64, %c51200_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %30 = hivm.hir.pointer_cast(%c21504_i64, %c50176_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %31 = hivm.hir.pointer_cast(%c20480_i64, %c49152_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %32 = hivm.hir.pointer_cast(%c19456_i64, %c48128_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %33 = hivm.hir.pointer_cast(%c18432_i64, %c47104_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %34 = hivm.hir.pointer_cast(%c17408_i64, %c46080_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %35 = hivm.hir.pointer_cast(%c16384_i64, %c45056_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %36 = hivm.hir.pointer_cast(%c15360_i64, %c44032_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %37 = hivm.hir.pointer_cast(%c14336_i64, %c43008_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %38 = hivm.hir.pointer_cast(%c13312_i64, %c41984_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %39 = hivm.hir.pointer_cast(%c12288_i64, %c40960_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %40 = hivm.hir.pointer_cast(%c11264_i64, %c39936_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %41 = hivm.hir.pointer_cast(%c10240_i64, %c38912_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %42 = hivm.hir.pointer_cast(%c9216_i64, %c37888_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %43 = hivm.hir.pointer_cast(%c8192_i64, %c36864_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %43 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %44 = hivm.hir.pointer_cast(%c7168_i64, %c35840_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %44 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %45 = hivm.hir.pointer_cast(%c6144_i64, %c34816_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %46 = hivm.hir.pointer_cast(%c5120_i64, %c33792_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %47 = hivm.hir.pointer_cast(%c4096_i64, %c32768_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %47 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %48 = hivm.hir.pointer_cast(%c3072_i64, %c31744_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %48 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %49 = hivm.hir.pointer_cast(%c2048_i64, %c30720_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %49 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %50 = hivm.hir.pointer_cast(%c1024_i64, %c29696_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %50 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %51 = hivm.hir.pointer_cast(%c0_i64_6, %c28672_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %51 {hivm.multi_buffer = 2 : i32} : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_mask_norm
    %52 = arith.divsi %arg8, %arg7 : i32
    %53 = arith.remsi %52, %arg6 : i32
    %54 = arith.divsi %arg8, %4 : i32
    %55 = arith.remsi %54, %arg5 : i32
    %56 = arith.divsi %55, %c4_i32 : i32
    %57 = arith.muli %56, %c4_i32 : i32
    %58 = arith.subi %55, %57 : i32
    %59 = arith.muli %56, %c512_i32 : i32
    %60 = arith.addi %59, %58 : i32
    %61 = arith.muli %60, %c64_i32 : i32
    %62 = arith.index_cast %61 : i32 to index
    %63 = arith.muli %53, %c64_i32 : i32
    %64 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast = memref.cast %51 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %19]
    hivm.hir.nd2nz {dst_continuous} ins(%64 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %65 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_14 = memref.cast %50 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%65 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_14 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %66 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_15 = memref.cast %49 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%66 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_15 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %67 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_16 = memref.cast %48 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%67 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID5>]
    %68 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_17 = memref.cast %47 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%68 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %69 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_18 = memref.cast %46 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%69 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %70 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_19 = memref.cast %45 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%70 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %71 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_20 = memref.cast %44 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%71 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_20 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %72 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_21 = memref.cast %43 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%72 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_21 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %73 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_22 = memref.cast %42 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%73 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_22 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %74 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_23 = memref.cast %41 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%74 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_23 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_24 = memref.cast %40 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%75 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_24 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %76 = arith.addi %63, %c16_i32 : i32
    %77 = arith.maxsi %76, %c0_i32 : i32
    %78 = arith.index_cast %77 : i32 to index
    %79 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%62, %78]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%79], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %80 = arith.addi %63, %c32_i32 : i32
    %81 = arith.maxsi %80, %c0_i32 : i32
    %82 = arith.index_cast %81 : i32 to index
    %83 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%62, %82]
    %reinterpret_cast_25 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %84 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%62, %82]
    %reinterpret_cast_26 = memref.reinterpret_cast %arg3 to offset: [%84], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %85 = arith.addi %63, %c48_i32 : i32
    %86 = arith.maxsi %85, %c0_i32 : i32
    %87 = arith.index_cast %86 : i32 to index
    %88 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%62, %87]
    %reinterpret_cast_27 = memref.reinterpret_cast %arg3 to offset: [%88], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %89 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%62, %87]
    %reinterpret_cast_28 = memref.reinterpret_cast %arg3 to offset: [%89], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %90 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%62, %87]
    %reinterpret_cast_29 = memref.reinterpret_cast %arg3 to offset: [%90], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_30 = memref.cast %39 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_30 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID6>]
    %cast_31 = memref.cast %38 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_25 : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_31 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID4>]
    %cast_32 = memref.cast %37 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_26 : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_32 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
    %cast_33 = memref.cast %36 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_27 : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_33 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
    %cast_34 = memref.cast %35 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_28 : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_34 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %cast_35 = memref.cast %34 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_29 : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_35 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %91 = hivm.hir.pointer_cast(%c0_i64_6) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_36 = memref.cast %91 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_16, %cast_30, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_36 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c5_i64_5, %c6_i64, %c-1_i64, %c-1_i64, %10, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %92 = memref_ext.alloc_workspace() from %arg2 offset = [%c12288] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_36 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%92 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_37 = memref.cast %33 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%92 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_37 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    %93 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_38 = memref.cast %93 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_39 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_19, %cast_31, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_38 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_39, %c4_i64, %c-1_i64_39, %c-1_i64_39, %10, %c-1_i64_39, %c-1_i64_39 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %94 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_38 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%94 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_40 = memref.cast %32 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%94 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_40 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
    %95 = hivm.hir.pointer_cast(%c2048_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_41 = memref.cast %95 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_42 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_20, %cast_32, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_41 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_42, %c3_i64_4, %c-1_i64_42, %c-1_i64_42, %10, %c-1_i64_42, %c-1_i64_42 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %96 = memref_ext.alloc_workspace() from %arg2 offset = [%c14336] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_41 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%96 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_43 = memref.cast %31 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%96 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_43 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
    %97 = hivm.hir.pointer_cast(%c3072_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_44 = memref.cast %97 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_45 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_22, %cast_33, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_44 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_45, %c2_i64_3, %c-1_i64_45, %c-1_i64_45, %10, %c-1_i64_45, %c-1_i64_45 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %98 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_44 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%98 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_46 = memref.cast %30 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%98 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_46 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
    %99 = hivm.hir.pointer_cast(%c4096_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_47 = memref.cast %99 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_48 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_23, %cast_34, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_47 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_48, %c1_i64_2, %c-1_i64_48, %c-1_i64_48, %10, %c-1_i64_48, %c-1_i64_48 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %100 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_47 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%100 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_49 = memref.cast %29 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %23]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%100 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_49 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
    %101 = hivm.hir.pointer_cast(%c5120_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_50 = memref.cast %101 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_51 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_24, %cast_35, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_50 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_51, %c0_i64_1, %c-1_i64_51, %c-1_i64_51, %10, %c-1_i64_51, %c-1_i64_51 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %102 = memref_ext.alloc_workspace() from %arg2 offset = [%c17408] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_50 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%102 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %cast_52 = memref.cast %28 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %21]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%102 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_52 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
    %103 = hivm.hir.pointer_cast(%c6144_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_53 = memref.cast %103 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %c-1_i64_54 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_37, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_53 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c5_i64, %c-1_i64_54, %c-1_i64_54, %c-1_i64_54, %10, %c-1_i64_54, %c-1_i64_54 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %104 = memref_ext.alloc_workspace() from %arg2 offset = [%c18432] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_53 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%104 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    annotation.mark %104 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %105 = memref_ext.alloc_workspace() from %arg2 offset = [%c19456] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_55 = memref.cast %27 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%105 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_55 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %106 = memref_ext.alloc_workspace() from %arg2 offset = [%c20480] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_56 = memref.cast %26 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%106 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_56 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %107 = hivm.hir.pointer_cast(%c7168_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_57 = memref.cast %107 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
    %c-1_i64_58 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_43, %cast_17, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_57 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c3_i64, %c-1_i64_58, %c-1_i64_58, %c-1_i64_58, %10, %c-1_i64_58, %c-1_i64_58 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %108 = memref_ext.alloc_workspace() from %arg2 offset = [%c21504] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_57 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%108 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
    annotation.mark %108 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %109 = memref_ext.alloc_workspace() from %arg2 offset = [%c22528] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_59 = memref.cast %25 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%109 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_59 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
    %110 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_60 = memref.cast %110 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
    %c-1_i64_61 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_52, %cast_21, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_60 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c-1_i64_61, %c-1_i64_61, %c-1_i64_61, %10, %c-1_i64_61, %c-1_i64_61 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %111 = memref_ext.alloc_workspace() from %arg2 offset = [%c23552] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_60 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%111 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
    annotation.mark %111 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %112 = hivm.hir.pointer_cast(%c9216_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_62 = memref.cast %112 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
    hivm.hir.mmadL1 ins(%cast_40, %cast_14, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_62 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    %c-1_i64_63 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_43, %cast_55, %false, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_62 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_63, %c1_i64, %c-1_i64_63, %c-1_i64_63, %10, %c-1_i64_63, %c-1_i64_63 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %113 = memref_ext.alloc_workspace() from %arg2 offset = [%c24576] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_62 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%113 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
    annotation.mark %113 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %114 = memref_ext.alloc_workspace() from %arg2 offset = [%c25600] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_64 = memref.cast %24 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %17]
    hivm.hir.nd2nz {dst_continuous} ins(%114 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_64 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %115 = hivm.hir.pointer_cast(%c10240_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_65 = memref.cast %115 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    hivm.hir.mmadL1 ins(%cast_49, %cast_18, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_65 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    %c-1_i64_66 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_52, %cast_59, %false, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_65 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_66, %c2_i64, %c-1_i64_66, %c-1_i64_66, %10, %c-1_i64_66, %c-1_i64_66 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %116 = memref_ext.alloc_workspace() from %arg2 offset = [%c26624] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_65 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%116 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    annotation.mark %116 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %117 = hivm.hir.pointer_cast(%c11264_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_67 = memref.cast %117 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    %c-1_i64_68 = arith.constant -1 : i64
    hivm.hir.mmadL1 ins(%cast_46, %cast_15, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_67 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_68, %c-1_i64_68, %c-1_i64_68, %19, %10, %c-1_i64_68, %c-1_i64_68 : i64, i64, i64, i64, i64, i64, i64)
    %c-1_i64_69 = arith.constant -1 : i64
    hivm.hir.mmadL1 ins(%cast_49, %cast_56, %false, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_67 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_69, %c-1_i64_69, %23, %c-1_i64_69, %10, %c-1_i64_69, %c-1_i64_69 : i64, i64, i64, i64, i64, i64, i64)
    %c-1_i64_70 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_52, %cast_64, %false, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_67 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_70, %c0_i64, %21, %17, %10, %c-1_i64_70, %c-1_i64_70 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %118 = memref_ext.alloc_workspace() from %arg2 offset = [%c27648] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_67 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%118 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    annotation.mark %118 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
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
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
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
