// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_64x64_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c18336_i64 = arith.constant 18336 : i64
  %c17824_i64 = arith.constant 17824 : i64
  %c17312_i64 = arith.constant 17312 : i64
  %c16800_i64 = arith.constant 16800 : i64
  %c16288_i64 = arith.constant 16288 : i64
  %c15776_i64 = arith.constant 15776 : i64
  %c15264_i64 = arith.constant 15264 : i64
  %c15232_i64 = arith.constant 15232 : i64
  %c14720_i64 = arith.constant 14720 : i64
  %c14688_i64 = arith.constant 14688 : i64
  %c14176_i64 = arith.constant 14176 : i64
  %c13664_i64 = arith.constant 13664 : i64
  %c13600_i64 = arith.constant 13600 : i64
  %c13088_i64 = arith.constant 13088 : i64
  %c13056_i64 = arith.constant 13056 : i64
  %c12544_i64 = arith.constant 12544 : i64
  %c12512_i64 = arith.constant 12512 : i64
  %c12000_i64 = arith.constant 12000 : i64
  %c11488_i64 = arith.constant 11488 : i64
  %c11424_i64 = arith.constant 11424 : i64
  %c10912_i64 = arith.constant 10912 : i64
  %c10880_i64 = arith.constant 10880 : i64
  %c10368_i64 = arith.constant 10368 : i64
  %c10336_i64 = arith.constant 10336 : i64
  %c9824_i64 = arith.constant 9824 : i64
  %c9312_i64 = arith.constant 9312 : i64
  %c9248_i64 = arith.constant 9248 : i64
  %c8736_i64 = arith.constant 8736 : i64
  %c8704_i64 = arith.constant 8704 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c8160_i64 = arith.constant 8160 : i64
  %c7648_i64 = arith.constant 7648 : i64
  %c7136_i64 = arith.constant 7136 : i64
  %c7072_i64 = arith.constant 7072 : i64
  %c7008_i64 = arith.constant 7008 : i64
  %c6496_i64 = arith.constant 6496 : i64
  %c6432_i64 = arith.constant 6432 : i64
  %c5920_i64 = arith.constant 5920 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c5344_i64 = arith.constant 5344 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c4768_i64 = arith.constant 4768 : i64
  %c4256_i64 = arith.constant 4256 : i64
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
  %c27648 = arith.constant 27648 : index
  %c25600 = arith.constant 25600 : index
  %c23552 = arith.constant 23552 : index
  %c22528 = arith.constant 22528 : index
  %c20480 = arith.constant 20480 : index
  %c18432 = arith.constant 18432 : index
  %c17408 = arith.constant 17408 : index
  %c15360 = arith.constant 15360 : index
  %c14336 = arith.constant 14336 : index
  %c13312 = arith.constant 13312 : index
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
  %c48_i32 = arith.constant 48 : i32
  %c32_i32 = arith.constant 32 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16 = arith.constant 16 : index
  %c2_i32 = arith.constant 2 : i32
  %c256_i32 = arith.constant 256 : i32
  %c18_i32 = arith.constant 18 : i32
  %c34_i32 = arith.constant 34 : i32
  %c50_i32 = arith.constant 50 : i32
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
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg9, %arg8 : i32
    %21 = arith.remsi %20, %arg7 : i32
    %22 = arith.divsi %arg9, %4 : i32
    %23 = arith.remsi %22, %arg6 : i32
    %24 = arith.divsi %21, %c4_i32 : i32
    %25 = arith.remsi %21, %c4_i32 : i32
    %26 = arith.muli %24, %arg5 : i32
    %27 = arith.muli %26, %c4_i32 : i32
    %28 = arith.addi %27, %25 : i32
    %29 = arith.muli %28, %c64_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %23, %c64_i32 : i32
    %32 = arith.maxsi %31, %c0_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%30, %33]
    %35 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %33]
    %36 = affine.min affine_map<()[s0] -> (16, s0)>()[%35]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %37 = arith.addi %31, %c16_i32 : i32
    %38 = arith.maxsi %37, %c0_i32 : i32
    %39 = arith.index_cast %38 : i32 to index
    %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%30, %39]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%30, %39]
    %42 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %39]
    %43 = affine.min affine_map<()[s0] -> (16, s0)>()[%42]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %44 = arith.addi %31, %c32_i32 : i32
    %45 = arith.maxsi %44, %c0_i32 : i32
    %46 = arith.index_cast %45 : i32 to index
    %47 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%30, %46]
    %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%30, %46]
    %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%30, %46]
    %50 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %46]
    %51 = affine.min affine_map<()[s0] -> (16, s0)>()[%50]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %52 = arith.addi %31, %c48_i32 : i32
    %53 = arith.maxsi %52, %c0_i32 : i32
    %54 = arith.index_cast %53 : i32 to index
    %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%30, %54]
    %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 48)>()[%30, %54]
    %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%30, %54]
    %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%30, %54]
    %59 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%8, %54]
    %60 = affine.min affine_map<()[s0] -> (16, s0)>()[%59]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %61 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %62 = arith.subi %c0_i32, %31 : i32
    %63 = arith.maxsi %62, %c0_i32 : i32
    %64 = arith.index_cast %63 : i32 to index
    %65 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%64, %35]
    %66 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%36, %65]
    %67 = arith.cmpi slt, %66, %c16 : index
    %subview_11 = memref.subview %reinterpret_cast[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %subview_11[0, 0] [%66, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_13 = memref.subview %61[%65, 0] [%66, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %67 {
      %collapse_shape_133 = memref.collapse_shape %61 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_133 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%subview_12 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    %68 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %69 = arith.subi %c-16_i32, %31 : i32
    %70 = arith.maxsi %69, %c0_i32 : i32
    %71 = arith.index_cast %70 : i32 to index
    %72 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%71, %42]
    %73 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%43, %72]
    %74 = arith.cmpi slt, %73, %c16 : index
    %subview_14 = memref.subview %reinterpret_cast_8[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_15 = memref.subview %subview_14[0, 0] [%73, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_16 = memref.subview %68[%72, 0] [%73, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %74 {
      %collapse_shape_133 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_133 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.load ins(%subview_15 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_16 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    %75 = hivm.hir.pointer_cast(%c3744_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %76 = arith.subi %c-32_i32, %31 : i32
    %77 = arith.maxsi %76, %c0_i32 : i32
    %78 = arith.index_cast %77 : i32 to index
    %79 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%78, %50]
    %80 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%51, %79]
    %81 = arith.cmpi slt, %80, %c16 : index
    %subview_17 = memref.subview %reinterpret_cast_9[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %subview_17[0, 0] [%80, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_19 = memref.subview %75[%79, 0] [%80, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %81 {
      %collapse_shape_133 = memref.collapse_shape %75 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_133 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.load ins(%subview_18 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_19 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %82 = hivm.hir.pointer_cast(%c4256_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %83 = arith.subi %c-48_i32, %31 : i32
    %84 = arith.maxsi %83, %c0_i32 : i32
    %85 = arith.index_cast %84 : i32 to index
    %86 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%85, %59]
    %87 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%60, %86]
    %88 = arith.cmpi slt, %87, %c16 : index
    %subview_20 = memref.subview %reinterpret_cast_10[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_21 = memref.subview %subview_20[0, 0] [%87, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_22 = memref.subview %82[%86, 0] [%87, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %88 {
      %collapse_shape_133 = memref.collapse_shape %82 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_133 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.load ins(%subview_21 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_22 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %89 = hivm.hir.pointer_cast(%c4768_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %61 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %89 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %90 = hivm.hir.pointer_cast(%c5280_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_23, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_24 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%90 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %91 = hivm.hir.pointer_cast(%c4768_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %91 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_24, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>>)
    %92 = hivm.hir.pointer_cast(%c4768_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %92 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_25, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_26 : memref<128xf32, #hivm.address_space<ub>>)
    %93 = hivm.hir.pointer_cast(%c5344_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %68 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %93 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %94 = hivm.hir.pointer_cast(%c5856_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_27, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_28 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%94 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    %95 = hivm.hir.pointer_cast(%c5344_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %95 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_28, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_29 : memref<128xf32, #hivm.address_space<ub>>)
    %96 = hivm.hir.pointer_cast(%c5344_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %96 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_29, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<128xf32, #hivm.address_space<ub>>)
    %97 = hivm.hir.pointer_cast(%c5920_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %75 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %97 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %98 = hivm.hir.pointer_cast(%c6432_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_31, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_32 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%98 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    %99 = hivm.hir.pointer_cast(%c5920_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_33 = memref.collapse_shape %99 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_32, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_33 : memref<128xf32, #hivm.address_space<ub>>)
    %100 = hivm.hir.pointer_cast(%c5920_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %100 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_33, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_34 : memref<128xf32, #hivm.address_space<ub>>)
    %101 = hivm.hir.pointer_cast(%c6496_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_35 = memref.collapse_shape %82 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %101 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %102 = hivm.hir.pointer_cast(%c7008_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%collapse_shape_4, %collapse_shape_35, %cst : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_36 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%102 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    %103 = hivm.hir.pointer_cast(%c6496_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_37 = memref.collapse_shape %103 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_36, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_37 : memref<128xf32, #hivm.address_space<ub>>)
    %104 = hivm.hir.pointer_cast(%c6496_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_38 = memref.collapse_shape %104 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_37, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_38 : memref<128xf32, #hivm.address_space<ub>>)
    %105 = arith.subi %arg5, %31 : i32
    %106 = arith.minsi %105, %c16_i32 : i32
    scf.for %arg10 = %c2_i32 to %106 step %c1_i32  : i32 {
      %175 = arith.addi %31, %arg10 : i32
      %176 = arith.muli %175, %c256_i32 : i32
      %177 = arith.index_cast %176 : i32 to index
      %178 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %177]
      %reinterpret_cast_133 = memref.reinterpret_cast %arg3 to offset: [%178], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %179 = hivm.hir.pointer_cast(%c7072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_133 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%179 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %180 = hivm.hir.pointer_cast(%c7072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%179, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%180 : memref<16xf32, #hivm.address_space<ub>>)
      %181 = hivm.hir.pointer_cast(%c7072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%180, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%181 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_134 = memref.expand_shape %181 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %182 = hivm.hir.pointer_cast(%c7136_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %183 = hivm.hir.pointer_cast(%c7648_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_134, %92 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%182 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%183 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %184 = hivm.hir.pointer_cast(%c8160_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %185 = hivm.hir.pointer_cast(%c8192_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%182 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%184 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%185 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_135 = memref.collapse_shape %184 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_136 = memref.subview %181[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %186 = hivm.hir.pointer_cast(%c8704_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_136, %collapse_shape_135 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%186 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %expand_shape_137 = memref.expand_shape %186 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %187 = hivm.hir.pointer_cast(%c8736_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_137 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%187 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %188 = arith.index_cast %arg10 : i32 to index
      %189 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%188]
      %190 = arith.cmpi slt, %188, %c0 : index
      %191 = arith.cmpi sge, %188, %c16 : index
      %192 = arith.ori %190, %191 : i1
      scf.if %192 {
      } else {
        %subview_138 = memref.subview %92[0, 0] [%188, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_139 = memref.subview %187[0, 0] [%188, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_140 = memref.collapse_shape %subview_138 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_141 = memref.collapse_shape %subview_139 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_140 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_141 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_142 = memref.subview %187[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_143 = memref.subview %92[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_144 = memref.collapse_shape %subview_142 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_145 = memref.collapse_shape %subview_143 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_144 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_145 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %107 = arith.minsi %105, %c32_i32 : i32
    scf.for %arg10 = %c18_i32 to %107 step %c1_i32  : i32 {
      %175 = arith.addi %31, %arg10 : i32
      %176 = arith.muli %175, %c256_i32 : i32
      %177 = arith.index_cast %176 : i32 to index
      %178 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 16)>()[%30, %177]
      %reinterpret_cast_133 = memref.reinterpret_cast %arg3 to offset: [%178], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %179 = hivm.hir.pointer_cast(%c9248_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_133 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%179 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %180 = hivm.hir.pointer_cast(%c9248_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%179, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%180 : memref<16xf32, #hivm.address_space<ub>>)
      %181 = hivm.hir.pointer_cast(%c9248_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%180, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%181 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_134 = memref.expand_shape %181 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %182 = hivm.hir.pointer_cast(%c9312_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %183 = hivm.hir.pointer_cast(%c9824_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_134, %96 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%182 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%183 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %184 = hivm.hir.pointer_cast(%c10336_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %185 = hivm.hir.pointer_cast(%c10368_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%182 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%184 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%185 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_135 = memref.collapse_shape %184 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_136 = memref.subview %181[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %186 = hivm.hir.pointer_cast(%c10880_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_136, %collapse_shape_135 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%186 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %187 = arith.subi %arg10, %c16_i32 : i32
      %expand_shape_137 = memref.expand_shape %186 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %188 = hivm.hir.pointer_cast(%c10912_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_137 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%188 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %189 = arith.index_cast %187 : i32 to index
      %190 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%189]
      %191 = arith.cmpi slt, %189, %c0 : index
      %192 = arith.cmpi sge, %189, %c16 : index
      %193 = arith.ori %191, %192 : i1
      scf.if %193 {
      } else {
        %subview_138 = memref.subview %96[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_139 = memref.subview %188[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_140 = memref.collapse_shape %subview_138 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_141 = memref.collapse_shape %subview_139 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_140 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_141 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_142 = memref.subview %188[0, 0] [%190, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_143 = memref.subview %96[0, 0] [%190, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_144 = memref.collapse_shape %subview_142 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_145 = memref.collapse_shape %subview_143 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_144 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_145 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %108 = arith.minsi %105, %c48_i32 : i32
    scf.for %arg10 = %c34_i32 to %108 step %c1_i32  : i32 {
      %175 = arith.addi %31, %arg10 : i32
      %176 = arith.muli %175, %c256_i32 : i32
      %177 = arith.index_cast %176 : i32 to index
      %178 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 32)>()[%30, %177]
      %reinterpret_cast_133 = memref.reinterpret_cast %arg3 to offset: [%178], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %179 = hivm.hir.pointer_cast(%c11424_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_133 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%179 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %180 = hivm.hir.pointer_cast(%c11424_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%179, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%180 : memref<16xf32, #hivm.address_space<ub>>)
      %181 = hivm.hir.pointer_cast(%c11424_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%180, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%181 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_134 = memref.expand_shape %181 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %182 = hivm.hir.pointer_cast(%c11488_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %183 = hivm.hir.pointer_cast(%c12000_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_134, %100 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%182 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%183 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %184 = hivm.hir.pointer_cast(%c12512_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %185 = hivm.hir.pointer_cast(%c12544_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%182 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%184 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%185 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_135 = memref.collapse_shape %184 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_136 = memref.subview %181[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %186 = hivm.hir.pointer_cast(%c13056_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_136, %collapse_shape_135 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%186 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %187 = arith.subi %arg10, %c32_i32 : i32
      %expand_shape_137 = memref.expand_shape %186 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %188 = hivm.hir.pointer_cast(%c13088_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_137 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%188 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %189 = arith.index_cast %187 : i32 to index
      %190 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%189]
      %191 = arith.cmpi slt, %189, %c0 : index
      %192 = arith.cmpi sge, %189, %c16 : index
      %193 = arith.ori %191, %192 : i1
      scf.if %193 {
      } else {
        %subview_138 = memref.subview %100[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_139 = memref.subview %188[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_140 = memref.collapse_shape %subview_138 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_141 = memref.collapse_shape %subview_139 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_140 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_141 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_142 = memref.subview %188[0, 0] [%190, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_143 = memref.subview %100[0, 0] [%190, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_144 = memref.collapse_shape %subview_142 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_145 = memref.collapse_shape %subview_143 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_144 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_145 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %109 = arith.minsi %105, %c64_i32 : i32
    scf.for %arg10 = %c50_i32 to %109 step %c1_i32  : i32 {
      %175 = arith.addi %31, %arg10 : i32
      %176 = arith.muli %175, %c256_i32 : i32
      %177 = arith.index_cast %176 : i32 to index
      %178 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 48)>()[%30, %177]
      %reinterpret_cast_133 = memref.reinterpret_cast %arg3 to offset: [%178], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %179 = hivm.hir.pointer_cast(%c13600_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%reinterpret_cast_133 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%179 : memref<16xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %180 = hivm.hir.pointer_cast(%c13600_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%179, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%180 : memref<16xf32, #hivm.address_space<ub>>)
      %181 = hivm.hir.pointer_cast(%c13600_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%180, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%181 : memref<16xf32, #hivm.address_space<ub>>)
      %expand_shape_134 = memref.expand_shape %181 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %182 = hivm.hir.pointer_cast(%c13664_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %183 = hivm.hir.pointer_cast(%c14176_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%expand_shape_134, %104 : memref<16x1xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%182 : memref<16x8xf32, #hivm.address_space<ub>>) temp_buffer(%183 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
      %184 = hivm.hir.pointer_cast(%c14688_i64) : memref<1x8xf32, #hivm.address_space<ub>>
      %185 = hivm.hir.pointer_cast(%c14720_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%182 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%184 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%185 : memref<128xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_135 = memref.collapse_shape %184 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %subview_136 = memref.subview %181[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %186 = hivm.hir.pointer_cast(%c15232_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%subview_136, %collapse_shape_135 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%186 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %187 = arith.subi %arg10, %c48_i32 : i32
      %expand_shape_137 = memref.expand_shape %186 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
      %188 = hivm.hir.pointer_cast(%c15264_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_137 : memref<1x8xf32, #hivm.address_space<ub>>) outs(%188 : memref<16x8xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
      %189 = arith.index_cast %187 : i32 to index
      %190 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%189]
      %191 = arith.cmpi slt, %189, %c0 : index
      %192 = arith.cmpi sge, %189, %c16 : index
      %193 = arith.ori %191, %192 : i1
      scf.if %193 {
      } else {
        %subview_138 = memref.subview %104[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_139 = memref.subview %188[0, 0] [%189, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_140 = memref.collapse_shape %subview_138 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_141 = memref.collapse_shape %subview_139 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_140 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_141 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %subview_142 = memref.subview %188[0, 0] [%190, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_143 = memref.subview %104[0, 0] [%190, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %collapse_shape_144 = memref.collapse_shape %subview_142 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %collapse_shape_145 = memref.collapse_shape %subview_143 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%collapse_shape_144 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_145 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
    } {pipeline.veconly}
    %110 = hivm.hir.pointer_cast(%c4768_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_39 = memref.collapse_shape %110 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_26, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_39 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %111 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_40 = memref.subview %111[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%110 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_40 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_40 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %112 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_41 = memref.subview %112[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%110 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_41 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_41 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %113 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_42 = memref.subview %113[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%110 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_42 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_42 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %114 = hivm.hir.pointer_cast(%c5344_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_43 = memref.collapse_shape %114 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_30, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_43 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %115 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_44 = memref.subview %115[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%114 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_44 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_44 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %116 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_45 = memref.subview %116[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%114 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_45 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_45 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %117 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_46 = memref.subview %117[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%114 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_46 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_46 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %118 = hivm.hir.pointer_cast(%c5920_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_47 = memref.collapse_shape %118 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_34, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_47 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %119 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_48 = memref.subview %119[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%118 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_48 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_48 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %120 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_49 = memref.subview %120[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%118 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_49 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_49 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %121 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_50 = memref.subview %121[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%118 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_50 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_50 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %122 = hivm.hir.pointer_cast(%c6496_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_51 = memref.collapse_shape %122 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_38, %collapse_shape_7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_51 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %123 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_52 = memref.subview %123[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%122 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_52 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_52 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %124 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_53 = memref.subview %124[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%122 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_53 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_53 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %125 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_54 = memref.subview %125[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%122 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%subview_54 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_54 : memref<16x8xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %126 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_55 = memref.subview %126[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %127 = hivm.hir.pointer_cast(%c15776_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_56 = memref.collapse_shape %subview_55 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_57 = memref.collapse_shape %127 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape_56 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_57 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %128 = hivm.hir.pointer_cast(%c15776_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_58 = memref.collapse_shape %128 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_57, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_58 : memref<128xf32, #hivm.address_space<ub>>)
    %129 = hivm.hir.pointer_cast(%c15776_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_59 = memref.collapse_shape %129 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_58, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_59 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %130 = memref_ext.alloc_workspace() from %arg2 offset = [%c14336] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_60 = memref.subview %130[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_61 = memref.collapse_shape %subview_60 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_59 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_61 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_60 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %131 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_62 = memref.subview %131[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_63 = memref.collapse_shape %subview_62 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_59 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_63 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_62 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %132 = memref_ext.alloc_workspace() from %arg2 offset = [%c17408] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_64 = memref.subview %132[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %133 = hivm.hir.pointer_cast(%c16288_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_65 = memref.collapse_shape %subview_64 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_66 = memref.collapse_shape %133 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%collapse_shape_65 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_66 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %134 = hivm.hir.pointer_cast(%c16288_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_67 = memref.collapse_shape %134 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_66, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_67 : memref<128xf32, #hivm.address_space<ub>>)
    %135 = hivm.hir.pointer_cast(%c16288_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_68 = memref.collapse_shape %135 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_67, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_68 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %136 = memref_ext.alloc_workspace() from %arg2 offset = [%c18432] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_69 = memref.subview %136[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_70 = memref.collapse_shape %subview_69 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_68 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_70 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_69 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %137 = memref_ext.alloc_workspace() from %arg2 offset = [%c20480] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_71 = memref.subview %137[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %138 = hivm.hir.pointer_cast(%c16800_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_72 = memref.collapse_shape %subview_71 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_73 = memref.collapse_shape %138 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%collapse_shape_72 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_73 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %139 = hivm.hir.pointer_cast(%c16800_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_74 = memref.collapse_shape %139 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_73, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_74 : memref<128xf32, #hivm.address_space<ub>>)
    %140 = hivm.hir.pointer_cast(%c16800_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_75 = memref.collapse_shape %140 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_74, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_75 : memref<128xf32, #hivm.address_space<ub>>)
    %141 = memref_ext.alloc_workspace() from %arg2 offset = [%c22528] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_76 = memref.subview %141[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %142 = hivm.hir.pointer_cast(%c17312_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_77 = memref.collapse_shape %subview_76 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_78 = memref.collapse_shape %142 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_77 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_78 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %143 = hivm.hir.pointer_cast(%c17312_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_79 = memref.collapse_shape %143 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_78, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_79 : memref<128xf32, #hivm.address_space<ub>>)
    %144 = hivm.hir.pointer_cast(%c17312_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_80 = memref.collapse_shape %144 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_79, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_80 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %145 = memref_ext.alloc_workspace() from %arg2 offset = [%c23552] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    %subview_81 = memref.subview %145[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_82 = memref.collapse_shape %subview_81 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_80 : memref<128xf32, #hivm.address_space<ub>>) outs(%collapse_shape_82 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {"inserted-store", tiled_op}
    annotation.mark %subview_81 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %146 = memref_ext.alloc_workspace() from %arg2 offset = [%c25600] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_83 = memref.subview %146[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %147 = hivm.hir.pointer_cast(%c17824_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_84 = memref.collapse_shape %subview_83 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_85 = memref.collapse_shape %147 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_84 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_85 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %148 = hivm.hir.pointer_cast(%c17824_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_86 = memref.collapse_shape %148 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_85, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_86 : memref<128xf32, #hivm.address_space<ub>>)
    %149 = hivm.hir.pointer_cast(%c17824_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_87 = memref.collapse_shape %149 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_86, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_87 : memref<128xf32, #hivm.address_space<ub>>)
    %150 = memref_ext.alloc_workspace() from %arg2 offset = [%c27648] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %subview_88 = memref.subview %150[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %151 = hivm.hir.pointer_cast(%c18336_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_89 = memref.collapse_shape %subview_88 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_90 = memref.collapse_shape %151 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_89 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_90 : memref<128xf32, #hivm.address_space<ub>>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %152 = hivm.hir.pointer_cast(%c18336_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_91 = memref.collapse_shape %152 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_90, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_91 : memref<128xf32, #hivm.address_space<ub>>)
    %153 = hivm.hir.pointer_cast(%c18336_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %collapse_shape_92 = memref.collapse_shape %153 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_91, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_92 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_93 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_94 = memref.subview %reinterpret_cast_93[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_95 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_96 = memref.subview %reinterpret_cast_95[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_97 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_98 = memref.subview %reinterpret_cast_97[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_99 = memref.reinterpret_cast %arg4 to offset: [%56], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_100 = memref.subview %reinterpret_cast_99[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_101 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_102 = memref.subview %reinterpret_cast_101[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_103 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_104 = memref.subview %reinterpret_cast_103[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_105 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_106 = memref.subview %reinterpret_cast_105[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_107 = memref.reinterpret_cast %arg4 to offset: [%55], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_108 = memref.subview %reinterpret_cast_107[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_109 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_110 = memref.subview %reinterpret_cast_109[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_111 = memref.reinterpret_cast %arg4 to offset: [%57], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_112 = memref.subview %reinterpret_cast_111[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_113 = memref.subview %110[%65, 0] [%66, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_114 = memref.subview %subview_94[0, 0] [%66, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_113 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_114 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %subview_115 = memref.subview %114[%72, 0] [%73, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_116 = memref.subview %subview_96[0, 0] [%73, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_115 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_116 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    %subview_117 = memref.subview %118[%79, 0] [%80, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_118 = memref.subview %subview_98[0, 0] [%80, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_117 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_118 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    %subview_119 = memref.subview %122[%86, 0] [%87, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_120 = memref.subview %subview_100[0, 0] [%87, 8] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_119 : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_120 : memref<?x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    %154 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%72, %10]
    %155 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%42, %10]
    %156 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%155, %72, %10]
    %157 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%156, %154]
    %158 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%154, %10]
    %subview_121 = memref.subview %129[%158, 0] [%157, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %159 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %43, %72]
    %160 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%159, %43, %72]
    %subview_122 = memref.subview %subview_102[0, 0] [%160, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_121 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_122 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    %161 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%79, %10]
    %162 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%50, %10]
    %163 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%162, %79, %10]
    %164 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%163, %161]
    %165 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%161, %10]
    %subview_123 = memref.subview %144[%165, 0] [%164, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %166 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %51, %79]
    %167 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%166, %51, %79]
    %subview_124 = memref.subview %subview_104[0, 0] [%167, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_123 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_124 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    %subview_125 = memref.subview %135[%165, 0] [%164, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_126 = memref.subview %subview_106[0, 0] [%167, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_125 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_126 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    %168 = affine.max affine_map<()[s0, s1] -> (s1 * 8, s0)>()[%86, %10]
    %169 = affine.min affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>()[%59, %10]
    %170 = affine.max affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>()[%169, %86, %10]
    %171 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%170, %168]
    %172 = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 8)>()[%168, %10]
    %subview_127 = memref.subview %153[%172, 0] [%171, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %173 = affine.min affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>()[%10, %60, %86]
    %174 = affine.min affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>()[%173, %60, %86]
    %subview_128 = memref.subview %subview_108[0, 0] [%174, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_127 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_128 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    %subview_129 = memref.subview %149[%172, 0] [%171, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_130 = memref.subview %subview_110[0, 0] [%174, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_129 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_130 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    %subview_131 = memref.subview %140[%172, 0] [%171, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_132 = memref.subview %subview_112[0, 0] [%174, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_131 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_132 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
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

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @merge_16x16_to_64x64_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c4_i64 = arith.constant 4 : i64
  %c3_i64 = arith.constant 3 : i64
  %c2_i64 = arith.constant 2 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c0_i64_2 = arith.constant 0 : i64
  %c0_i64_3 = arith.constant 0 : i64
  %c0_i64_4 = arith.constant 0 : i64
  %c0_i64_5 = arith.constant 0 : i64
  %c0_i64_6 = arith.constant 0 : i64
  %c1_i64_7 = arith.constant 1 : i64
  %c27648_i64 = arith.constant 27648 : i64
  %c26624_i64 = arith.constant 26624 : i64
  %c25600_i64 = arith.constant 25600 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c23552_i64 = arith.constant 23552 : i64
  %c22528_i64 = arith.constant 22528 : i64
  %c21504_i64 = arith.constant 21504 : i64
  %c20480_i64 = arith.constant 20480 : i64
  %c19456_i64 = arith.constant 19456 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c15360_i64 = arith.constant 15360 : i64
  %c14336_i64 = arith.constant 14336 : i64
  %c13312_i64 = arith.constant 13312 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64_8 = arith.constant 0 : i64
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
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
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
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
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
    hivm.hir.set_mask_norm
    %12 = arith.divsi %arg9, %arg8 : i32
    %13 = arith.remsi %12, %arg7 : i32
    %14 = arith.divsi %arg9, %4 : i32
    %15 = arith.remsi %14, %arg6 : i32
    %16 = arith.divsi %13, %c4_i32 : i32
    %17 = arith.remsi %13, %c4_i32 : i32
    %18 = arith.muli %16, %arg5 : i32
    %19 = arith.muli %18, %c4_i32 : i32
    %20 = arith.addi %19, %17 : i32
    %21 = arith.muli %20, %c64_i32 : i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = arith.muli %15, %c64_i32 : i32
    %24 = arith.addi %23, %c16_i32 : i32
    %25 = arith.maxsi %24, %c0_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%22, %26]
    %28 = arith.addi %23, %c32_i32 : i32
    %29 = arith.maxsi %28, %c0_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%22, %30]
    %32 = arith.addi %23, %c48_i32 : i32
    %33 = arith.maxsi %32, %c0_i32 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%22, %34]
    %36 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %26]
    %37 = affine.min affine_map<()[s0] -> (16, s0)>()[%36]
    %38 = arith.subi %c-16_i32, %23 : i32
    %39 = arith.maxsi %38, %c0_i32 : i32
    %40 = arith.index_cast %39 : i32 to index
    %41 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%40, %36]
    %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%37, %41]
    %43 = arith.cmpi slt, %42, %c16 : index
    %44 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %30]
    %45 = affine.min affine_map<()[s0] -> (16, s0)>()[%44]
    %46 = arith.subi %c-32_i32, %23 : i32
    %47 = arith.maxsi %46, %c0_i32 : i32
    %48 = arith.index_cast %47 : i32 to index
    %49 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%48, %44]
    %50 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%45, %49]
    %51 = arith.cmpi slt, %50, %c16 : index
    %52 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %34]
    %53 = affine.min affine_map<()[s0] -> (16, s0)>()[%52]
    %54 = arith.subi %c-48_i32, %23 : i32
    %55 = arith.maxsi %54, %c0_i32 : i32
    %56 = arith.index_cast %55 : i32 to index
    %57 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%56, %52]
    %58 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%53, %57]
    %59 = arith.cmpi slt, %58, %c16 : index
    %60 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %61 = hivm.hir.pointer_cast(%c0_i64_8) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast = memref.cast %61 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%60 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %62 = memref_ext.alloc_workspace() from %arg2 offset = [%c1024] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %63 = hivm.hir.pointer_cast(%c1024_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_9 = memref.cast %63 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.nd2nz {dst_continuous} ins(%62 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %64 = memref_ext.alloc_workspace() from %arg2 offset = [%c2048] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %65 = hivm.hir.pointer_cast(%c2048_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_10 = memref.cast %65 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.nd2nz {dst_continuous} ins(%64 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_10 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %66 = memref_ext.alloc_workspace() from %arg2 offset = [%c3072] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %67 = hivm.hir.pointer_cast(%c3072_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_11 = memref.cast %67 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%66 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %68 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %69 = hivm.hir.pointer_cast(%c4096_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_12 = memref.cast %69 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%68 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_12 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %70 = memref_ext.alloc_workspace() from %arg2 offset = [%c5120] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %71 = hivm.hir.pointer_cast(%c5120_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_13 = memref.cast %71 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%70 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %72 = memref_ext.alloc_workspace() from %arg2 offset = [%c6144] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %73 = hivm.hir.pointer_cast(%c6144_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_14 = memref.cast %73 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%72 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_14 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %74 = memref_ext.alloc_workspace() from %arg2 offset = [%c7168] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %75 = hivm.hir.pointer_cast(%c7168_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_15 = memref.cast %75 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%74 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_15 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %76 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %77 = hivm.hir.pointer_cast(%c8192_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_16 = memref.cast %77 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%76 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %78 = memref_ext.alloc_workspace() from %arg2 offset = [%c9216] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %79 = hivm.hir.pointer_cast(%c9216_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_17 = memref.cast %79 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%78 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_17 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %80 = memref_ext.alloc_workspace() from %arg2 offset = [%c10240] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %81 = hivm.hir.pointer_cast(%c10240_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_18 = memref.cast %81 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%80 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %82 = memref_ext.alloc_workspace() from %arg2 offset = [%c11264] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %83 = hivm.hir.pointer_cast(%c11264_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_19 = memref.cast %83 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.nd2nz {dst_continuous} ins(%82 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_19 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %84 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%22, %30]
    %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%84], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %85 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>()[%22, %34]
    %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %86 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>()[%22, %34]
    %reinterpret_cast_24 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %87 = hivm.hir.pointer_cast(%c12288_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_25 = memref.cast %87 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%42, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %88 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%37, %41]
    %89 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%41]
    %90 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%41]
    %subview_26 = memref.subview %87[0, %89, %90, 0] [2, %88, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_27 = memref.cast %subview_26 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %43 {
      %collapse_shape = memref.collapse_shape %87 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_27 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %91 = hivm.hir.pointer_cast(%c13312_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_28 = memref.cast %91 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview_29 = memref.subview %reinterpret_cast_20[0, 0] [%50, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %92 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%45, %49]
    %93 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%49]
    %94 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%49]
    %subview_30 = memref.subview %91[0, %93, %94, 0] [2, %92, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_31 = memref.cast %subview_30 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %51 {
      %collapse_shape = memref.collapse_shape %91 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview_29 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_31 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %95 = hivm.hir.pointer_cast(%c14336_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_32 = memref.cast %95 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview_33 = memref.subview %reinterpret_cast_21[0, 0] [%50, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_34 = memref.subview %95[0, %93, %94, 0] [2, %92, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_35 = memref.cast %subview_34 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %51 {
      %collapse_shape = memref.collapse_shape %95 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview_33 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_35 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %96 = hivm.hir.pointer_cast(%c15360_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_36 = memref.cast %96 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview_37 = memref.subview %reinterpret_cast_22[0, 0] [%58, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %97 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%53, %57]
    %98 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%57]
    %99 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%57]
    %subview_38 = memref.subview %96[0, %98, %99, 0] [2, %97, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_39 = memref.cast %subview_38 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %59 {
      %collapse_shape = memref.collapse_shape %96 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview_37 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_39 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %100 = hivm.hir.pointer_cast(%c16384_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_40 = memref.cast %100 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview_41 = memref.subview %reinterpret_cast_23[0, 0] [%58, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_42 = memref.subview %100[0, %98, %99, 0] [2, %97, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_43 = memref.cast %subview_42 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %59 {
      %collapse_shape = memref.collapse_shape %100 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview_41 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_43 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %101 = hivm.hir.pointer_cast(%c17408_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_44 = memref.cast %101 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    %subview_45 = memref.subview %reinterpret_cast_24[0, 0] [%58, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_46 = memref.subview %101[0, %98, %99, 0] [2, %97, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
    %cast_47 = memref.cast %subview_46 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
    scf.if %59 {
      %collapse_shape = memref.collapse_shape %101 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<cbuf>>)
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
    } {hivm.unlikely_condition}
    hivm.hir.nd2nz {dst_continuous} ins(%subview_45 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_47 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
    %102 = hivm.hir.pointer_cast(%c0_i64_8) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_48 = memref.cast %102 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_11, %cast_25, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_48 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_6, %c1_i64_7, %c-1_i64, %c-1_i64, %11, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %103 = memref_ext.alloc_workspace() from %arg2 offset = [%c12288] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_48 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%103 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %104 = hivm.hir.pointer_cast(%c18432_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_49 = memref.cast %104 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%103 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_49 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    %105 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_50 = memref.cast %105 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_51 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_49, %cast, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_50 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_5, %c-1_i64_51, %c-1_i64_51, %c0_i64, %11, %c-1_i64_51, %c-1_i64_51 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %106 = memref_ext.alloc_workspace() from %arg2 offset = [%c13312] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_50 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%106 : memref<16x16xf32, #hivm.address_space<gm>>)
    annotation.mark %106 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %107 = memref_ext.alloc_workspace() from %arg2 offset = [%c14336] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %108 = hivm.hir.pointer_cast(%c19456_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_52 = memref.cast %108 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%107 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_52 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %109 = memref_ext.alloc_workspace() from %arg2 offset = [%c15360] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %110 = hivm.hir.pointer_cast(%c20480_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_53 = memref.cast %110 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%109 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_53 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %111 = hivm.hir.pointer_cast(%c2048_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_54 = memref.cast %111 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_14, %cast_32, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_54 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %112 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_54 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%112 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %113 = hivm.hir.pointer_cast(%c21504_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_55 = memref.cast %113 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%112 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_55 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
    %114 = hivm.hir.pointer_cast(%c3072_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_56 = memref.cast %114 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_57 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_55, %cast_12, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_56 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_4, %c-1_i64_57, %c-1_i64_57, %c-1_i64_57, %11, %c-1_i64_57, %c-1_i64_57 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %115 = memref_ext.alloc_workspace() from %arg2 offset = [%c17408] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_56 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%115 : memref<16x16xf32, #hivm.address_space<gm>>)
    annotation.mark %115 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %116 = memref_ext.alloc_workspace() from %arg2 offset = [%c18432] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %117 = hivm.hir.pointer_cast(%c22528_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_58 = memref.cast %117 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%116 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_58 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %118 = hivm.hir.pointer_cast(%c4096_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_59 = memref.cast %118 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_17, %cast_44, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_59 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %119 = memref_ext.alloc_workspace() from %arg2 offset = [%c19456] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_59 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%119 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    %120 = hivm.hir.pointer_cast(%c23552_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_60 = memref.cast %120 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%119 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_60 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
    %121 = hivm.hir.pointer_cast(%c5120_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_61 = memref.cast %121 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %c-1_i64_62 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_60, %cast_15, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_61 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_3, %c-1_i64_62, %c-1_i64_62, %c-1_i64_62, %11, %c-1_i64_62, %c-1_i64_62 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %122 = memref_ext.alloc_workspace() from %arg2 offset = [%c20480] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_61 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%122 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    annotation.mark %122 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %123 = hivm.hir.pointer_cast(%c6144_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_63 = memref.cast %123 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %c-1_i64_64 = arith.constant -1 : i64
    hivm.hir.mmadL1 ins(%cast_28, %cast_9, %true, %50, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_63 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_64, %c-1_i64_64, %c-1_i64_64, %c1_i64, %11, %c-1_i64_64, %c-1_i64_64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_32, %cast_52, %false, %50, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_63 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %124 = memref_ext.alloc_workspace() from %arg2 offset = [%c21504] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_63 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%124 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    %125 = hivm.hir.pointer_cast(%c24576_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_65 = memref.cast %125 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%124 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_65 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
    %126 = hivm.hir.pointer_cast(%c7168_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_66 = memref.cast %126 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
    %c-1_i64_67 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_16, %cast_65, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_66 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_67, %c0_i64_2, %c-1_i64_67, %c-1_i64_67, %11, %c-1_i64_67, %c-1_i64_67 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %127 = memref_ext.alloc_workspace() from %arg2 offset = [%c22528] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_66 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%127 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
    annotation.mark %127 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %128 = memref_ext.alloc_workspace() from %arg2 offset = [%c23552] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %129 = hivm.hir.pointer_cast(%c25600_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_68 = memref.cast %129 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%128 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_68 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    %130 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_69 = memref.cast %130 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
    hivm.hir.mmadL1 ins(%cast_40, %cast_13, %true, %58, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_69 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_44, %cast_58, %false, %58, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_69 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %131 = memref_ext.alloc_workspace() from %arg2 offset = [%c24576] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_69 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%131 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
    %132 = hivm.hir.pointer_cast(%c26624_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_70 = memref.cast %132 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%131 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_70 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
    %133 = hivm.hir.pointer_cast(%c9216_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_71 = memref.cast %133 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
    %c-1_i64_72 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_18, %cast_70, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_71 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_72, %c0_i64_1, %c-1_i64_72, %c-1_i64_72, %11, %c-1_i64_72, %c-1_i64_72 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %134 = memref_ext.alloc_workspace() from %arg2 offset = [%c25600] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_71 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%134 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
    annotation.mark %134 : memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %135 = hivm.hir.pointer_cast(%c10240_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_73 = memref.cast %135 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    %c-1_i64_74 = arith.constant -1 : i64
    hivm.hir.mmadL1 ins(%cast_36, %cast_10, %true, %58, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_73 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_74, %c-1_i64_74, %c-1_i64_74, %c2_i64, %11, %c-1_i64_74, %c-1_i64_74 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.mmadL1 ins(%cast_40, %cast_53, %false, %58, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_73 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_44, %cast_68, %false, %58, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_73 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %136 = memref_ext.alloc_workspace() from %arg2 offset = [%c26624] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_73 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%136 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    %137 = hivm.hir.pointer_cast(%c27648_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %cast_75 = memref.cast %137 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.nd2nz {dst_continuous} ins(%136 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%cast_75 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
    %138 = hivm.hir.pointer_cast(%c11264_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
    %cast_76 = memref.cast %138 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    %c-1_i64_77 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_19, %cast_75, %true, %c16, %c16, %c16 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_76 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_77, %c0_i64_0, %c3_i64, %c4_i64, %11, %c-1_i64_77, %c-1_i64_77 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %139 = memref_ext.alloc_workspace() from %arg2 offset = [%c27648] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_76 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%139 : memref<16x16xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    annotation.mark %139 : memref<16x16xf32, #hivm.address_space<gm>>
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
