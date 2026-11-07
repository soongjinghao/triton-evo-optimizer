// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c180480_i64 = arith.constant 180480 : i64
  %c164096_i64 = arith.constant 164096 : i64
  %c164032_i64 = arith.constant 164032 : i64
  %c147648_i64 = arith.constant 147648 : i64
  %c131264_i64 = arith.constant 131264 : i64
  %c130752_i64 = arith.constant 130752 : i64
  %c130720_i64 = arith.constant 130720 : i64
  %c114336_i64 = arith.constant 114336 : i64
  %c114272_i64 = arith.constant 114272 : i64
  %c97888_i64 = arith.constant 97888 : i64
  %c81504_i64 = arith.constant 81504 : i64
  %c80992_i64 = arith.constant 80992 : i64
  %c79968_i64 = arith.constant 79968 : i64
  %c79936_i64 = arith.constant 79936 : i64
  %c78912_i64 = arith.constant 78912 : i64
  %c78848_i64 = arith.constant 78848 : i64
  %c77824_i64 = arith.constant 77824 : i64
  %c76800_i64 = arith.constant 76800 : i64
  %c75776_i64 = arith.constant 75776 : i64
  %c67584_i64 = arith.constant 67584 : i64
  %c66560_i64 = arith.constant 66560 : i64
  %c50176_i64 = arith.constant 50176 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c17408_i64 = arith.constant 17408 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 0.00223214296 : f32
  %cst_1 = arith.constant 4.480000e+02 : f32
  %cst_2 = arith.constant -4.480000e+02 : f32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_3 = arith.constant 9.99999974E-5 : f32
  %c0 = arith.constant 0 : index
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg10 : i32
  %5 = arith.remsi %4, %arg9 : i32
  %6 = arith.muli %arg10, %arg9 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg8 : i32
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview = memref.subview %9[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %10 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c17408_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %11 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_4 : memref<4096xf32, #hivm.address_space<ub>>)
  %12 = arith.muli %8, %c32_i32 : i32
  %13 = arith.muli %5, %c128_i32 : i32
  %14 = arith.index_cast %12 : i32 to index
  %15 = arith.index_cast %arg7 : i32 to index
  %16 = arith.index_cast %13 : i32 to index
  %17 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%16, %14, %15]
  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [32, 128], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %18 = hivm.hir.pointer_cast(%c33792_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %19 = arith.index_cast %arg6 : i32 to index
  %20 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%14, %19]
  %21 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%20, %14]
  %22 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%21, %14]
  %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%16, %15]
  %24 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%23, %16]
  %25 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%21, %14]
  %26 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%24, %16]
  %27 = arith.cmpi slt, %25, %c32 : index
  %28 = arith.cmpi slt, %26, %c128 : index
  %29 = arith.ori %27, %28 : i1
  %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%25, %26] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_7 = memref.subview %18[0, 0] [%25, %26] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
  scf.if %29 {
    %collapse_shape_87 = memref.collapse_shape %18 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_87 : memref<4096xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %30 = hivm.hir.pointer_cast(%c50176_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_8 = memref.collapse_shape %18 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %collapse_shape_9 = memref.collapse_shape %30 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vabs ins(%collapse_shape_8 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<4096xf32, #hivm.address_space<ub>>)
  %31 = hivm.hir.pointer_cast(%c66560_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_10 = memref.subview %31[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %32 = hivm.hir.pointer_cast(%c67584_i64) : memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vreduce <max> ins(%30 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%subview_10 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%32 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
  %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %33 = hivm.hir.pointer_cast(%c75776_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_12 = memref.subview %33[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %collapse_shape_11 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %subview_12 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast_17, %reinterpret_cast_17 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_22 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %34 = hivm.hir.pointer_cast(%c75776_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_23 = memref.subview %34[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_24, %offset_25, %sizes_26, %strides_27 = memref.extract_strided_metadata %subview_23 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_22 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_28 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %35 = hivm.hir.pointer_cast(%c0_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_29 = memref.subview %35[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_30, %offset_31, %sizes_32, %strides_33 = memref.extract_strided_metadata %subview_29 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_34 = memref.reinterpret_cast %base_buffer_30 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%reinterpret_cast, %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_34 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %36 = hivm.hir.pointer_cast(%c0_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_35 = memref.subview %36[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_36, %offset_37, %sizes_38, %strides_39 = memref.extract_strided_metadata %subview_35 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_40 = memref.reinterpret_cast %base_buffer_36 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_34 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_40 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %37 = hivm.hir.pointer_cast(%c76800_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_41 = memref.subview %37[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_42, %offset_43, %sizes_44, %strides_45 = memref.extract_strided_metadata %subview_41 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_46 = memref.reinterpret_cast %base_buffer_42 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%reinterpret_cast_17, %cst_3 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_46 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %38 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_47 = memref.subview %38[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_48, %offset_49, %sizes_50, %strides_51 = memref.extract_strided_metadata %subview_47 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_52 = memref.reinterpret_cast %base_buffer_48 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %39 = hivm.hir.pointer_cast(%c78848_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%reinterpret_cast_28, %cst_3, %reinterpret_cast_46 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_52 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%39 : memref<16xf32, #hivm.address_space<ub>>)
  %40 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_53 = memref.subview %40[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_54, %offset_55, %sizes_56, %strides_57 = memref.extract_strided_metadata %subview_53 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_58 = memref.reinterpret_cast %base_buffer_54 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %41 = hivm.hir.pointer_cast(%c79936_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%reinterpret_cast_40, %reinterpret_cast_17, %reinterpret_cast_52 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_58 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%41 : memref<8xf32, #hivm.address_space<ub>>)
  %42 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_59 = memref.subview %42[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_60, %offset_61, %sizes_62, %strides_63 = memref.extract_strided_metadata %subview_59 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_64 = memref.reinterpret_cast %base_buffer_60 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%reinterpret_cast_58, %cst_0 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_64 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %subview_59 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
  %43 = hivm.hir.pointer_cast(%c33792_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %44 = hivm.hir.pointer_cast(%c79968_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vdiv ins(%18, %expand_shape : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%43 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%44 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
  %45 = hivm.hir.pointer_cast(%c80992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_65 = memref.collapse_shape %43 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %collapse_shape_66 = memref.collapse_shape %45 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_65, %collapse_shape_65 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_66 : memref<4096xi1, #hivm.address_space<ub>>)
  %46 = hivm.hir.pointer_cast(%c80992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_67 = memref.collapse_shape %46 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_66 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_67 : memref<4096xi1, #hivm.address_space<ub>>)
  %47 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_68 = memref.collapse_shape %47 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_68 : memref<4096xi1, #hivm.address_space<ub>>)
  %48 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_69 = memref.collapse_shape %48 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_68 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_69 : memref<4096xi1, #hivm.address_space<ub>>)
  %49 = hivm.hir.pointer_cast(%c81504_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_70 = memref.collapse_shape %49 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%collapse_shape_65, %cst_2 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_70 : memref<4096xf32, #hivm.address_space<ub>>)
  %50 = hivm.hir.pointer_cast(%c97888_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_71 = memref.collapse_shape %50 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %51 = hivm.hir.pointer_cast(%c114272_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_67, %cst_2, %collapse_shape_70 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_71 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<16xf32, #hivm.address_space<ub>>)
  %52 = hivm.hir.pointer_cast(%c114336_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_72 = memref.collapse_shape %52 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %53 = hivm.hir.pointer_cast(%c130720_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_69, %collapse_shape_65, %collapse_shape_71 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_72 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<8xf32, #hivm.address_space<ub>>)
  %54 = hivm.hir.pointer_cast(%c130752_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_73 = memref.collapse_shape %54 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_72, %collapse_shape_72 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_73 : memref<4096xi1, #hivm.address_space<ub>>)
  %55 = hivm.hir.pointer_cast(%c130752_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_74 = memref.collapse_shape %55 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_73 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_74 : memref<4096xi1, #hivm.address_space<ub>>)
  %56 = hivm.hir.pointer_cast(%c17408_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_75 = memref.collapse_shape %56 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_4, %collapse_shape_4 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_75 : memref<4096xi1, #hivm.address_space<ub>>)
  %57 = hivm.hir.pointer_cast(%c17408_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_76 = memref.collapse_shape %57 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_75 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_76 : memref<4096xi1, #hivm.address_space<ub>>)
  %58 = hivm.hir.pointer_cast(%c131264_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_77 = memref.collapse_shape %58 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vmin ins(%collapse_shape_72, %cst_1 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_77 : memref<4096xf32, #hivm.address_space<ub>>)
  %59 = hivm.hir.pointer_cast(%c147648_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_78 = memref.collapse_shape %59 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %60 = hivm.hir.pointer_cast(%c164032_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_74, %cst_1, %collapse_shape_77 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_78 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<16xf32, #hivm.address_space<ub>>)
  %61 = hivm.hir.pointer_cast(%c164096_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_79 = memref.collapse_shape %61 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %62 = hivm.hir.pointer_cast(%c180480_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_76, %collapse_shape_72, %collapse_shape_78 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_79 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%62 : memref<8xf32, #hivm.address_space<ub>>)
  %reinterpret_cast_80 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [32, 128], strides: [%15, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %63 = hivm.hir.pointer_cast(%c164096_i64) : memref<32x128xf16, #hivm.address_space<ub>>
  %collapse_shape_81 = memref.collapse_shape %63 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_79 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_81 : memref<4096xf16, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %subview_82 = memref.subview %63[0, 0] [%25, %26] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
  %subview_83 = memref.subview %reinterpret_cast_80[0, 0] [%25, %26] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_82 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_83 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  %64 = arith.divsi %arg7, %c128_i32 : i32
  %65 = arith.index_cast %64 : i32 to index
  %66 = arith.index_cast %5 : i32 to index
  %67 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%66, %14, %65]
  %reinterpret_cast_84 = memref.reinterpret_cast %arg5 to offset: [%67], sizes: [32], strides: [%65] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  %subview_85 = memref.subview %subview_59[0] [%22] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
  %subview_86 = memref.subview %reinterpret_cast_84[0] [%22] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_85 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_86 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
