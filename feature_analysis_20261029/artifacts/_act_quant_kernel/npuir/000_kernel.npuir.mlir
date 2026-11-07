// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c181504_i64 = arith.constant 181504 : i64
  %c165120_i64 = arith.constant 165120 : i64
  %c165056_i64 = arith.constant 165056 : i64
  %c148672_i64 = arith.constant 148672 : i64
  %c132288_i64 = arith.constant 132288 : i64
  %c131776_i64 = arith.constant 131776 : i64
  %c131744_i64 = arith.constant 131744 : i64
  %c115360_i64 = arith.constant 115360 : i64
  %c115296_i64 = arith.constant 115296 : i64
  %c98912_i64 = arith.constant 98912 : i64
  %c82528_i64 = arith.constant 82528 : i64
  %c82016_i64 = arith.constant 82016 : i64
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
  %cst = arith.constant 0.693147182 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst_2 = arith.constant 0.00223214296 : f32
  %cst_3 = arith.constant 4.480000e+02 : f32
  %cst_4 = arith.constant -4.480000e+02 : f32
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_5 = arith.constant 9.99999974E-5 : f32
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
  hivm.hir.vbrc ins(%cst_5 : f32) outs(%reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %10 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_4 : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c17408_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %11 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>>)
  %12 = arith.muli %8, %c32_i32 : i32
  %13 = arith.muli %5, %c128_i32 : i32
  %14 = arith.index_cast %12 : i32 to index
  %15 = arith.index_cast %arg7 : i32 to index
  %16 = arith.index_cast %13 : i32 to index
  %17 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%16, %14, %15]
  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [32, 128], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
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
  %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%25, %26] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_9 = memref.subview %18[0, 0] [%25, %26] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
  scf.if %29 {
    %collapse_shape_131 = memref.collapse_shape %18 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_131 : memref<4096xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %30 = hivm.hir.pointer_cast(%c50176_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_10 = memref.collapse_shape %18 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %collapse_shape_11 = memref.collapse_shape %30 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vabs ins(%collapse_shape_10 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<4096xf32, #hivm.address_space<ub>>)
  %31 = hivm.hir.pointer_cast(%c66560_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_12 = memref.subview %31[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %32 = hivm.hir.pointer_cast(%c67584_i64) : memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vreduce <max> ins(%30 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%subview_12 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%32 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
  %collapse_shape_13 = memref.collapse_shape %subview_12 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %33 = hivm.hir.pointer_cast(%c75776_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_14 = memref.subview %33[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %collapse_shape_13 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %subview_14 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast_19, %reinterpret_cast_19 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_24 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %34 = hivm.hir.pointer_cast(%c75776_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_25 = memref.subview %34[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %subview_25 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_30 = memref.reinterpret_cast %base_buffer_26 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_24 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_30 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %35 = hivm.hir.pointer_cast(%c0_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_31 = memref.subview %35[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_32, %offset_33, %sizes_34, %strides_35 = memref.extract_strided_metadata %subview_31 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_36 = memref.reinterpret_cast %base_buffer_32 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%reinterpret_cast, %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_36 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %36 = hivm.hir.pointer_cast(%c0_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_37 = memref.subview %36[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_38, %offset_39, %sizes_40, %strides_41 = memref.extract_strided_metadata %subview_37 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_42 = memref.reinterpret_cast %base_buffer_38 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_36 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_42 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %37 = hivm.hir.pointer_cast(%c76800_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_43 = memref.subview %37[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_44, %offset_45, %sizes_46, %strides_47 = memref.extract_strided_metadata %subview_43 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_48 = memref.reinterpret_cast %base_buffer_44 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%reinterpret_cast_19, %cst_5 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_48 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %38 = hivm.hir.pointer_cast(%c77824_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_49 = memref.subview %38[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_50, %offset_51, %sizes_52, %strides_53 = memref.extract_strided_metadata %subview_49 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_54 = memref.reinterpret_cast %base_buffer_50 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %39 = hivm.hir.pointer_cast(%c78848_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%reinterpret_cast_30, %cst_5, %reinterpret_cast_48 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_54 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%39 : memref<16xf32, #hivm.address_space<ub>>)
  %40 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_55 = memref.subview %40[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_56, %offset_57, %sizes_58, %strides_59 = memref.extract_strided_metadata %subview_55 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_60 = memref.reinterpret_cast %base_buffer_56 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %41 = hivm.hir.pointer_cast(%c79936_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%reinterpret_cast_42, %reinterpret_cast_19, %reinterpret_cast_54 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_60 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%41 : memref<8xf32, #hivm.address_space<ub>>)
  %42 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_61 = memref.subview %42[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_62, %offset_63, %sizes_64, %strides_65 = memref.extract_strided_metadata %subview_61 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_66 = memref.reinterpret_cast %base_buffer_62 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%reinterpret_cast_60, %cst_2 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_66 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %43 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_67 = memref.subview %43[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_68, %offset_69, %sizes_70, %strides_71 = memref.extract_strided_metadata %subview_67 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%reinterpret_cast_66 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_72 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %44 = hivm.hir.pointer_cast(%c79968_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_73 = memref.subview %44[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_74, %offset_75, %sizes_76, %strides_77 = memref.extract_strided_metadata %subview_73 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_78 = memref.reinterpret_cast %base_buffer_74 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%reinterpret_cast_78 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %45 = hivm.hir.pointer_cast(%c79968_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_79 = memref.subview %45[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_80, %offset_81, %sizes_82, %strides_83 = memref.extract_strided_metadata %subview_79 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_84 = memref.reinterpret_cast %base_buffer_80 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%reinterpret_cast_78 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_84 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %46 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_85 = memref.subview %46[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_86, %offset_87, %sizes_88, %strides_89 = memref.extract_strided_metadata %subview_85 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_90 = memref.reinterpret_cast %base_buffer_86 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vdiv ins(%reinterpret_cast_72, %reinterpret_cast_84 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_90 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %47 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_91 = memref.subview %47[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_92, %offset_93, %sizes_94, %strides_95 = memref.extract_strided_metadata %subview_91 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_96 = memref.reinterpret_cast %base_buffer_92 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%reinterpret_cast_90 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_96 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) round_mode = <ceil>
  %48 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_97 = memref.subview %48[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_98, %offset_99, %sizes_100, %strides_101 = memref.extract_strided_metadata %subview_97 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_102 = memref.reinterpret_cast %base_buffer_98 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%reinterpret_cast_96, %cst : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_102 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %49 = hivm.hir.pointer_cast(%c78912_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_103 = memref.subview %49[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_104, %offset_105, %sizes_106, %strides_107 = memref.extract_strided_metadata %subview_103 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_108 = memref.reinterpret_cast %base_buffer_104 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vexp ins(%reinterpret_cast_102 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_108 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %subview_103 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
  %50 = hivm.hir.pointer_cast(%c33792_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %51 = hivm.hir.pointer_cast(%c80992_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vdiv ins(%18, %expand_shape : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%50 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
  %52 = hivm.hir.pointer_cast(%c82016_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_109 = memref.collapse_shape %50 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %collapse_shape_110 = memref.collapse_shape %52 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_109, %collapse_shape_109 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_110 : memref<4096xi1, #hivm.address_space<ub>>)
  %53 = hivm.hir.pointer_cast(%c82016_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_111 = memref.collapse_shape %53 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_110 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_111 : memref<4096xi1, #hivm.address_space<ub>>)
  %54 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_112 = memref.collapse_shape %54 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_112 : memref<4096xi1, #hivm.address_space<ub>>)
  %55 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_113 = memref.collapse_shape %55 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_112 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_113 : memref<4096xi1, #hivm.address_space<ub>>)
  %56 = hivm.hir.pointer_cast(%c82528_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_114 = memref.collapse_shape %56 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%collapse_shape_109, %cst_4 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_114 : memref<4096xf32, #hivm.address_space<ub>>)
  %57 = hivm.hir.pointer_cast(%c98912_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_115 = memref.collapse_shape %57 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %58 = hivm.hir.pointer_cast(%c115296_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_111, %cst_4, %collapse_shape_114 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_115 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%58 : memref<16xf32, #hivm.address_space<ub>>)
  %59 = hivm.hir.pointer_cast(%c115360_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_116 = memref.collapse_shape %59 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %60 = hivm.hir.pointer_cast(%c131744_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_113, %collapse_shape_109, %collapse_shape_115 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_116 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<8xf32, #hivm.address_space<ub>>)
  %61 = hivm.hir.pointer_cast(%c131776_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_117 = memref.collapse_shape %61 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_116, %collapse_shape_116 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_117 : memref<4096xi1, #hivm.address_space<ub>>)
  %62 = hivm.hir.pointer_cast(%c131776_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_118 = memref.collapse_shape %62 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_117 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_118 : memref<4096xi1, #hivm.address_space<ub>>)
  %63 = hivm.hir.pointer_cast(%c17408_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_119 = memref.collapse_shape %63 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_119 : memref<4096xi1, #hivm.address_space<ub>>)
  %64 = hivm.hir.pointer_cast(%c17408_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_120 = memref.collapse_shape %64 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_119 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_120 : memref<4096xi1, #hivm.address_space<ub>>)
  %65 = hivm.hir.pointer_cast(%c132288_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_121 = memref.collapse_shape %65 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vmin ins(%collapse_shape_116, %cst_3 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_121 : memref<4096xf32, #hivm.address_space<ub>>)
  %66 = hivm.hir.pointer_cast(%c148672_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_122 = memref.collapse_shape %66 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %67 = hivm.hir.pointer_cast(%c165056_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_118, %cst_3, %collapse_shape_121 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_122 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
  %68 = hivm.hir.pointer_cast(%c165120_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_123 = memref.collapse_shape %68 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %69 = hivm.hir.pointer_cast(%c181504_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_120, %collapse_shape_116, %collapse_shape_122 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_123 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<8xf32, #hivm.address_space<ub>>)
  %reinterpret_cast_124 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [32, 128], strides: [%15, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %70 = hivm.hir.pointer_cast(%c165120_i64) : memref<32x128xf16, #hivm.address_space<ub>>
  %collapse_shape_125 = memref.collapse_shape %70 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_123 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_125 : memref<4096xf16, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %subview_126 = memref.subview %70[0, 0] [%25, %26] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
  %subview_127 = memref.subview %reinterpret_cast_124[0, 0] [%25, %26] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_126 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_127 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  %71 = arith.divsi %arg7, %c128_i32 : i32
  %72 = arith.index_cast %71 : i32 to index
  %73 = arith.index_cast %5 : i32 to index
  %74 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%73, %14, %72]
  %reinterpret_cast_128 = memref.reinterpret_cast %arg5 to offset: [%74], sizes: [32], strides: [%72] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  %subview_129 = memref.subview %subview_103[0] [%22] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
  %subview_130 = memref.subview %reinterpret_cast_128[0] [%22] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_129 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_130 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
