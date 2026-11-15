// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c51200_i64 = arith.constant 51200 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c152064_i64 = arith.constant 152064 : i64
  %c135680_i64 = arith.constant 135680 : i64
  %c119296_i64 = arith.constant 119296 : i64
  %c102912_i64 = arith.constant 102912 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c105472_i64 = arith.constant 105472 : i64
  %c104448_i64 = arith.constant 104448 : i64
  %c103424_i64 = arith.constant 103424 : i64
  %c119808_i64 = arith.constant 119808 : i64
  %c118784_i64 = arith.constant 118784 : i64
  %c102400_i64 = arith.constant 102400 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c101376_i64 = arith.constant 101376 : i64
  %c84992_i64 = arith.constant 84992 : i64
  %c68608_i64 = arith.constant 68608 : i64
  %c67584_i64 = arith.constant 67584 : i64
  %cst = arith.constant 9.99999974E-5 : f32
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst_0 = arith.constant -4.480000e+02 : f32
  %cst_1 = arith.constant 4.480000e+02 : f32
  %cst_2 = arith.constant 0.00223214296 : f32
  %c127_i32 = arith.constant 127 : i32
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst_3 = arith.constant 0.000000e+00 : f32
  %cst_4 = arith.constant 2.000000e+00 : f32
  %cst_5 = arith.constant 0.693147182 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview = memref.subview %5[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %6 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %7 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>>)
  %8 = arith.addi %arg7, %c127_i32 : i32
  %9 = arith.divsi %8, %c128_i32 : i32
  %10 = arith.index_cast %arg7 : i32 to index
  %11 = arith.index_cast %arg6 : i32 to index
  %12 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_7 = memref.subview %12[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_8, %offset_9, %sizes_10, %strides_11 = memref.extract_strided_metadata %subview_7 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast, %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_12 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_13 = memref.subview %13[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_14, %offset_15, %sizes_16, %strides_17 = memref.extract_strided_metadata %subview_13 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer_14 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_12 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_18 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_19 = memref.subview %14[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %subview_19 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_4 : f32) outs(%reinterpret_cast_24 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_25 = memref.subview %15[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %subview_25 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_30 = memref.reinterpret_cast %base_buffer_26 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%reinterpret_cast_24 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_30 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_31 = memref.collapse_shape %16 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<4096xi1, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_32 = memref.collapse_shape %17 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_31 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<4096xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_33 = memref.collapse_shape %18 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_6, %collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<4096xi1, #hivm.address_space<ub>>)
  %19 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_34 = memref.collapse_shape %19 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_33 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<4096xi1, #hivm.address_space<ub>>)
  %20 = arith.divsi %arg7, %c128_i32 : i32
  %21 = arith.index_cast %20 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %22 = arith.index_cast %arg11 : i32 to index
    %23 = arith.index_cast %3 : i32 to index
    %24 = arith.index_cast %1 : i32 to index
    %25 = arith.index_cast %c48_i32 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%22, %23, %25]
    %27 = arith.index_cast %26 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %28 = arith.select %27, %c1_i64, %c2_i64 : i64
    %29 = arith.index_cast %26 : index to i1
    %c0_i64_35 = arith.constant 0 : i64
    %c1_i64_36 = arith.constant 1 : i64
    %30 = arith.select %29, %c0_i64_35, %c1_i64_36 : i64
    %31 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<32x128xf16, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c32768_i64, %c33792_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %34 = arith.divsi %arg11, %4 : i32
    %35 = arith.remsi %34, %arg8 : i32
    %36 = arith.divsi %35, %9 : i32
    %37 = arith.remsi %35, %9 : i32
    %38 = arith.muli %36, %c32_i32 : i32
    %39 = arith.muli %37, %c128_i32 : i32
    %40 = arith.index_cast %38 : i32 to index
    %41 = arith.index_cast %39 : i32 to index
    %42 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%41, %40, %10]
    %reinterpret_cast_37 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 128], strides: [%10, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %43 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%40, %11]
    %44 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%43, %40]
    %45 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%44, %40]
    %46 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%41, %10]
    %47 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%46, %41]
    %48 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%44, %40]
    %49 = affine.max affine_map<()[s0] -> (0, s0)>()[%48]
    %50 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%47, %41]
    %51 = affine.max affine_map<()[s0] -> (0, s0)>()[%50]
    %52 = arith.cmpi slt, %49, %c32 : index
    %53 = arith.cmpi slt, %51, %c128 : index
    %54 = arith.ori %52, %53 : i1
    %subview_38 = memref.subview %reinterpret_cast_37[0, 0] [%49, %51] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_39 = memref.subview %33[0, 0] [%49, %51] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %54 {
      %collapse_shape_133 = memref.collapse_shape %33 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%collapse_shape_133 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %28]
    hivm.hir.load ins(%subview_38 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_39 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_3 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %55 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_40 = memref.collapse_shape %33 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_41 = memref.collapse_shape %55 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%collapse_shape_40 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_41 : memref<4096xf32, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_42 = memref.subview %56[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%c119808_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%55 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%subview_42 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%57 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_43 = memref.collapse_shape %subview_42 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %58 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_44 = memref.subview %58[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_45, %offset_46, %sizes_47, %strides_48 = memref.extract_strided_metadata %collapse_shape_43 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_49 = memref.reinterpret_cast %base_buffer_45 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_50, %offset_51, %sizes_52, %strides_53 = memref.extract_strided_metadata %subview_44 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_54 = memref.reinterpret_cast %base_buffer_50 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%reinterpret_cast_49, %reinterpret_cast_49 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_54 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %59 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_55 = memref.subview %59[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_56, %offset_57, %sizes_58, %strides_59 = memref.extract_strided_metadata %subview_55 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_60 = memref.reinterpret_cast %base_buffer_56 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%reinterpret_cast_54 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_60 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_61 = memref.subview %60[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_62, %offset_63, %sizes_64, %strides_65 = memref.extract_strided_metadata %subview_61 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_66 = memref.reinterpret_cast %base_buffer_62 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%reinterpret_cast_49, %cst : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_66 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c104448_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_67 = memref.subview %61[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_68, %offset_69, %sizes_70, %strides_71 = memref.extract_strided_metadata %subview_67 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c105472_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_60, %cst, %reinterpret_cast_66 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_72 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%62 : memref<16xf32, #hivm.address_space<ub>>)
    %subview_73 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_74, %offset_75, %sizes_76, %strides_77 = memref.extract_strided_metadata %subview_73 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_78 = memref.reinterpret_cast %base_buffer_74 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c102400_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %30]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_18, %reinterpret_cast_49, %reinterpret_cast_72 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_78 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%63 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_79 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_80, %offset_81, %sizes_82, %strides_83 = memref.extract_strided_metadata %subview_79 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_84 = memref.reinterpret_cast %base_buffer_80 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%reinterpret_cast_78, %cst_2 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_84 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_85 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_86, %offset_87, %sizes_88, %strides_89 = memref.extract_strided_metadata %subview_85 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_90 = memref.reinterpret_cast %base_buffer_86 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%reinterpret_cast_84 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_90 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_91 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_92, %offset_93, %sizes_94, %strides_95 = memref.extract_strided_metadata %subview_91 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_96 = memref.reinterpret_cast %base_buffer_92 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%reinterpret_cast_90, %reinterpret_cast_30 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_96 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_97 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_98, %offset_99, %sizes_100, %strides_101 = memref.extract_strided_metadata %subview_97 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_102 = memref.reinterpret_cast %base_buffer_98 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%reinterpret_cast_96 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_102 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) round_mode = <ceil>
    %subview_103 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_104, %offset_105, %sizes_106, %strides_107 = memref.extract_strided_metadata %subview_103 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_108 = memref.reinterpret_cast %base_buffer_104 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%reinterpret_cast_102, %cst_5 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_108 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_109 = memref.subview %32[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_110, %offset_111, %sizes_112, %strides_113 = memref.extract_strided_metadata %subview_109 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_114 = memref.reinterpret_cast %base_buffer_110 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%reinterpret_cast_108 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_114 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %subview_109 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
    %64 = hivm.hir.pointer_cast(%c102400_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%33, %expand_shape : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%33 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%64 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %65 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_115 = memref.collapse_shape %33 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_116 = memref.collapse_shape %65 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_115, %collapse_shape_115 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_116 : memref<4096xi1, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_117 = memref.collapse_shape %66 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_116 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_117 : memref<4096xi1, #hivm.address_space<ub>>)
    %67 = hivm.hir.pointer_cast(%c102912_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_118 = memref.collapse_shape %67 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_115, %cst_0 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_118 : memref<4096xf32, #hivm.address_space<ub>>)
    %68 = hivm.hir.pointer_cast(%c119296_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_119 = memref.collapse_shape %68 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %69 = hivm.hir.pointer_cast(%c135680_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_117, %cst_0, %collapse_shape_118 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_119 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<16xf32, #hivm.address_space<ub>>)
    %70 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_120 = memref.collapse_shape %70 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %71 = hivm.hir.pointer_cast(%c118784_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_32, %collapse_shape_115, %collapse_shape_119 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_120 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %28]
    %72 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_121 = memref.collapse_shape %72 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_120, %collapse_shape_120 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_121 : memref<4096xi1, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_122 = memref.collapse_shape %73 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_121 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_122 : memref<4096xi1, #hivm.address_space<ub>>)
    %74 = hivm.hir.pointer_cast(%c119296_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_123 = memref.collapse_shape %74 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_120, %cst_1 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_123 : memref<4096xf32, #hivm.address_space<ub>>)
    %75 = hivm.hir.pointer_cast(%c135680_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_124 = memref.collapse_shape %75 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %76 = hivm.hir.pointer_cast(%c152064_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_122, %cst_1, %collapse_shape_123 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_124 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%76 : memref<16xf32, #hivm.address_space<ub>>)
    %77 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_125 = memref.collapse_shape %77 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %78 = hivm.hir.pointer_cast(%c118784_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_34, %collapse_shape_120, %collapse_shape_124 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_125 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%78 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_126 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [32, 128], strides: [%10, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_127 = memref.collapse_shape %31 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_125 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_127 : memref<4096xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_128 = memref.subview %31[0, 0] [%49, %51] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_129 = memref.subview %reinterpret_cast_126[0, 0] [%49, %51] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_128 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_129 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    %79 = arith.index_cast %37 : i32 to index
    %80 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%79, %40, %21]
    %reinterpret_cast_130 = memref.reinterpret_cast %arg5 to offset: [%80], sizes: [32], strides: [%21] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview_131 = memref.subview %subview_109[0] [%45] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_132 = memref.subview %reinterpret_cast_130[0] [%45] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_131 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_132 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %30]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
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
