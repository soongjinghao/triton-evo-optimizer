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
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant -4.480000e+02 : f32
  %cst_2 = arith.constant 4.480000e+02 : f32
  %cst_3 = arith.constant 0.00223214296 : f32
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst_4 = arith.constant 0.000000e+00 : f32
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
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xf32, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %7 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_5 : memref<4096xf32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg7 : i32 to index
  %9 = arith.index_cast %arg6 : i32 to index
  %10 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_6 = memref.subview %10[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_7, %offset_8, %sizes_9, %strides_10 = memref.extract_strided_metadata %subview_6 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast, %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_12 = memref.subview %11[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %subview_12 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_11 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_17 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_18 = memref.subview %12[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %collapse_shape_19 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%reinterpret_cast_24 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_25 = memref.collapse_shape %13 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<4096xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_26 = memref.collapse_shape %14 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_25 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<4096xi1, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_27 = memref.collapse_shape %15 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_5, %collapse_shape_5 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<4096xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_28 = memref.collapse_shape %16 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_27 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<4096xi1, #hivm.address_space<ub>>)
  %17 = arith.divsi %arg7, %c128_i32 : i32
  %18 = arith.index_cast %17 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %19 = arith.index_cast %arg11 : i32 to index
    %20 = arith.index_cast %3 : i32 to index
    %21 = arith.index_cast %1 : i32 to index
    %22 = arith.index_cast %c48_i32 : i32 to index
    %23 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%19, %20, %22]
    %24 = arith.index_cast %23 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %25 = arith.select %24, %c1_i64, %c2_i64 : i64
    %26 = arith.index_cast %23 : index to i1
    %c0_i64_29 = arith.constant 0 : i64
    %c1_i64_30 = arith.constant 1 : i64
    %27 = arith.select %26, %c0_i64_29, %c1_i64_30 : i64
    %28 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<32x128xf16, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c32768_i64, %c33792_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %31 = arith.divsi %arg11, %arg10 : i32
    %32 = arith.remsi %31, %arg9 : i32
    %33 = arith.divsi %arg11, %4 : i32
    %34 = arith.remsi %33, %arg8 : i32
    %35 = arith.muli %34, %c32_i32 : i32
    %36 = arith.muli %32, %c128_i32 : i32
    %37 = arith.index_cast %35 : i32 to index
    %38 = arith.index_cast %36 : i32 to index
    %39 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%38, %37, %8]
    %reinterpret_cast_31 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [32, 128], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %40 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%37, %9]
    %41 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%40, %37]
    %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%41, %37]
    %43 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%38, %8]
    %44 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%43, %38]
    %45 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%41, %37]
    %46 = affine.max affine_map<()[s0] -> (0, s0)>()[%45]
    %47 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%44, %38]
    %48 = affine.max affine_map<()[s0] -> (0, s0)>()[%47]
    %49 = arith.cmpi slt, %46, %c32 : index
    %50 = arith.cmpi slt, %48, %c128 : index
    %51 = arith.ori %49, %50 : i1
    %subview_32 = memref.subview %reinterpret_cast_31[0, 0] [%46, %48] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_33 = memref.subview %30[0, 0] [%46, %48] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %51 {
      %collapse_shape_104 = memref.collapse_shape %30 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_4 : f32) outs(%collapse_shape_104 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %25]
    hivm.hir.load ins(%subview_32 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_33 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %52 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %30 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_35 = memref.collapse_shape %52 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%collapse_shape_34 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<4096xf32, #hivm.address_space<ub>>)
    %53 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_36 = memref.subview %53[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c119808_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%52 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%subview_36 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%54 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %55 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_38 = memref.subview %55[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_39, %offset_40, %sizes_41, %strides_42 = memref.extract_strided_metadata %collapse_shape_37 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_43 = memref.reinterpret_cast %base_buffer_39 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_44, %offset_45, %sizes_46, %strides_47 = memref.extract_strided_metadata %subview_38 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_48 = memref.reinterpret_cast %base_buffer_44 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%reinterpret_cast_43, %reinterpret_cast_43 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_48 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_49 = memref.subview %56[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_50, %offset_51, %sizes_52, %strides_53 = memref.extract_strided_metadata %subview_49 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_54 = memref.reinterpret_cast %base_buffer_50 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%reinterpret_cast_48 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_54 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_55 = memref.subview %57[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_56, %offset_57, %sizes_58, %strides_59 = memref.extract_strided_metadata %subview_55 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_60 = memref.reinterpret_cast %base_buffer_56 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%reinterpret_cast_43, %cst : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_60 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c104448_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_61 = memref.subview %58[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_62, %offset_63, %sizes_64, %strides_65 = memref.extract_strided_metadata %subview_61 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_66 = memref.reinterpret_cast %base_buffer_62 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c105472_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_54, %cst, %reinterpret_cast_60 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_66 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
    %subview_67 = memref.subview %29[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_68, %offset_69, %sizes_70, %strides_71 = memref.extract_strided_metadata %subview_67 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_72 = memref.reinterpret_cast %base_buffer_68 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %60 = hivm.hir.pointer_cast(%c102400_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %27]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_17, %reinterpret_cast_43, %reinterpret_cast_66 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_72 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%60 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_73 = memref.subview %29[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_74, %offset_75, %sizes_76, %strides_77 = memref.extract_strided_metadata %subview_73 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_78 = memref.reinterpret_cast %base_buffer_74 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%reinterpret_cast_72, %cst_3 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_78 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_79 = memref.subview %61[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_80 = memref.collapse_shape %subview_79 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_81, %offset_82, %sizes_83, %strides_84 = memref.extract_strided_metadata %collapse_shape_80 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_85 = memref.reinterpret_cast %base_buffer_81 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%reinterpret_cast_24, %reinterpret_cast_78 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_85 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c103424_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%30, %subview_79 : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%30 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%62 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %63 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_86 = memref.collapse_shape %30 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_87 = memref.collapse_shape %63 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_86, %collapse_shape_86 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_87 : memref<4096xi1, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_88 = memref.collapse_shape %64 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_87 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_88 : memref<4096xi1, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c102912_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_89 = memref.collapse_shape %65 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_86, %cst_1 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_89 : memref<4096xf32, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c119296_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_90 = memref.collapse_shape %66 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c135680_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_88, %cst_1, %collapse_shape_89 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_90 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<16xf32, #hivm.address_space<ub>>)
    %68 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_91 = memref.collapse_shape %68 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %69 = hivm.hir.pointer_cast(%c118784_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_26, %collapse_shape_86, %collapse_shape_90 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_91 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %25]
    %70 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_92 = memref.collapse_shape %70 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_91, %collapse_shape_91 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_92 : memref<4096xi1, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_93 = memref.collapse_shape %71 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_92 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_93 : memref<4096xi1, #hivm.address_space<ub>>)
    %72 = hivm.hir.pointer_cast(%c119296_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_94 = memref.collapse_shape %72 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_91, %cst_2 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_94 : memref<4096xf32, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c135680_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_95 = memref.collapse_shape %73 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %74 = hivm.hir.pointer_cast(%c152064_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_93, %cst_2, %collapse_shape_94 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_95 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%74 : memref<16xf32, #hivm.address_space<ub>>)
    %75 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_96 = memref.collapse_shape %75 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %76 = hivm.hir.pointer_cast(%c118784_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_28, %collapse_shape_91, %collapse_shape_95 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_96 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%76 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_97 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [32, 128], strides: [%8, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_98 = memref.collapse_shape %28 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_96 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_98 : memref<4096xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_99 = memref.subview %28[0, 0] [%46, %48] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_100 = memref.subview %reinterpret_cast_97[0, 0] [%46, %48] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_99 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_100 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    %77 = arith.index_cast %32 : i32 to index
    %78 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%77, %37, %18]
    %reinterpret_cast_101 = memref.reinterpret_cast %arg5 to offset: [%78], sizes: [32], strides: [%18] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview_102 = memref.subview %subview_73[0] [%42] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_103 = memref.subview %reinterpret_cast_101[0] [%42] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_102 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_103 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %27]
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
