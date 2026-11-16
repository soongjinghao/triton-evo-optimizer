// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c51200_i64 = arith.constant 51200 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c153088_i64 = arith.constant 153088 : i64
  %c136704_i64 = arith.constant 136704 : i64
  %c120320_i64 = arith.constant 120320 : i64
  %c103936_i64 = arith.constant 103936 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c106496_i64 = arith.constant 106496 : i64
  %c105472_i64 = arith.constant 105472 : i64
  %c104448_i64 = arith.constant 104448 : i64
  %c120832_i64 = arith.constant 120832 : i64
  %c119808_i64 = arith.constant 119808 : i64
  %c103424_i64 = arith.constant 103424 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c102400_i64 = arith.constant 102400 : i64
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
  %cst_5 = arith.constant 2.000000e+00 : f32
  %cst_6 = arith.constant 0.693147182 : f32
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
  %collapse_shape_7 = memref.collapse_shape %7 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_7 : memref<4096xf32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg7 : i32 to index
  %9 = arith.index_cast %arg6 : i32 to index
  %10 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_8 = memref.subview %10[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %subview_8 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer_9 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast, %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_13 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_14 = memref.subview %11[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %subview_14 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_13 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_19 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_20 = memref.subview %12[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_21, %offset_22, %sizes_23, %strides_24 = memref.extract_strided_metadata %subview_20 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_25 = memref.reinterpret_cast %base_buffer_21 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_5 : f32) outs(%reinterpret_cast_25 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_26 = memref.subview %13[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_27, %offset_28, %sizes_29, %strides_30 = memref.extract_strided_metadata %subview_26 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_31 = memref.reinterpret_cast %base_buffer_27 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%reinterpret_cast_25 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_31 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_32 = memref.subview %14[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_34, %offset_35, %sizes_36, %strides_37 = memref.extract_strided_metadata %collapse_shape_33 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_38 = memref.reinterpret_cast %base_buffer_34 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%reinterpret_cast_38 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_39 = memref.collapse_shape %15 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_39 : memref<4096xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_40 = memref.collapse_shape %16 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_39 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<4096xi1, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_41 = memref.collapse_shape %17 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_7, %collapse_shape_7 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_41 : memref<4096xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_42 = memref.collapse_shape %18 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_41 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_42 : memref<4096xi1, #hivm.address_space<ub>>)
  %19 = arith.divsi %arg7, %c128_i32 : i32
  %20 = arith.index_cast %19 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %21 = arith.index_cast %arg11 : i32 to index
    %22 = arith.index_cast %3 : i32 to index
    %23 = arith.index_cast %1 : i32 to index
    %24 = arith.index_cast %c48_i32 : i32 to index
    %25 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%21, %22, %24]
    %26 = arith.index_cast %25 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %27 = arith.select %26, %c1_i64, %c2_i64 : i64
    %28 = arith.index_cast %25 : index to i1
    %c0_i64_43 = arith.constant 0 : i64
    %c1_i64_44 = arith.constant 1 : i64
    %29 = arith.select %28, %c0_i64_43, %c1_i64_44 : i64
    %30 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<32x128xf16, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c32768_i64, %c33792_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %33 = arith.divsi %arg11, %arg10 : i32
    %34 = arith.remsi %33, %arg9 : i32
    %35 = arith.divsi %arg11, %4 : i32
    %36 = arith.remsi %35, %arg8 : i32
    %37 = arith.muli %36, %c32_i32 : i32
    %38 = arith.muli %34, %c128_i32 : i32
    %39 = arith.index_cast %37 : i32 to index
    %40 = arith.index_cast %38 : i32 to index
    %41 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%40, %39, %8]
    %reinterpret_cast_45 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [32, 128], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %42 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%39, %9]
    %43 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%42, %39]
    %44 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%43, %39]
    %45 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%40, %8]
    %46 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%45, %40]
    %47 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%43, %39]
    %48 = affine.max affine_map<()[s0] -> (0, s0)>()[%47]
    %49 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%46, %40]
    %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
    %51 = arith.cmpi slt, %48, %c32 : index
    %52 = arith.cmpi slt, %50, %c128 : index
    %53 = arith.ori %51, %52 : i1
    %subview_46 = memref.subview %reinterpret_cast_45[0, 0] [%48, %50] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_47 = memref.subview %32[0, 0] [%48, %50] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %53 {
      %collapse_shape_148 = memref.collapse_shape %32 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_4 : f32) outs(%collapse_shape_148 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %27]
    hivm.hir.load ins(%subview_46 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_47 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %54 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_48 = memref.collapse_shape %32 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_49 = memref.collapse_shape %54 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%collapse_shape_48 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_49 : memref<4096xf32, #hivm.address_space<ub>>)
    %55 = hivm.hir.pointer_cast(%c119808_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_50 = memref.subview %55[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%c120832_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%54 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%subview_50 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%56 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_51 = memref.collapse_shape %subview_50 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_52 = memref.subview %57[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_53, %offset_54, %sizes_55, %strides_56 = memref.extract_strided_metadata %collapse_shape_51 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_57 = memref.reinterpret_cast %base_buffer_53 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_58, %offset_59, %sizes_60, %strides_61 = memref.extract_strided_metadata %subview_52 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_62 = memref.reinterpret_cast %base_buffer_58 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%reinterpret_cast_57, %reinterpret_cast_57 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_62 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_63 = memref.subview %58[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_64, %offset_65, %sizes_66, %strides_67 = memref.extract_strided_metadata %subview_63 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_68 = memref.reinterpret_cast %base_buffer_64 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%reinterpret_cast_62 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_68 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %59 = hivm.hir.pointer_cast(%c104448_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_69 = memref.subview %59[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_70, %offset_71, %sizes_72, %strides_73 = memref.extract_strided_metadata %subview_69 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_74 = memref.reinterpret_cast %base_buffer_70 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%reinterpret_cast_57, %cst : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_74 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c105472_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_75 = memref.subview %60[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_76, %offset_77, %sizes_78, %strides_79 = memref.extract_strided_metadata %subview_75 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_80 = memref.reinterpret_cast %base_buffer_76 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c106496_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_68, %cst, %reinterpret_cast_74 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_80 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%61 : memref<16xf32, #hivm.address_space<ub>>)
    %subview_81 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_82, %offset_83, %sizes_84, %strides_85 = memref.extract_strided_metadata %subview_81 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_86 = memref.reinterpret_cast %base_buffer_82 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c103424_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %29]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_19, %reinterpret_cast_57, %reinterpret_cast_80 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_86 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%62 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_87 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_88, %offset_89, %sizes_90, %strides_91 = memref.extract_strided_metadata %subview_87 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_92 = memref.reinterpret_cast %base_buffer_88 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%reinterpret_cast_86, %cst_3 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_92 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_93 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_94, %offset_95, %sizes_96, %strides_97 = memref.extract_strided_metadata %subview_93 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_98 = memref.reinterpret_cast %base_buffer_94 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%reinterpret_cast_92 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_98 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_99 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_100, %offset_101, %sizes_102, %strides_103 = memref.extract_strided_metadata %subview_99 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_104 = memref.reinterpret_cast %base_buffer_100 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%reinterpret_cast_98, %reinterpret_cast_31 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_104 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_105 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_106, %offset_107, %sizes_108, %strides_109 = memref.extract_strided_metadata %subview_105 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_110 = memref.reinterpret_cast %base_buffer_106 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%reinterpret_cast_104 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_110 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) round_mode = <ceil>
    %subview_111 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_112, %offset_113, %sizes_114, %strides_115 = memref.extract_strided_metadata %subview_111 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_116 = memref.reinterpret_cast %base_buffer_112 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%reinterpret_cast_110, %cst_6 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_116 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_117 = memref.subview %31[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_118, %offset_119, %sizes_120, %strides_121 = memref.extract_strided_metadata %subview_117 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_122 = memref.reinterpret_cast %base_buffer_118 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%reinterpret_cast_116 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_122 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_123 = memref.subview %63[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %collapse_shape_124 = memref.collapse_shape %subview_123 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_125, %offset_126, %sizes_127, %strides_128 = memref.extract_strided_metadata %collapse_shape_124 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_129 = memref.reinterpret_cast %base_buffer_125 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%reinterpret_cast_38, %reinterpret_cast_122 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_129 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c104448_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%32, %subview_123 : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%32 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%64 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %65 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_130 = memref.collapse_shape %32 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_131 = memref.collapse_shape %65 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_130, %collapse_shape_130 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_131 : memref<4096xi1, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_132 = memref.collapse_shape %66 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_131 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_132 : memref<4096xi1, #hivm.address_space<ub>>)
    %67 = hivm.hir.pointer_cast(%c103936_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_133 = memref.collapse_shape %67 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_130, %cst_1 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_133 : memref<4096xf32, #hivm.address_space<ub>>)
    %68 = hivm.hir.pointer_cast(%c120320_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_134 = memref.collapse_shape %68 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %69 = hivm.hir.pointer_cast(%c136704_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_132, %cst_1, %collapse_shape_133 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_134 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<16xf32, #hivm.address_space<ub>>)
    %70 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_135 = memref.collapse_shape %70 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %71 = hivm.hir.pointer_cast(%c119808_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_40, %collapse_shape_130, %collapse_shape_134 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_135 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %27]
    %72 = hivm.hir.pointer_cast(%c119808_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_136 = memref.collapse_shape %72 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_135, %collapse_shape_135 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_136 : memref<4096xi1, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c119808_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_137 = memref.collapse_shape %73 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_136 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_137 : memref<4096xi1, #hivm.address_space<ub>>)
    %74 = hivm.hir.pointer_cast(%c120320_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_138 = memref.collapse_shape %74 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_135, %cst_2 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_138 : memref<4096xf32, #hivm.address_space<ub>>)
    %75 = hivm.hir.pointer_cast(%c136704_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_139 = memref.collapse_shape %75 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %76 = hivm.hir.pointer_cast(%c153088_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_137, %cst_2, %collapse_shape_138 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_139 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%76 : memref<16xf32, #hivm.address_space<ub>>)
    %77 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_140 = memref.collapse_shape %77 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %78 = hivm.hir.pointer_cast(%c119808_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_42, %collapse_shape_135, %collapse_shape_139 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_140 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%78 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_141 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [32, 128], strides: [%8, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_142 = memref.collapse_shape %30 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_140 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_142 : memref<4096xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_143 = memref.subview %30[0, 0] [%48, %50] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_144 = memref.subview %reinterpret_cast_141[0, 0] [%48, %50] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_143 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_144 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    %79 = arith.index_cast %34 : i32 to index
    %80 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%79, %39, %20]
    %reinterpret_cast_145 = memref.reinterpret_cast %arg5 to offset: [%80], sizes: [32], strides: [%20] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview_146 = memref.subview %subview_117[0] [%44] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_147 = memref.subview %reinterpret_cast_145[0] [%44] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_146 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_147 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %29]
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
