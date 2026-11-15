// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c51200_i64 = arith.constant 51200 : i64
  %c34816_i64 = arith.constant 34816 : i64
  %c151040_i64 = arith.constant 151040 : i64
  %c134656_i64 = arith.constant 134656 : i64
  %c118272_i64 = arith.constant 118272 : i64
  %c101888_i64 = arith.constant 101888 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c104448_i64 = arith.constant 104448 : i64
  %c103424_i64 = arith.constant 103424 : i64
  %c102400_i64 = arith.constant 102400 : i64
  %c118784_i64 = arith.constant 118784 : i64
  %c117760_i64 = arith.constant 117760 : i64
  %c101376_i64 = arith.constant 101376 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c84992_i64 = arith.constant 84992 : i64
  %c68608_i64 = arith.constant 68608 : i64
  %c67584_i64 = arith.constant 67584 : i64
  %cst = arith.constant 9.99999974E-5 : f32
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst_0 = arith.constant 0.00223214296 : f32
  %cst_1 = arith.constant -4.480000e+02 : f32
  %cst_2 = arith.constant 4.480000e+02 : f32
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst_3 = arith.constant 0.000000e+00 : f32
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
  %collapse_shape_4 = memref.collapse_shape %7 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_4 : memref<4096xf32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg7 : i32 to index
  %9 = arith.index_cast %arg6 : i32 to index
  %10 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_5 = memref.subview %10[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %subview_5 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast, %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_11 = memref.subview %11[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_12, %offset_13, %sizes_14, %strides_15 = memref.extract_strided_metadata %subview_11 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_16 = memref.reinterpret_cast %base_buffer_12 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%reinterpret_cast_10 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_16 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_17 = memref.collapse_shape %12 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<4096xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_18 = memref.collapse_shape %13 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_17 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<4096xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_19 = memref.collapse_shape %14 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_4, %collapse_shape_4 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<4096xi1, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
  %collapse_shape_20 = memref.collapse_shape %15 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_19 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<4096xi1, #hivm.address_space<ub>>)
  %16 = arith.divsi %arg7, %c128_i32 : i32
  %17 = arith.index_cast %16 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %18 = arith.index_cast %arg11 : i32 to index
    %19 = arith.index_cast %3 : i32 to index
    %20 = arith.index_cast %1 : i32 to index
    %21 = arith.index_cast %c48_i32 : i32 to index
    %22 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%18, %19, %21]
    %23 = arith.index_cast %22 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %24 = arith.select %23, %c1_i64, %c2_i64 : i64
    %25 = arith.index_cast %22 : index to i1
    %c0_i64_21 = arith.constant 0 : i64
    %c1_i64_22 = arith.constant 1 : i64
    %26 = arith.select %25, %c0_i64_21, %c1_i64_22 : i64
    %27 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<32x128xf16, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c32768_i64, %c33792_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %30 = arith.divsi %arg11, %arg10 : i32
    %31 = arith.remsi %30, %arg9 : i32
    %32 = arith.divsi %arg11, %4 : i32
    %33 = arith.remsi %32, %arg8 : i32
    %34 = arith.muli %33, %c32_i32 : i32
    %35 = arith.muli %31, %c128_i32 : i32
    %36 = arith.index_cast %34 : i32 to index
    %37 = arith.index_cast %35 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%37, %36, %8]
    %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [32, 128], strides: [%8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%36, %9]
    %40 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%39, %36]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %36]
    %42 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%37, %8]
    %43 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%42, %37]
    %44 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%40, %36]
    %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
    %46 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%43, %37]
    %47 = affine.max affine_map<()[s0] -> (0, s0)>()[%46]
    %48 = arith.cmpi slt, %45, %c32 : index
    %49 = arith.cmpi slt, %47, %c128 : index
    %50 = arith.ori %48, %49 : i1
    %subview_24 = memref.subview %reinterpret_cast_23[0, 0] [%45, %47] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_25 = memref.subview %29[0, 0] [%45, %47] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %50 {
      %collapse_shape_89 = memref.collapse_shape %29 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%collapse_shape_89 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %24]
    hivm.hir.load ins(%subview_24 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_25 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_3 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %51 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %29 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %51 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%collapse_shape_26 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<4096xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c117760_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_28 = memref.subview %52[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c118784_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%51 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%subview_28 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%53 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_30 = memref.subview %54[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_31, %offset_32, %sizes_33, %strides_34 = memref.extract_strided_metadata %collapse_shape_29 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_35 = memref.reinterpret_cast %base_buffer_31 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_36, %offset_37, %sizes_38, %strides_39 = memref.extract_strided_metadata %subview_30 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_40 = memref.reinterpret_cast %base_buffer_36 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%reinterpret_cast_35, %reinterpret_cast_35 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_40 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %55 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_41 = memref.subview %55[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_42, %offset_43, %sizes_44, %strides_45 = memref.extract_strided_metadata %subview_41 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_46 = memref.reinterpret_cast %base_buffer_42 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%reinterpret_cast_40 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_46 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_47 = memref.subview %56[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_48, %offset_49, %sizes_50, %strides_51 = memref.extract_strided_metadata %subview_47 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_52 = memref.reinterpret_cast %base_buffer_48 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%reinterpret_cast_35, %cst : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_52 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_53 = memref.subview %57[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_54, %offset_55, %sizes_56, %strides_57 = memref.extract_strided_metadata %subview_53 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_58 = memref.reinterpret_cast %base_buffer_54 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %58 = hivm.hir.pointer_cast(%c104448_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_46, %cst, %reinterpret_cast_52 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_58 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%58 : memref<16xf32, #hivm.address_space<ub>>)
    %subview_59 = memref.subview %28[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_60, %offset_61, %sizes_62, %strides_63 = memref.extract_strided_metadata %subview_59 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_64 = memref.reinterpret_cast %base_buffer_60 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c101376_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %26]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_16, %reinterpret_cast_35, %reinterpret_cast_58 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_64 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%59 : memref<8xf32, #hivm.address_space<ub>>)
    %subview_65 = memref.subview %28[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_66, %offset_67, %sizes_68, %strides_69 = memref.extract_strided_metadata %subview_65 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_70 = memref.reinterpret_cast %base_buffer_66 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%reinterpret_cast_64, %cst_0 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_70 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %subview_65 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
    %60 = hivm.hir.pointer_cast(%c101376_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%29, %expand_shape : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%29 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %61 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_71 = memref.collapse_shape %29 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_72 = memref.collapse_shape %61 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_71, %collapse_shape_71 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_72 : memref<4096xi1, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_73 = memref.collapse_shape %62 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_72 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_73 : memref<4096xi1, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c101888_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_74 = memref.collapse_shape %63 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_71, %cst_1 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_74 : memref<4096xf32, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c118272_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_75 = memref.collapse_shape %64 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %65 = hivm.hir.pointer_cast(%c134656_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_73, %cst_1, %collapse_shape_74 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_75 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%65 : memref<16xf32, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_76 = memref.collapse_shape %66 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c117760_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_18, %collapse_shape_71, %collapse_shape_75 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_76 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %24]
    %68 = hivm.hir.pointer_cast(%c117760_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_77 = memref.collapse_shape %68 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_76, %collapse_shape_76 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_77 : memref<4096xi1, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c117760_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_78 = memref.collapse_shape %69 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_77 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_78 : memref<4096xi1, #hivm.address_space<ub>>)
    %70 = hivm.hir.pointer_cast(%c118272_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_79 = memref.collapse_shape %70 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_76, %cst_2 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_79 : memref<4096xf32, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c134656_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_80 = memref.collapse_shape %71 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %72 = hivm.hir.pointer_cast(%c151040_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_78, %cst_2, %collapse_shape_79 : memref<4096xi1, #hivm.address_space<ub>>, f32, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_80 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%72 : memref<16xf32, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c34816_i64, %c51200_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_81 = memref.collapse_shape %73 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %74 = hivm.hir.pointer_cast(%c117760_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_20, %collapse_shape_76, %collapse_shape_80 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_81 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%74 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_82 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [32, 128], strides: [%8, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_83 = memref.collapse_shape %27 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_81 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_83 : memref<4096xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_84 = memref.subview %27[0, 0] [%45, %47] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_85 = memref.subview %reinterpret_cast_82[0, 0] [%45, %47] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_84 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_85 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    %75 = arith.index_cast %31 : i32 to index
    %76 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%75, %36, %17]
    %reinterpret_cast_86 = memref.reinterpret_cast %arg5 to offset: [%76], sizes: [32], strides: [%17] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview_87 = memref.subview %subview_65[0] [%41] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_88 = memref.subview %reinterpret_cast_86[0] [%41] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_87 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_88 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %26]
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
