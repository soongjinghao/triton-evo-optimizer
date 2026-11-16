// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c2048_i64 = arith.constant 2048 : i64
  %c1376_i64 = arith.constant 1376 : i64
  %c1792_i64 = arith.constant 1792 : i64
  %c1120_i64 = arith.constant 1120 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c1760_i64 = arith.constant 1760 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c896_i64 = arith.constant 896 : i64
  %c864_i64 = arith.constant 864 : i64
  %c736_i64 = arith.constant 736 : i64
  %c704_i64 = arith.constant 704 : i64
  %c640_i64 = arith.constant 640 : i64
  %c512_i64 = arith.constant 512 : i64
  %c384_i64 = arith.constant 384 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1632_i64 = arith.constant 1632 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0x7F800000 : f32
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8xf32, #hivm.address_space<ub>>
  %subview = memref.subview %5[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %6 = arith.extsi %arg10 : i32 to i64
  %7 = arith.index_cast %arg9 : i32 to index
  %8 = arith.extsi %arg7 : i32 to i64
  %9 = arith.extsi %arg8 : i32 to i64
  %10 = arith.muli %arg11, %arg9 : i32
  %11 = arith.extsi %10 : i32 to i64
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg15 = %3 to %1 step %c48_i32  : i32 {
    %12 = arith.index_cast %arg15 : i32 to index
    %13 = arith.index_cast %3 : i32 to index
    %14 = arith.index_cast %1 : i32 to index
    %15 = arith.index_cast %c48_i32 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%12, %13, %15]
    %17 = arith.index_cast %16 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %18 = arith.select %17, %c0_i64_2, %c1_i64 : i64
    %19 = hivm.hir.pointer_cast(%c1376_i64, %c2048_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c1120_i64, %c1792_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c1056_i64, %c1760_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c128_i64, %c1632_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<4x8xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %23 = arith.divsi %arg15, %4 : i32
    %24 = arith.remsi %23, %arg12 : i32
    %25 = arith.extsi %24 : i32 to i64
    %26 = arith.divsi %25, %c8_i64 : i64
    %27 = arith.remsi %25, %c8_i64 : i64
    %28 = arith.muli %26, %6 : i64
    %29 = arith.index_cast %28 : i64 to index
    %30 = arith.index_cast %27 : i64 to index
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [4], strides: [%7] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %22[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast_3 : memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %32 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_5 = memref.subview %32[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %subview_4 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_5 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%reinterpret_cast_10, %reinterpret_cast_10 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_16 = memref.subview %33[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_17, %offset_18, %sizes_19, %strides_20 = memref.extract_strided_metadata %subview_16 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_21 = memref.reinterpret_cast %base_buffer_17 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%reinterpret_cast_15 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_21 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %34 = hivm.hir.pointer_cast(%c384_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_22 = memref.subview %34[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %subview_22 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%reinterpret_cast_10, %reinterpret_cast : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_27 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %35 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_28 = memref.subview %35[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_29, %offset_30, %sizes_31, %strides_32 = memref.extract_strided_metadata %subview_28 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%reinterpret_cast_21, %reinterpret_cast_27 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_33 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %36 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview_34 = memref.subview %36[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_35, %offset_36, %sizes_37, %strides_38 = memref.extract_strided_metadata %subview_34 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_39 = memref.reinterpret_cast %base_buffer_35 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %37 = hivm.hir.pointer_cast(%c640_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_33, %cst_0, %reinterpret_cast_10 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_39 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%37 : memref<16xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c704_i64) : memref<1xf32, #hivm.address_space<ub>>
    %base_buffer_40, %offset_41, %sizes_42, %strides_43 = memref.extract_strided_metadata %38 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_44 = memref.reinterpret_cast %base_buffer_40 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c736_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%reinterpret_cast_39 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_44 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%39 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %40 = memref.load %38[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %41 = arith.cmpf oeq, %40, %cst_0 : f32
    %42 = arith.select %41, %cst_1, %40 : f32
    %43 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview_45 = memref.subview %43[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %44 = arith.subf %cst_1, %42 : f32
    %base_buffer_46, %offset_47, %sizes_48, %strides_49 = memref.extract_strided_metadata %subview_45 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_50 = memref.reinterpret_cast %base_buffer_46 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%reinterpret_cast_39, %44 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_50 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview_51 = memref.subview %45[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_52, %offset_53, %sizes_54, %strides_55 = memref.extract_strided_metadata %subview_51 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_56 = memref.reinterpret_cast %base_buffer_52 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%reinterpret_cast_50 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_56 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c864_i64) : memref<1xf32, #hivm.address_space<ub>>
    %base_buffer_57, %offset_58, %sizes_59, %strides_60 = memref.extract_strided_metadata %46 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_61 = memref.reinterpret_cast %base_buffer_57 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c896_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%reinterpret_cast_56 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_61 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%47 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %48 = memref.load %46[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %48, %49[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %50 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%49 : memref<1xf32, #hivm.address_space<ub>>) outs(%50 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %51 = memref.load %50[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %52 = arith.addf %51, %42 : f32
    %53 = arith.addi %28, %27 : i64
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %18]
    memref.store %52, %21[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %54 = arith.index_cast %53 : i64 to index
    %reinterpret_cast_62 = memref.reinterpret_cast %arg6 to offset: [%54], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%21 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_62 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %18]
    %55 = arith.muli %26, %8 : i64
    %56 = arith.muli %27, %9 : i64
    %57 = arith.addi %55, %56 : i64
    %58 = arith.addi %11, %28 : i64
    %59 = arith.addi %58, %27 : i64
    %60 = arith.index_cast %59 : i64 to index
    %reinterpret_cast_63 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %61 = memref.load %reinterpret_cast_63[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %62 = arith.subf %61, %52 : f32
    %63 = arith.cmpf une, %62, %62 : f32
    %64 = arith.cmpf oeq, %62, %cst : f32
    %65 = arith.ori %63, %64 : i1
    %66 = arith.select %65, %cst_0, %62 : f32
    %67 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %66, %67[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %68 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%67 : memref<1xf32, #hivm.address_space<ub>>) outs(%68 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %69 = memref.load %68[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %70 = arith.index_cast %57 : i64 to index
    %reinterpret_cast_64 = memref.reinterpret_cast %arg3 to offset: [%70], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_64 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%20 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %18]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%20, %69 : memref<64xf32, #hivm.address_space<ub>>, f32) outs(%19 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_65 = memref.reinterpret_cast %arg4 to offset: [%70], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%19 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_65 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %18]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
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
