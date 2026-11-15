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
  %6 = arith.index_cast %arg9 : i32 to index
  %7 = arith.muli %arg11, %arg9 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg15 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg15 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_2, %c1_i64 : i64
    %15 = hivm.hir.pointer_cast(%c1376_i64, %c2048_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c1120_i64, %c1792_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c1056_i64, %c1760_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c128_i64, %c1632_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<4x8xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg15, %4 : i32
    %20 = arith.remsi %19, %arg12 : i32
    %21 = arith.divsi %20, %arg13 : i32
    %22 = arith.remsi %20, %arg13 : i32
    %23 = arith.muli %21, %arg10 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = arith.index_cast %22 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %25]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [4], strides: [%6] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %18[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast_3 : memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %27 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_5 = memref.subview %27[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %subview_4 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_5 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%reinterpret_cast_10, %reinterpret_cast_10 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_16 = memref.subview %28[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_17, %offset_18, %sizes_19, %strides_20 = memref.extract_strided_metadata %subview_16 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_21 = memref.reinterpret_cast %base_buffer_17 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%reinterpret_cast_15 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_21 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %29 = hivm.hir.pointer_cast(%c384_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_22 = memref.subview %29[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %subview_22 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%reinterpret_cast_10, %reinterpret_cast : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_27 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
    %subview_28 = memref.subview %30[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_29, %offset_30, %sizes_31, %strides_32 = memref.extract_strided_metadata %subview_28 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%reinterpret_cast_21, %reinterpret_cast_27 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_33 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    %31 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview_34 = memref.subview %31[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_35, %offset_36, %sizes_37, %strides_38 = memref.extract_strided_metadata %subview_34 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_39 = memref.reinterpret_cast %base_buffer_35 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c640_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%reinterpret_cast_33, %cst_0, %reinterpret_cast_10 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, f32, memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_39 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%32 : memref<16xf32, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c704_i64) : memref<1xf32, #hivm.address_space<ub>>
    %base_buffer_40, %offset_41, %sizes_42, %strides_43 = memref.extract_strided_metadata %33 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_44 = memref.reinterpret_cast %base_buffer_40 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c736_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%reinterpret_cast_39 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_44 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%34 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %35 = memref.load %33[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %36 = arith.cmpf oeq, %35, %cst_0 : f32
    %37 = arith.select %36, %cst_1, %35 : f32
    %38 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview_45 = memref.subview %38[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %39 = arith.subf %cst_1, %37 : f32
    %base_buffer_46, %offset_47, %sizes_48, %strides_49 = memref.extract_strided_metadata %subview_45 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_50 = memref.reinterpret_cast %base_buffer_46 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%reinterpret_cast_39, %39 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_50 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview_51 = memref.subview %40[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_52, %offset_53, %sizes_54, %strides_55 = memref.extract_strided_metadata %subview_51 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_56 = memref.reinterpret_cast %base_buffer_52 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%reinterpret_cast_50 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_56 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c864_i64) : memref<1xf32, #hivm.address_space<ub>>
    %base_buffer_57, %offset_58, %sizes_59, %strides_60 = memref.extract_strided_metadata %41 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_61 = memref.reinterpret_cast %base_buffer_57 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c896_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%reinterpret_cast_56 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_61 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%42 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %43 = memref.load %41[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %43, %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %45 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%44 : memref<1xf32, #hivm.address_space<ub>>) outs(%45 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %46 = memref.load %45[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %47 = arith.addf %46, %37 : f32
    %48 = arith.addi %23, %22 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %14]
    memref.store %47, %17[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %49 = arith.index_cast %48 : i32 to index
    %reinterpret_cast_62 = memref.reinterpret_cast %arg6 to offset: [%49], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%17 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_62 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %14]
    %50 = arith.addi %7, %23 : i32
    %51 = arith.addi %50, %22 : i32
    %52 = arith.index_cast %51 : i32 to index
    %reinterpret_cast_63 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %53 = memref.load %reinterpret_cast_63[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %54 = arith.subf %53, %47 : f32
    %55 = arith.cmpf une, %54, %54 : f32
    %56 = arith.cmpf oeq, %54, %cst : f32
    %57 = arith.ori %55, %56 : i1
    %58 = arith.select %57, %cst_0, %54 : f32
    %59 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %58, %59[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %60 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%59 : memref<1xf32, #hivm.address_space<ub>>) outs(%60 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %61 = memref.load %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %62 = arith.muli %21, %arg7 : i32
    %63 = arith.muli %22, %arg8 : i32
    %64 = arith.addi %62, %63 : i32
    %65 = arith.index_cast %64 : i32 to index
    %reinterpret_cast_64 = memref.reinterpret_cast %arg3 to offset: [%65], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_64 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%16 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%16, %61 : memref<64xf32, #hivm.address_space<ub>>, f32) outs(%15 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_65 = memref.reinterpret_cast %arg4 to offset: [%65], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%15 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_65 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
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
