// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_chunk_cumsum_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: f32, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c29600_i64 = arith.constant 29600 : i64
  %c29536_i64 = arith.constant 29536 : i64
  %c28512_i64 = arith.constant 28512 : i64
  %c27488_i64 = arith.constant 27488 : i64
  %c27424_i64 = arith.constant 27424 : i64
  %c26400_i64 = arith.constant 26400 : i64
  %c25376_i64 = arith.constant 25376 : i64
  %c29568_i64 = arith.constant 29568 : i64
  %c25344_i64 = arith.constant 25344 : i64
  %c25280_i64 = arith.constant 25280 : i64
  %c24256_i64 = arith.constant 24256 : i64
  %c24224_i64 = arith.constant 24224 : i64
  %c23200_i64 = arith.constant 23200 : i64
  %c23136_i64 = arith.constant 23136 : i64
  %c22112_i64 = arith.constant 22112 : i64
  %c21088_i64 = arith.constant 21088 : i64
  %c21056_i64 = arith.constant 21056 : i64
  %c20032_i64 = arith.constant 20032 : i64
  %c20000_i64 = arith.constant 20000 : i64
  %c18976_i64 = arith.constant 18976 : i64
  %c18912_i64 = arith.constant 18912 : i64
  %c17888_i64 = arith.constant 17888 : i64
  %c16864_i64 = arith.constant 16864 : i64
  %c16832_i64 = arith.constant 16832 : i64
  %c15808_i64 = arith.constant 15808 : i64
  %c15776_i64 = arith.constant 15776 : i64
  %c14752_i64 = arith.constant 14752 : i64
  %c13728_i64 = arith.constant 13728 : i64
  %c12704_i64 = arith.constant 12704 : i64
  %c12640_i64 = arith.constant 12640 : i64
  %c11616_i64 = arith.constant 11616 : i64
  %c10592_i64 = arith.constant 10592 : i64
  %c9568_i64 = arith.constant 9568 : i64
  %c9504_i64 = arith.constant 9504 : i64
  %c8480_i64 = arith.constant 8480 : i64
  %c7456_i64 = arith.constant 7456 : i64
  %c6432_i64 = arith.constant 6432 : i64
  %c6336_i64 = arith.constant 6336 : i64
  %c6304_i64 = arith.constant 6304 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c3104_i64 = arith.constant 3104 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c32_i32 = arith.constant 32 : i32
  %cst_2 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg22, %arg23 : i32
  %1 = arith.muli %0, %arg24 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg24 : i32
  %5 = arith.divsi %3, %arg24 : i32
  %6 = arith.remsi %5, %arg23 : i32
  %7 = arith.muli %arg24, %arg23 : i32
  %8 = arith.divsi %3, %7 : i32
  %9 = arith.remsi %8, %arg22 : i32
  %10 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %10 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<256xf32, #hivm.address_space<ub>>)
  %11 = arith.muli %9, %arg14 : i32
  %12 = arith.muli %6, %arg11 : i32
  %13 = arith.muli %12, %arg15 : i32
  %14 = arith.addi %11, %13 : i32
  %15 = arith.extsi %14 : i32 to i64
  %16 = arith.muli %9, %arg16 : i32
  %17 = arith.muli %6, %arg17 : i32
  %18 = arith.addi %16, %17 : i32
  %19 = arith.extsi %18 : i32 to i64
  %20 = arith.muli %9, %arg19 : i32
  %21 = arith.muli %6, %arg20 : i32
  %22 = arith.addi %20, %21 : i32
  %23 = arith.extsi %22 : i32 to i64
  %24 = arith.muli %4, %c32_i32 : i32
  %25 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x8xi32, #hivm.address_space<ub>>
  %subview = memref.subview %25[0, 0] [32, 1] [1, 1] : memref<32x8xi32, #hivm.address_space<ub>> to memref<32xi32, strided<[8]>, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%subview : memref<32xi32, strided<[8]>, #hivm.address_space<ub>>)
  %26 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x8xi32, #hivm.address_space<ub>>
  %subview_3 = memref.subview %26[0, 0] [32, 1] [1, 1] : memref<32x8xi32, #hivm.address_space<ub>> to memref<32xi32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xi32, strided<[8]>, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<i32, #hivm.address_space<ub>> to memref<32x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  %base_buffer_4, %offset_5, %sizes_6, %strides_7 = memref.extract_strided_metadata %subview_3 : memref<32xi32, strided<[8]>, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_8 = memref.reinterpret_cast %base_buffer_4 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<i32, #hivm.address_space<ub>> to memref<32x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%reinterpret_cast, %24 : memref<32x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>, i32) outs(%reinterpret_cast_8 : memref<32x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %27 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_9 = memref.subview %27[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %subview_9 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%reinterpret_cast_8 : memref<32x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_14 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %28 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%28 : memref<8xi32, #hivm.address_space<ub>>)
  %29 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%28 : memref<8xi32, #hivm.address_space<ub>>) outs(%29 : memref<8xf32, #hivm.address_space<ub>>)
  %30 = arith.subi %arg9, %12 : i32
  %31 = arith.minsi %arg11, %30 : i32
  %32 = arith.sitofp %31 : i32 to f32
  %33 = arith.sitofp %arg10 : i32 to f32
  %34 = hivm.hir.pointer_cast(%c2080_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_15 = memref.subview %34[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %collapse_shape_16 = memref.collapse_shape %subview_15 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_17, %offset_18, %sizes_19, %strides_20 = memref.extract_strided_metadata %collapse_shape_16 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_21 = memref.reinterpret_cast %base_buffer_17 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%33 : f32) outs(%reinterpret_cast_21 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %35 = hivm.hir.pointer_cast(%c2080_i64) : memref<32x256x1xi1, #hivm.address_space<ub>>
  %subview_22 = memref.subview %35[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x256x1xi1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  %collapse_shape_23 = memref.collapse_shape %subview_22 [[0, 1]] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> into memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_24, %offset_25, %sizes_26, %strides_27 = memref.extract_strided_metadata %collapse_shape_23 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast_14, %reinterpret_cast_21 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_28 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) compare_mode = <lt>
  %36 = hivm.hir.pointer_cast(%c3104_i64) : memref<1x8xf32, #hivm.address_space<ub>>
  %collapse_shape_29 = memref.collapse_shape %36 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%32 : f32) outs(%collapse_shape_29 : memref<8xf32, #hivm.address_space<ub>>)
  %37 = hivm.hir.pointer_cast(%c3104_i64) : memref<1x8xi1, #hivm.address_space<ub>>
  %collapse_shape_30 = memref.collapse_shape %37 [[0, 1]] : memref<1x8xi1, #hivm.address_space<ub>> into memref<8xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%29, %collapse_shape_29 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<8xi1, #hivm.address_space<ub>>) compare_mode = <lt>
  %38 = hivm.hir.pointer_cast(%c3136_i64) : memref<32x16x1xf16, #hivm.address_space<ub>>
  %subview_31 = memref.subview %38[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x16x1xf16, #hivm.address_space<ub>> to memref<32x1xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  %collapse_shape_32 = memref.collapse_shape %subview_31 [[0, 1]] : memref<32x1xf16, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf16, strided<[16]>, #hivm.address_space<ub>>
  %base_buffer_33, %offset_34, %sizes_35, %strides_36 = memref.extract_strided_metadata %collapse_shape_32 : memref<32xf16, strided<[16]>, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_37 = memref.reinterpret_cast %base_buffer_33 to offset: [0], sizes: [32, 1], strides: [16, 1] : memref<f16, #hivm.address_space<ub>> to memref<32x1xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  %39 = hivm.hir.pointer_cast(%c4160_i64) : memref<48xf16, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%reinterpret_cast_28 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_37 : memref<32x1xf16, strided<[16, 1]>, #hivm.address_space<ub>>) temp_buffer(%39 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
  %40 = hivm.hir.pointer_cast(%c4256_i64) : memref<32x16x1xf16, #hivm.address_space<ub>>
  %subview_38 = memref.subview %40[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x16x1xf16, #hivm.address_space<ub>> to memref<32x8xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  %41 = hivm.hir.pointer_cast(%c5280_i64) : memref<512xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%subview_31 : memref<32x1xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview_38 : memref<32x8xf16, strided<[16, 1]>, #hivm.address_space<ub>>) temp_buffer(%41 : memref<512xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
  %42 = hivm.hir.pointer_cast(%c4256_i64) : memref<32x256x1xi1, #hivm.address_space<ub>>
  %subview_39 = memref.subview %42[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x256x1xi1, #hivm.address_space<ub>> to memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%subview_38, %cst : memref<32x8xf16, strided<[16, 1]>, #hivm.address_space<ub>>, f16) outs(%subview_39 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %43 = hivm.hir.pointer_cast(%c4256_i64) : memref<32x256x1xi1, #hivm.address_space<ub>>
  %subview_40 = memref.subview %43[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x256x1xi1, #hivm.address_space<ub>> to memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%subview_39 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_40 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %44 = hivm.hir.pointer_cast(%c6304_i64) : memref<1x8xf16, #hivm.address_space<ub>>
  %collapse_shape_41 = memref.collapse_shape %44 [[0, 1]] : memref<1x8xf16, #hivm.address_space<ub>> into memref<8xf16, #hivm.address_space<ub>>
  %45 = hivm.hir.pointer_cast(%c6336_i64) : memref<48xf16, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%collapse_shape_30 : memref<8xi1, #hivm.address_space<ub>>) outs(%collapse_shape_41 : memref<8xf16, #hivm.address_space<ub>>) temp_buffer(%45 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
  %46 = hivm.hir.pointer_cast(%c6432_i64) : memref<32x16x1xf16, #hivm.address_space<ub>>
  %subview_42 = memref.subview %46[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x16x1xf16, #hivm.address_space<ub>> to memref<32x8xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%44 : memref<1x8xf16, #hivm.address_space<ub>>) outs(%subview_42 : memref<32x8xf16, strided<[16, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
  %47 = hivm.hir.pointer_cast(%c6432_i64) : memref<32x256x1xi1, #hivm.address_space<ub>>
  %subview_43 = memref.subview %47[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x256x1xi1, #hivm.address_space<ub>> to memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%subview_42, %cst : memref<32x8xf16, strided<[16, 1]>, #hivm.address_space<ub>>, f16) outs(%subview_43 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %48 = hivm.hir.pointer_cast(%c6432_i64) : memref<32x256x1xi1, #hivm.address_space<ub>>
  %subview_44 = memref.subview %48[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x256x1xi1, #hivm.address_space<ub>> to memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%subview_43 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_44 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %49 = hivm.hir.pointer_cast(%c6432_i64) : memref<32x256x1xi1, #hivm.address_space<ub>>
  %subview_45 = memref.subview %49[0, 0, 0] [32, 8, 1] [1, 1, 1] : memref<32x256x1xi1, #hivm.address_space<ub>> to memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vand ins(%subview_40, %subview_44 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_45 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
  %50 = hivm.hir.pointer_cast(%c7456_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  scf.for %arg25 = %c0 to %c32 step %c1 {
    %95 = arith.index_cast %arg25 : index to i32
    %96 = arith.addi %24, %95 : i32
    %97 = arith.sitofp %96 : i32 to f32
    %98 = arith.fptosi %97 : f32 to i32
    scf.for %arg26 = %c0 to %c8 step %c1 {
      %99 = arith.index_cast %arg26 : index to i32
      %100 = arith.sitofp %99 : i32 to f32
      %101 = arith.fptosi %100 : f32 to i32
      %102 = arith.muli %101, %arg15 : i32
      %103 = arith.addi %98, %102 : i32
      %104 = arith.extsi %103 : i32 to i64
      %105 = arith.addi %15, %104 : i64
      %106 = arith.index_cast %105 : i64 to index
      %reinterpret_cast_106 = memref.reinterpret_cast %arg3 to offset: [%106], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %107 = memref.load %reinterpret_cast_106[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.store %107, %50[%arg25, %arg26] : memref<32x8xf32, #hivm.address_space<ub>>
    } {ExtractedLoadOrStore}
  } {ExtractedLoadOrStore}
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %51 = hivm.hir.pointer_cast(%c8480_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %52 = hivm.hir.pointer_cast(%c9504_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%subview_45, %50, %cst_2 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x8xf32, #hivm.address_space<ub>>, f32) outs(%51 : memref<32x8xf32, #hivm.address_space<ub>>) temp_buffer(%52 : memref<16xf32, #hivm.address_space<ub>>)
  %53 = hivm.hir.pointer_cast(%c9568_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_46 = memref.subview %53[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_47, %offset_48, %sizes_49, %strides_50 = memref.extract_strided_metadata %subview_46 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_51 = memref.reinterpret_cast %base_buffer_47 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%33 : f32) outs(%reinterpret_cast_51 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %54 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x256xi1, #hivm.address_space<ub>>
  %subview_52 = memref.subview %54[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
  %base_buffer_53, %offset_54, %sizes_55, %strides_56 = memref.extract_strided_metadata %subview_52 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_57 = memref.reinterpret_cast %base_buffer_53 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%reinterpret_cast_14, %reinterpret_cast_51 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_57 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>) compare_mode = <lt>
  %55 = hivm.hir.pointer_cast(%c10592_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_58 = memref.subview %55[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  scf.for %arg25 = %c0 to %c32 step %c1 {
    %95 = arith.index_cast %arg25 : index to i32
    %96 = arith.addi %24, %95 : i32
    %97 = arith.sitofp %96 : i32 to f32
    %98 = arith.fptosi %97 : f32 to i32
    %99 = arith.index_cast %98 : i32 to index
    %reinterpret_cast_106 = memref.reinterpret_cast %arg5 to offset: [%99], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %100 = memref.load %reinterpret_cast_106[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    memref.store %100, %subview_58[%arg25] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  } {ExtractedLoadOrStore}
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %56 = hivm.hir.pointer_cast(%c11616_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_59 = memref.subview %56[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_60, %offset_61, %sizes_62, %strides_63 = memref.extract_strided_metadata %subview_58 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_64 = memref.reinterpret_cast %base_buffer_60 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %base_buffer_65, %offset_66, %sizes_67, %strides_68 = memref.extract_strided_metadata %subview_59 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_69 = memref.reinterpret_cast %base_buffer_65 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %57 = hivm.hir.pointer_cast(%c12640_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%reinterpret_cast_57, %reinterpret_cast_64, %cst_2 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_69 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%57 : memref<16xf32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %subview_59 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
  %58 = hivm.hir.pointer_cast(%c8480_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %59 = hivm.hir.pointer_cast(%c12704_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%51, %expand_shape : memref<32x8xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%58 : memref<32x8xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
  %60 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_70 = memref.collapse_shape %58 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %collapse_shape_71 = memref.collapse_shape %60 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_70, %collapse_shape : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_71 : memref<256xi1, #hivm.address_space<ub>>) compare_mode = <le>
  %61 = hivm.hir.pointer_cast(%c13728_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_72 = memref.collapse_shape %61 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vexp ins(%collapse_shape_70 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_72 : memref<256xf32, #hivm.address_space<ub>>)
  %62 = hivm.hir.pointer_cast(%c13728_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_73 = memref.collapse_shape %62 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%collapse_shape_72, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_73 : memref<256xf32, #hivm.address_space<ub>>)
  %63 = hivm.hir.pointer_cast(%c13728_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_74 = memref.collapse_shape %63 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%collapse_shape_73 : memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_74 : memref<256xf32, #hivm.address_space<ub>>)
  %64 = hivm.hir.pointer_cast(%c14752_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_75 = memref.collapse_shape %64 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %65 = hivm.hir.pointer_cast(%c15776_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_71, %collapse_shape_74, %collapse_shape_70 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_75 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%65 : memref<8xf32, #hivm.address_space<ub>>)
  %66 = hivm.hir.pointer_cast(%c15808_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_76 = memref.collapse_shape %66 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg12 : f32) outs(%collapse_shape_76 : memref<256xf32, #hivm.address_space<ub>>)
  %67 = hivm.hir.pointer_cast(%c16832_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_77 = memref.collapse_shape %67 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_75, %collapse_shape_75 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_77 : memref<256xi1, #hivm.address_space<ub>>)
  %68 = hivm.hir.pointer_cast(%c16832_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_77 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_78 : memref<256xi1, #hivm.address_space<ub>>)
  %69 = hivm.hir.pointer_cast(%c15808_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_79 = memref.collapse_shape %69 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_76, %collapse_shape_76 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_79 : memref<256xi1, #hivm.address_space<ub>>)
  %70 = hivm.hir.pointer_cast(%c15808_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_80 = memref.collapse_shape %70 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_79 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_80 : memref<256xi1, #hivm.address_space<ub>>)
  %71 = hivm.hir.pointer_cast(%c16864_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_81 = memref.collapse_shape %71 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%collapse_shape_75, %arg12 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_81 : memref<256xf32, #hivm.address_space<ub>>)
  %72 = hivm.hir.pointer_cast(%c17888_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_82 = memref.collapse_shape %72 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %73 = hivm.hir.pointer_cast(%c18912_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_78, %arg12, %collapse_shape_81 : memref<256xi1, #hivm.address_space<ub>>, f32, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_82 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%73 : memref<16xf32, #hivm.address_space<ub>>)
  %74 = hivm.hir.pointer_cast(%c18976_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_83 = memref.collapse_shape %74 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %75 = hivm.hir.pointer_cast(%c20000_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_80, %collapse_shape_75, %collapse_shape_82 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_83 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%75 : memref<8xf32, #hivm.address_space<ub>>)
  %76 = hivm.hir.pointer_cast(%c20032_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_84 = memref.collapse_shape %76 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg13 : f32) outs(%collapse_shape_84 : memref<256xf32, #hivm.address_space<ub>>)
  %77 = hivm.hir.pointer_cast(%c21056_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_85 = memref.collapse_shape %77 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_83, %collapse_shape_83 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_85 : memref<256xi1, #hivm.address_space<ub>>)
  %78 = hivm.hir.pointer_cast(%c21056_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_86 = memref.collapse_shape %78 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_85 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_86 : memref<256xi1, #hivm.address_space<ub>>)
  %79 = hivm.hir.pointer_cast(%c20032_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_87 = memref.collapse_shape %79 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_84, %collapse_shape_84 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_87 : memref<256xi1, #hivm.address_space<ub>>)
  %80 = hivm.hir.pointer_cast(%c20032_i64) : memref<32x8xi1, #hivm.address_space<ub>>
  %collapse_shape_88 = memref.collapse_shape %80 [[0, 1]] : memref<32x8xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_87 : memref<256xi1, #hivm.address_space<ub>>) outs(%collapse_shape_88 : memref<256xi1, #hivm.address_space<ub>>)
  %81 = hivm.hir.pointer_cast(%c21088_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_89 = memref.collapse_shape %81 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vmin ins(%collapse_shape_83, %arg13 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_89 : memref<256xf32, #hivm.address_space<ub>>)
  %82 = hivm.hir.pointer_cast(%c22112_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_90 = memref.collapse_shape %82 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %83 = hivm.hir.pointer_cast(%c23136_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_86, %arg13, %collapse_shape_89 : memref<256xi1, #hivm.address_space<ub>>, f32, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_90 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%83 : memref<16xf32, #hivm.address_space<ub>>)
  %84 = hivm.hir.pointer_cast(%c23200_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %collapse_shape_91 = memref.collapse_shape %84 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
  %85 = hivm.hir.pointer_cast(%c24224_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_88, %collapse_shape_83, %collapse_shape_90 : memref<256xi1, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_91 : memref<256xf32, #hivm.address_space<ub>>) temp_buffer(%85 : memref<8xf32, #hivm.address_space<ub>>)
  %86 = hivm.hir.pointer_cast(%c24256_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %87 = hivm.hir.pointer_cast(%c25280_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%subview_45, %84, %cst_2 : memref<32x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x8xf32, #hivm.address_space<ub>>, f32) outs(%86 : memref<32x8xf32, #hivm.address_space<ub>>) temp_buffer(%87 : memref<16xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %88 = arith.sitofp %arg11 : i32 to f32
  %view = memref.view %arg1[%c64][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
  hivm.hir.sync_block_lock lock_var(%view : memref<1xi64, #hivm.address_space<gm>>)
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg25 = %c0 to %c32 step %c1 {
    %95 = arith.index_cast %arg25 : index to i32
    %96 = arith.addi %24, %95 : i32
    %97 = arith.sitofp %96 : i32 to f32
    %98 = arith.fptosi %97 : f32 to i32
    %99 = arith.muli %98, %arg18 : i32
    %100 = arith.cmpf olt, %97, %33 : f32
    scf.for %arg26 = %c0 to %c8 step %c1 {
      %101 = affine.apply affine_map<()[s0, s1] -> (s0 mod 2)>()[%arg26, %arg25]
      %102 = arith.index_cast %101 : index to i1
      %c0_i64_106 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %103 = arith.select %102, %c0_i64_106, %c1_i64 : i64
      %104 = hivm.hir.pointer_cast(%c25344_i64, %c29568_i64) : memref<1xf32, #hivm.address_space<ub>>
      annotation.mark %104 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
      %105 = arith.index_cast %arg26 : index to i32
      %106 = arith.sitofp %105 : i32 to f32
      %107 = arith.fptosi %106 : f32 to i32
      %108 = arith.addi %99, %107 : i32
      %109 = arith.extsi %108 : i32 to i64
      %110 = arith.addi %19, %109 : i64
      %111 = memref.load %86[%arg25, %arg26] {DiscreteMemAccess} : memref<32x8xf32, #hivm.address_space<ub>>
      %112 = arith.cmpf olt, %106, %88 : f32
      %113 = arith.andi %100, %112 : i1
      scf.if %113 {
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %103]
        memref.store %111, %104[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %114 = arith.index_cast %110 : i64 to index
        %reinterpret_cast_107 = memref.reinterpret_cast %arg6 to offset: [%114], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%104 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_107 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %103]
      } {pipeline.veconly}
    } {ExtractedLoadOrStore, pipeline.veconly}
  } {ExtractedLoadOrStore}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.sync_block_unlock lock_var(%view : memref<1xi64, #hivm.address_space<gm>>)
  %89 = hivm.hir.pointer_cast(%c25376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_92 = memref.subview %89[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  scf.for %arg25 = %c0 to %c32 step %c1 {
    %95 = arith.index_cast %arg25 : index to i32
    %96 = arith.addi %24, %95 : i32
    %97 = arith.sitofp %96 : i32 to f32
    %98 = arith.fptosi %97 : f32 to i32
    %99 = arith.index_cast %98 : i32 to index
    %reinterpret_cast_106 = memref.reinterpret_cast %arg4 to offset: [%99], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %100 = memref.load %reinterpret_cast_106[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    memref.store %100, %subview_92[%arg25] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  } {ExtractedLoadOrStore}
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %90 = hivm.hir.pointer_cast(%c26400_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %subview_93 = memref.subview %90[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
  %base_buffer_94, %offset_95, %sizes_96, %strides_97 = memref.extract_strided_metadata %subview_92 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_98 = memref.reinterpret_cast %base_buffer_94 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %base_buffer_99, %offset_100, %sizes_101, %strides_102 = memref.extract_strided_metadata %subview_93 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_103 = memref.reinterpret_cast %base_buffer_99 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %91 = hivm.hir.pointer_cast(%c27424_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vsel ins(%reinterpret_cast_57, %reinterpret_cast_98, %cst_2 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_103 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) temp_buffer(%91 : memref<16xf32, #hivm.address_space<ub>>)
  %expand_shape_104 = memref.expand_shape %subview_93 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
  %92 = hivm.hir.pointer_cast(%c24256_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  %93 = hivm.hir.pointer_cast(%c27488_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%86, %expand_shape_104 : memref<32x8xf32, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>) outs(%92 : memref<32x8xf32, #hivm.address_space<ub>>) temp_buffer(%93 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %94 = hivm.hir.pointer_cast(%c28512_i64) : memref<32x8xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg25 = %c0 to %c32 step %c1 {
    %95 = memref.load %92[%arg25, %c0] : memref<32x8xf32, #hivm.address_space<ub>>
    memref.store %95, %94[%arg25, %c0] : memref<32x8xf32, #hivm.address_space<ub>>
    scf.for %arg26 = %c1 to %c8 step %c1 {
      %96 = memref.load %92[%arg25, %arg26] : memref<32x8xf32, #hivm.address_space<ub>>
      %97 = arith.subi %arg26, %c1 : index
      %98 = memref.load %94[%arg25, %97] : memref<32x8xf32, #hivm.address_space<ub>>
      %99 = arith.addf %96, %98 : f32
      memref.store %99, %94[%arg25, %arg26] : memref<32x8xf32, #hivm.address_space<ub>>
    }
  }
  %view_105 = memref.view %arg1[%c0][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
  hivm.hir.sync_block_lock lock_var(%view_105 : memref<1xi64, #hivm.address_space<gm>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg25 = %c0 to %c32 step %c1 {
    %95 = arith.index_cast %arg25 : index to i32
    %96 = arith.addi %24, %95 : i32
    %97 = arith.sitofp %96 : i32 to f32
    %98 = arith.fptosi %97 : f32 to i32
    %99 = arith.muli %98, %arg21 : i32
    %100 = arith.cmpf olt, %97, %33 : f32
    scf.for %arg26 = %c0 to %c8 step %c1 {
      %101 = affine.apply affine_map<()[s0, s1] -> (s0 mod 2)>()[%arg26, %arg25]
      %102 = arith.index_cast %101 : index to i1
      %c0_i64_106 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %103 = arith.select %102, %c0_i64_106, %c1_i64 : i64
      %104 = hivm.hir.pointer_cast(%c29536_i64, %c29600_i64) : memref<1xf32, #hivm.address_space<ub>>
      annotation.mark %104 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
      %105 = arith.index_cast %arg26 : index to i32
      %106 = arith.sitofp %105 : i32 to f32
      %107 = arith.fptosi %106 : f32 to i32
      %108 = arith.addi %99, %107 : i32
      %109 = arith.extsi %108 : i32 to i64
      %110 = arith.addi %23, %109 : i64
      %111 = memref.load %94[%arg25, %arg26] {DiscreteMemAccess} : memref<32x8xf32, #hivm.address_space<ub>>
      %112 = arith.cmpf olt, %106, %88 : f32
      %113 = arith.andi %100, %112 : i1
      scf.if %113 {
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %103]
        memref.store %111, %104[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %114 = arith.index_cast %110 : i64 to index
        %reinterpret_cast_107 = memref.reinterpret_cast %arg7 to offset: [%114], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%104 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_107 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %103]
      } {pipeline.veconly}
    } {ExtractedLoadOrStore, pipeline.veconly}
  } {ExtractedLoadOrStore}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.sync_block_unlock lock_var(%view_105 : memref<1xi64, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
