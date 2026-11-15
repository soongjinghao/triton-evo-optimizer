// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @rwkv_mm_sparsity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c109216_i64 = arith.constant 109216 : i64
  %c175264_i64 = arith.constant 175264 : i64
  %c108960_i64 = arith.constant 108960 : i64
  %c108896_i64 = arith.constant 108896 : i64
  %c76128_i64 = arith.constant 76128 : i64
  %c72032_i64 = arith.constant 72032 : i64
  %c67936_i64 = arith.constant 67936 : i64
  %c51552_i64 = arith.constant 51552 : i64
  %c51456_i64 = arith.constant 51456 : i64
  %c51200_i64 = arith.constant 51200 : i64
  %c142496_i64 = arith.constant 142496 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1952_i64 = arith.constant 1952 : i64
  %c1824_i64 = arith.constant 1824 : i64
  %c1792_i64 = arith.constant 1792 : i64
  %c141984_i64 = arith.constant 141984 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c768_i64 = arith.constant 768 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c512_i32 = arith.constant 512 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c8_i32 = arith.constant 8 : i32
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<128xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c512_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : memref<64xi32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg9 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %13 = arith.select %12, %c3_i64, %c4_i64 : i64
    %14 = arith.index_cast %11 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %15 = arith.select %14, %c1_i64, %c2_i64 : i64
    %16 = arith.index_cast %11 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64_2 = arith.constant 1 : i64
    %17 = arith.select %16, %c0_i64_1, %c1_i64_2 : i64
    %18 = hivm.hir.pointer_cast(%c108960_i64, %c175264_i64) : memref<1x64xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<1x64xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c18432_i64, %c142496_i64) : memref<128x64xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c1280_i64, %c141984_i64) : memref<128x1xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<128x1xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %21 = arith.divsi %arg9, %4 : i32
    %22 = arith.remsi %21, %arg6 : i32
    %23 = arith.remsi %22, %c8_i32 : i32
    %24 = arith.divsi %22, %c8_i32 : i32
    %25 = arith.muli %23, %c64_i32 : i32
    %26 = hivm.hir.pointer_cast(%c768_i64) : memref<64xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%6, %25 : memref<64xi32, #hivm.address_space<ub>>, i32) outs(%26 : memref<64xi32, #hivm.address_space<ub>>)
    %27 = hivm.hir.pointer_cast(%c1024_i64) : memref<64xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmax ins(%26, %c512_i32 : memref<64xi32, #hivm.address_space<ub>>, i32) outs(%27 : memref<64xi32, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c1024_i64) : memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%27, %26 : memref<64xi32, #hivm.address_space<ub>>, memref<64xi32, #hivm.address_space<ub>>) outs(%28 : memref<64xi1, #hivm.address_space<ub>>)
    %29 = hivm.hir.pointer_cast(%c1024_i64) : memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%28 : memref<64xi1, #hivm.address_space<ub>>) outs(%29 : memref<64xi1, #hivm.address_space<ub>>)
    %30 = arith.muli %24, %c128_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [128, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %32 = affine.max affine_map<()[s0] -> (1024, s0)>()[%31]
    %33 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%32, %31]
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%33, %31]
    %35 = arith.cmpi slt, %34, %c128 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%34, 1] [1, 1] : memref<128x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %20[0, 0] [%34, 1] [1, 1] : memref<128x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.if %35 {
      %collapse_shape_23 = memref.collapse_shape %20 [[0, 1]] : memref<128x1xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_23 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %15]
    hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %collapse_shape_5 = memref.collapse_shape %20 [[0, 1]] : memref<128x1xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %36 = hivm.hir.pointer_cast(%c1792_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%collapse_shape_5, %5 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%36 : memref<128xi1, #hivm.address_space<ub>>)
    %37 = hivm.hir.pointer_cast(%c1792_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%36 : memref<128xi1, #hivm.address_space<ub>>) outs(%37 : memref<128xi1, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c1824_i64) : memref<64xf16, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c1952_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%29 : memref<64xi1, #hivm.address_space<ub>>) outs(%38 : memref<64xf16, #hivm.address_space<ub>>) temp_buffer(%39 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %expand_shape = memref.expand_shape %38 [[0, 1]] output_shape [1, 64] : memref<64xf16, #hivm.address_space<ub>> into memref<1x64xf16, #hivm.address_space<ub>>
    %40 = hivm.hir.pointer_cast(%c2048_i64) : memref<128x64xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape : memref<1x64xf16, #hivm.address_space<ub>>) outs(%40 : memref<128x64xf16, #hivm.address_space<ub>>) broadcast_dims = [0]
    %41 = hivm.hir.pointer_cast(%c2048_i64) : memref<128x64xi1, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %40 [[0, 1]] : memref<128x64xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %41 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_6, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_7 : memref<8192xi1, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c2048_i64) : memref<128x64xi1, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %42 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_7 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<8192xi1, #hivm.address_space<ub>>)
    %43 = arith.index_cast %25 : i32 to index
    %44 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%43, %31]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [128, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %45 = affine.max affine_map<()[s0] -> (512, s0)>()[%43]
    %46 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%45, %43]
    %47 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%46, %43]
    %48 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%33, %31]
    %49 = affine.max affine_map<()[s0] -> (0, s0)>()[%48]
    %50 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%46, %43]
    %51 = affine.max affine_map<()[s0] -> (0, s0)>()[%50]
    %52 = arith.cmpi slt, %49, %c128 : index
    %53 = arith.cmpi slt, %51, %c64 : index
    %54 = arith.ori %52, %53 : i1
    %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%49, %51] [1, 1] : memref<128x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %19[0, 0] [%49, %51] [1, 1] : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    scf.if %54 {
      %collapse_shape_23 = memref.collapse_shape %19 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_23 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%subview_10 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %55 = hivm.hir.pointer_cast(%c51200_i64) : memref<128xf16, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%c51456_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%37 : memref<128xi1, #hivm.address_space<ub>>) outs(%55 : memref<128xf16, #hivm.address_space<ub>>) temp_buffer(%56 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %expand_shape_12 = memref.expand_shape %55 [[0, 1]] output_shape [128, 1] : memref<128xf16, #hivm.address_space<ub>> into memref<128x1xf16, #hivm.address_space<ub>>
    %57 = hivm.hir.pointer_cast(%c51552_i64) : memref<128x64xf16, #hivm.address_space<ub>>
    %58 = hivm.hir.pointer_cast(%c67936_i64) : memref<2048xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_12 : memref<128x1xf16, #hivm.address_space<ub>>) outs(%57 : memref<128x64xf16, #hivm.address_space<ub>>) temp_buffer(%58 : memref<2048xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
    %59 = hivm.hir.pointer_cast(%c51552_i64) : memref<128x64xi1, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %57 [[0, 1]] : memref<128x64xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %59 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_13, %cst_0 : memref<8192xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_14 : memref<8192xi1, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c51552_i64) : memref<128x64xi1, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %60 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_14 : memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<8192xi1, #hivm.address_space<ub>>)
    %61 = hivm.hir.pointer_cast(%c51552_i64) : memref<128x64xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %61 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vand ins(%collapse_shape_15, %collapse_shape_8 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xi1, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8192xi1, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c72032_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%20, %19 : memref<128x1xf32, #hivm.address_space<ub>>, memref<128x64xf32, #hivm.address_space<ub>>) outs(%19 : memref<128x64xf32, #hivm.address_space<ub>>) temp_buffer(%62 : memref<1024xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %15]
    %63 = hivm.hir.pointer_cast(%c76128_i64) : memref<128x64xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %19 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %63 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %64 = hivm.hir.pointer_cast(%c108896_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_16, %collapse_shape_17, %cst : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_18 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%64 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    %65 = hivm.hir.pointer_cast(%c109216_i64) : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %17]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%63 : memref<128x64xf32, #hivm.address_space<ub>>) outs(%18 : memref<1x64xf32, #hivm.address_space<ub>>) temp_buffer(%65 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %collapse_shape_19 = memref.collapse_shape %18 [[0, 1]] : memref<1x64xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg5 to offset: [%43], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_21 = memref.subview %collapse_shape_19[0] [%47] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_22 = memref.subview %reinterpret_cast_20[0] [%47] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_22 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %17]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
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
