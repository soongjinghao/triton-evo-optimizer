// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_quantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c7104_i64 = arith.constant 7104 : i64
  %c6048_i64 = arith.constant 6048 : i64
  %c7072_i64 = arith.constant 7072 : i64
  %c6016_i64 = arith.constant 6016 : i64
  %c5984_i64 = arith.constant 5984 : i64
  %c5920_i64 = arith.constant 5920 : i64
  %c6560_i64 = arith.constant 6560 : i64
  %c5408_i64 = arith.constant 5408 : i64
  %c5376_i64 = arith.constant 5376 : i64
  %c4864_i64 = arith.constant 4864 : i64
  %c4352_i64 = arith.constant 4352 : i64
  %c4320_i64 = arith.constant 4320 : i64
  %c4288_i64 = arith.constant 4288 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c3680_i64 = arith.constant 3680 : i64
  %c3168_i64 = arith.constant 3168 : i64
  %c2656_i64 = arith.constant 2656 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2336_i64 = arith.constant 2336 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c1792_i64 = arith.constant 1792 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c6304_i64 = arith.constant 6304 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant -4.480000e+02 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 4.480000e+02 : f32
  %c128_i32 = arith.constant 128 : i32
  %c4_i32 = arith.constant 4 : i32
  %c128 = arith.constant 128 : index
  %cst_2 = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg13, %arg14 : i32
  %1 = arith.muli %0, %arg15 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg15, %arg14 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<128xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%6 : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg16 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg16 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c0_i64_3 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_3, %c1_i64 : i64
    %14 = arith.index_cast %11 : index to i1
    %c1_i64_4 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %15 = arith.select %14, %c1_i64_4, %c2_i64 : i64
    %16 = hivm.hir.pointer_cast(%c6048_i64, %c7104_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c6016_i64, %c7072_i64) : memref<1xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<1xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c5408_i64, %c6560_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c1024_i64, %c6304_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %20 = arith.divsi %arg16, %arg15 : i32
    %21 = arith.remsi %20, %arg14 : i32
    %22 = arith.divsi %arg16, %4 : i32
    %23 = arith.remsi %22, %arg13 : i32
    %24 = arith.cmpi slt, %21, %c4_i32 : i32
    scf.if %24 {
      %25 = arith.muli %21, %c128_i32 : i32
      %26 = arith.muli %23, %arg11 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %25 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%27, %28]
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%29], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %30 = affine.max affine_map<()[s0] -> (512, s0)>()[%28]
      %31 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%30, %28]
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%31, %28]
      %33 = arith.cmpi slt, %32, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%32] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %19[0] [%32] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %33 {
        hivm.hir.vbrc ins(%cst_2 : bf16) outs(%19 : memref<128xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %34 = hivm.hir.pointer_cast(%c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%19 : memref<128xbf16, #hivm.address_space<ub>>) outs(%34 : memref<128xf32, #hivm.address_space<ub>>)
      %35 = hivm.hir.pointer_cast(%c1792_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vabs ins(%34 : memref<128xf32, #hivm.address_space<ub>>) outs(%35 : memref<128xf32, #hivm.address_space<ub>>)
      %36 = hivm.hir.pointer_cast(%c2304_i64) : memref<1xf32, #hivm.address_space<ub>>
      %37 = hivm.hir.pointer_cast(%c2336_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <max> ins(%35 : memref<128xf32, #hivm.address_space<ub>>) outs(%36 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%37 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %38 = memref.load %36[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %39 = arith.divf %38, %cst_1 : f32
      %40 = arith.divf %cst_0, %39 : f32
      %41 = hivm.hir.pointer_cast(%c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%34, %40 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%41 : memref<128xf32, #hivm.address_space<ub>>)
      %42 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%41, %41 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%42 : memref<128xi1, #hivm.address_space<ub>>)
      %43 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%42 : memref<128xi1, #hivm.address_space<ub>>) outs(%43 : memref<128xi1, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.vcmp ins(%6, %6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%44 : memref<128xi1, #hivm.address_space<ub>>)
      %45 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%44 : memref<128xi1, #hivm.address_space<ub>>) outs(%45 : memref<128xi1, #hivm.address_space<ub>>)
      %46 = hivm.hir.pointer_cast(%c2656_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmin ins(%41, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%46 : memref<128xf32, #hivm.address_space<ub>>)
      %47 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xf32, #hivm.address_space<ub>>
      %48 = hivm.hir.pointer_cast(%c3680_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%43, %cst_1, %46 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%47 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%48 : memref<16xf32, #hivm.address_space<ub>>)
      %49 = hivm.hir.pointer_cast(%c3744_i64) : memref<128xf32, #hivm.address_space<ub>>
      %50 = hivm.hir.pointer_cast(%c4256_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%45, %41, %47 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%49 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<8xf32, #hivm.address_space<ub>>)
      %51 = hivm.hir.pointer_cast(%c4288_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.vcmp ins(%5, %5 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%51 : memref<128xi1, #hivm.address_space<ub>>)
      %52 = hivm.hir.pointer_cast(%c4288_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%51 : memref<128xi1, #hivm.address_space<ub>>) outs(%52 : memref<128xi1, #hivm.address_space<ub>>)
      %53 = hivm.hir.pointer_cast(%c4320_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.vcmp ins(%49, %49 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%53 : memref<128xi1, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c4320_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%53 : memref<128xi1, #hivm.address_space<ub>>) outs(%54 : memref<128xi1, #hivm.address_space<ub>>)
      %55 = hivm.hir.pointer_cast(%c4352_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmax ins(%49, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%55 : memref<128xf32, #hivm.address_space<ub>>)
      %56 = hivm.hir.pointer_cast(%c4864_i64) : memref<128xf32, #hivm.address_space<ub>>
      %57 = hivm.hir.pointer_cast(%c5376_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%52, %49, %55 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%56 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%57 : memref<8xf32, #hivm.address_space<ub>>)
      %58 = hivm.hir.pointer_cast(%c5408_i64, %c6560_i64) : memref<128xf32, #hivm.address_space<ub>>
      %59 = hivm.hir.pointer_cast(%c5920_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%54, %cst, %56 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%58 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%58 : memref<128xf32, #hivm.address_space<ub>>) outs(%18 : memref<128xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %60 = arith.muli %23, %arg8 : i32
      %61 = arith.index_cast %60 : i32 to index
      %62 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%61, %28]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%62], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %63 = arith.muli %23, %arg9 : i32
      %subview_7 = memref.subview %18[0] [%32] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      %subview_8 = memref.subview %reinterpret_cast_6[0] [%32] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_7 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
      %64 = hivm.hir.pointer_cast(%c5984_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %39, %64[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %65 = hivm.hir.pointer_cast(%c5984_i64) : memref<1xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%64 : memref<1xf32, #hivm.address_space<ub>>) outs(%65 : memref<1xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %66 = memref.load %65[%c0] : memref<1xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %13]
      memref.store %66, %17[%c0] : memref<1xbf16, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %67 = arith.index_cast %63 : i32 to index
      %68 = arith.index_cast %21 : i32 to index
      %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%67, %68]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%69], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%17 : memref<1xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %13]
    } else {
      %25 = arith.subi %21, %c4_i32 : i32
      %26 = arith.muli %25, %c128_i32 : i32
      %27 = arith.muli %23, %arg12 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %26 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%30], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.muli %23, %arg10 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%32, %29]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max affine_map<()[s0] -> (64, s0)>()[%29]
      %35 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%34, %29]
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %29]
      %37 = arith.cmpi slt, %36, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%36] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %16[0] [%36] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %37 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst_2 : bf16) outs(%16 : memref<128xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%36] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
