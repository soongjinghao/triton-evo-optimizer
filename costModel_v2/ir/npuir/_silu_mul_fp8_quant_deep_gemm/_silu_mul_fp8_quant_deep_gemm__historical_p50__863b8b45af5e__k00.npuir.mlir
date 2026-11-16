// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_silu_mul_fp8_quant_deep_gemm(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c7488_i64 = arith.constant 7488 : i64
  %c7424_i64 = arith.constant 7424 : i64
  %c6912_i64 = arith.constant 6912 : i64
  %c6880_i64 = arith.constant 6880 : i64
  %c6368_i64 = arith.constant 6368 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c5824_i64 = arith.constant 5824 : i64
  %c5792_i64 = arith.constant 5792 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c5216_i64 = arith.constant 5216 : i64
  %c4704_i64 = arith.constant 4704 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c4128_i64 = arith.constant 4128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c4032_i64 = arith.constant 4032 : i64
  %c4000_i64 = arith.constant 4000 : i64
  %c3968_i64 = arith.constant 3968 : i64
  %c3936_i64 = arith.constant 3936 : i64
  %c3904_i64 = arith.constant 3904 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c3616_i64 = arith.constant 3616 : i64
  %c3104_i64 = arith.constant 3104 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c1824_i64 = arith.constant 1824 : i64
  %c1312_i64 = arith.constant 1312 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c544_i64 = arith.constant 544 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.00223214296 : f32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c256_i32 = arith.constant 256 : i32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_2 = arith.constant 4.480000e+02 : f32
  %cst_3 = arith.constant -4.480000e+02 : f32
  %c128 = arith.constant 128 : index
  %cst_4 = arith.constant 0.000000e+00 : f16
  %cst_5 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst_0, %5[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %6 = hivm.hir.pointer_cast(%c32_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%6 : memref<128xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c544_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%7 : memref<128xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vcmp ins(%5, %5 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%8 : memref<1xi1, #hivm.address_space<ub>>)
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%8 : memref<1xi1, #hivm.address_space<ub>>) outs(%9 : memref<1xi1, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%6, %6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%10 : memref<128xi1, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%10 : memref<128xi1, #hivm.address_space<ub>>) outs(%11 : memref<128xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%7, %7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%12 : memref<128xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%12 : memref<128xi1, #hivm.address_space<ub>>) outs(%13 : memref<128xi1, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg10, %4 : i32
    %15 = arith.remsi %14, %arg7 : i32
    %16 = arith.remsi %15, %c8_i32 : i32
    %17 = arith.divsi %15, %c8_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = arith.cmpi slt, %16, %19 : i32
    %21 = arith.muli %17, %c8_i32 : i32
    %22 = arith.addi %21, %16 : i32
    %23 = arith.muli %22, %c256_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = hivm.hir.pointer_cast(%c1056_i64) : memref<128xf16, #hivm.address_space<ub>>
    %26 = arith.index_castui %20 : i1 to index
    %27 = affine.min affine_map<()[s0] -> (128, s0 * 128)>()[%26]
    %28 = affine.max affine_map<()[s0] -> (0, s0)>()[%27]
    %29 = arith.cmpi slt, %28, %c128 : index
    %subview = memref.subview %reinterpret_cast_6[0] [%28] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %25[0] [%28] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    scf.if %29 {
      hivm.hir.vbrc ins(%cst_4 : f16) outs(%25 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %30 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%25 : memref<128xf16, #hivm.address_space<ub>>) outs(%30 : memref<128xf32, #hivm.address_space<ub>>)
    %31 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%24]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %32 = hivm.hir.pointer_cast(%c1824_i64) : memref<128xf16, #hivm.address_space<ub>>
    %subview_9 = memref.subview %reinterpret_cast_8[0] [%28] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %32[0] [%28] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    scf.if %29 {
      hivm.hir.vbrc ins(%cst_4 : f16) outs(%32 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %33 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%32 : memref<128xf16, #hivm.address_space<ub>>) outs(%33 : memref<128xf32, #hivm.address_space<ub>>)
    %34 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%30, %cst_5 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%34 : memref<128xf32, #hivm.address_space<ub>>)
    %35 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%34, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%35 : memref<128xf32, #hivm.address_space<ub>>)
    %36 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%35 : memref<128xf32, #hivm.address_space<ub>>) outs(%36 : memref<128xf32, #hivm.address_space<ub>>)
    %37 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%36, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<128xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%30, %37 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%38 : memref<128xf32, #hivm.address_space<ub>>)
    %39 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%38, %33 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%39 : memref<128xf32, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c3104_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vabs ins(%39 : memref<128xf32, #hivm.address_space<ub>>) outs(%40 : memref<128xf32, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c3616_i64) : memref<1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c3648_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%40 : memref<128xf32, #hivm.address_space<ub>>) outs(%41 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %43 = memref.load %41[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c3904_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %43, %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %45 = hivm.hir.pointer_cast(%c3936_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%44, %44 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%45 : memref<1xi1, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c3936_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%45 : memref<1xi1, #hivm.address_space<ub>>) outs(%46 : memref<1xi1, #hivm.address_space<ub>>)
    %47 = hivm.hir.pointer_cast(%c3968_i64) : memref<1xf32, #hivm.address_space<ub>>
    %48 = arith.maximumf %43, %cst_0 : f32
    memref.store %48, %47[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %49 = hivm.hir.pointer_cast(%c4000_i64) : memref<1xf32, #hivm.address_space<ub>>
    %50 = hivm.hir.pointer_cast(%c4032_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%46, %cst_0, %47 : memref<1xi1, #hivm.address_space<ub>>, f32, memref<1xf32, #hivm.address_space<ub>>) outs(%49 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<16xf32, #hivm.address_space<ub>>)
    %51 = hivm.hir.pointer_cast(%c4096_i64) : memref<1xf32, #hivm.address_space<ub>>
    %52 = hivm.hir.pointer_cast(%c4128_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%9, %44, %49 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%51 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%52 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %53 = memref.load %51[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %54 = arith.mulf %53, %cst : f32
    %55 = arith.divf %cst_2, %53 : f32
    %56 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%39, %55 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%56 : memref<128xf32, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c4160_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%56, %56 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%57 : memref<128xi1, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c4160_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%57 : memref<128xi1, #hivm.address_space<ub>>) outs(%58 : memref<128xi1, #hivm.address_space<ub>>)
    %59 = hivm.hir.pointer_cast(%c4192_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%56, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%59 : memref<128xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c4704_i64) : memref<128xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c5216_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%58, %cst_0, %59 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%60 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<16xf32, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c5280_i64) : memref<128xf32, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c5792_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%11, %56, %60 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%62 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%63 : memref<8xf32, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c5824_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%62, %62 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%64 : memref<128xi1, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c5824_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%64 : memref<128xi1, #hivm.address_space<ub>>) outs(%65 : memref<128xi1, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c5856_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%62, %cst_3 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%66 : memref<128xf32, #hivm.address_space<ub>>)
    %67 = hivm.hir.pointer_cast(%c6368_i64) : memref<128xf32, #hivm.address_space<ub>>
    %68 = hivm.hir.pointer_cast(%c6880_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%13, %62, %66 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%67 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%68 : memref<8xf32, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c6912_i64) : memref<128xf32, #hivm.address_space<ub>>
    %70 = hivm.hir.pointer_cast(%c7424_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%65, %cst_3, %67 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%69 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<16xf32, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c6912_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%69 : memref<128xf32, #hivm.address_space<ub>>) outs(%71 : memref<128xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %72 = arith.muli %22, %c128_i32 : i32
    %73 = arith.index_cast %72 : i32 to index
    %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%73], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_12 = memref.subview %71[0] [%28] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    %subview_13 = memref.subview %reinterpret_cast_11[0] [%28] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_12 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_13 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    scf.if %20 {
      %74 = hivm.hir.pointer_cast(%c7488_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      memref.store %54, %74[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %75 = arith.index_cast %21 : i32 to index
      %76 = arith.index_cast %16 : i32 to index
      %77 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%75, %76]
      %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%77], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%74 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_14 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
