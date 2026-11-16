// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_silu_mul_fp8_quant_deep_gemm(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c8544_i64 = arith.constant 8544 : i64
  %c7488_i64 = arith.constant 7488 : i64
  %c7424_i64 = arith.constant 7424 : i64
  %c8032_i64 = arith.constant 8032 : i64
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
  %c7776_i64 = arith.constant 7776 : i64
  %c1824_i64 = arith.constant 1824 : i64
  %c1312_i64 = arith.constant 1312 : i64
  %c7520_i64 = arith.constant 7520 : i64
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
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %14 = arith.index_cast %arg10 : i32 to index
    %15 = arith.index_cast %3 : i32 to index
    %16 = arith.index_cast %1 : i32 to index
    %17 = arith.index_cast %c48_i32 : i32 to index
    %18 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%14, %15, %17]
    %19 = arith.index_cast %18 : index to i1
    %c0_i64_6 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %20 = arith.select %19, %c0_i64_6, %c1_i64 : i64
    %21 = hivm.hir.pointer_cast(%c7488_i64, %c8544_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c6912_i64, %c8032_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c1824_i64, %c7776_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c1056_i64, %c7520_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %25 = arith.divsi %arg10, %4 : i32
    %26 = arith.remsi %25, %arg7 : i32
    %27 = arith.remsi %26, %c8_i32 : i32
    %28 = arith.divsi %26, %c8_i32 : i32
    %29 = arith.index_cast %28 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%29], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %30 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %31 = arith.cmpi slt, %27, %30 : i32
    %32 = arith.muli %28, %c8_i32 : i32
    %33 = arith.addi %32, %27 : i32
    %34 = arith.muli %33, %c256_i32 : i32
    %35 = arith.index_cast %34 : i32 to index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %36 = arith.index_castui %31 : i1 to index
    %37 = affine.min affine_map<()[s0] -> (128, s0 * 128)>()[%36]
    %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
    %39 = arith.cmpi slt, %38, %c128 : index
    %subview = memref.subview %reinterpret_cast_7[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %24[0] [%38] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    scf.if %39 {
      hivm.hir.vbrc ins(%cst_4 : f16) outs(%24 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %40 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%24 : memref<128xf16, #hivm.address_space<ub>>) outs(%40 : memref<128xf32, #hivm.address_space<ub>>)
    %41 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%35]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %reinterpret_cast_9[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %23[0] [%38] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    scf.if %39 {
      hivm.hir.vbrc ins(%cst_4 : f16) outs(%23 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_10 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %42 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%23 : memref<128xf16, #hivm.address_space<ub>>) outs(%42 : memref<128xf32, #hivm.address_space<ub>>)
    %43 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%40, %cst_5 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%43 : memref<128xf32, #hivm.address_space<ub>>)
    %44 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%43, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%44 : memref<128xf32, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%44 : memref<128xf32, #hivm.address_space<ub>>) outs(%45 : memref<128xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%45, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%46 : memref<128xf32, #hivm.address_space<ub>>)
    %47 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%40, %46 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%47 : memref<128xf32, #hivm.address_space<ub>>)
    %48 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%47, %42 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%48 : memref<128xf32, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c3104_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vabs ins(%48 : memref<128xf32, #hivm.address_space<ub>>) outs(%49 : memref<128xf32, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c3616_i64) : memref<1xf32, #hivm.address_space<ub>>
    %51 = hivm.hir.pointer_cast(%c3648_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%49 : memref<128xf32, #hivm.address_space<ub>>) outs(%50 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %52 = memref.load %50[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c3904_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %52, %53[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %54 = hivm.hir.pointer_cast(%c3936_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%53, %53 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xi1, #hivm.address_space<ub>>)
    %55 = hivm.hir.pointer_cast(%c3936_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%54 : memref<1xi1, #hivm.address_space<ub>>) outs(%55 : memref<1xi1, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c3968_i64) : memref<1xf32, #hivm.address_space<ub>>
    %57 = arith.maximumf %52, %cst_0 : f32
    memref.store %57, %56[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %58 = hivm.hir.pointer_cast(%c4000_i64) : memref<1xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c4032_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%55, %cst_0, %56 : memref<1xi1, #hivm.address_space<ub>>, f32, memref<1xf32, #hivm.address_space<ub>>) outs(%58 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c4096_i64) : memref<1xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c4128_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%9, %53, %58 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%60 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %62 = memref.load %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %63 = arith.mulf %62, %cst : f32
    %64 = arith.divf %cst_2, %62 : f32
    %65 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%48, %64 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%65 : memref<128xf32, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c4160_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%65, %65 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%66 : memref<128xi1, #hivm.address_space<ub>>)
    %67 = hivm.hir.pointer_cast(%c4160_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%66 : memref<128xi1, #hivm.address_space<ub>>) outs(%67 : memref<128xi1, #hivm.address_space<ub>>)
    %68 = hivm.hir.pointer_cast(%c4192_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%65, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%68 : memref<128xf32, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c4704_i64) : memref<128xf32, #hivm.address_space<ub>>
    %70 = hivm.hir.pointer_cast(%c5216_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%67, %cst_0, %68 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%69 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<16xf32, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c5280_i64) : memref<128xf32, #hivm.address_space<ub>>
    %72 = hivm.hir.pointer_cast(%c5792_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%11, %65, %69 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%71 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%72 : memref<8xf32, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c5824_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%71, %71 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%73 : memref<128xi1, #hivm.address_space<ub>>)
    %74 = hivm.hir.pointer_cast(%c5824_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%73 : memref<128xi1, #hivm.address_space<ub>>) outs(%74 : memref<128xi1, #hivm.address_space<ub>>)
    %75 = hivm.hir.pointer_cast(%c5856_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%71, %cst_3 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%75 : memref<128xf32, #hivm.address_space<ub>>)
    %76 = hivm.hir.pointer_cast(%c6368_i64) : memref<128xf32, #hivm.address_space<ub>>
    %77 = hivm.hir.pointer_cast(%c6880_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%13, %71, %75 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%76 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%77 : memref<8xf32, #hivm.address_space<ub>>)
    %78 = hivm.hir.pointer_cast(%c6912_i64, %c8032_i64) : memref<128xf32, #hivm.address_space<ub>>
    %79 = hivm.hir.pointer_cast(%c7424_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %20]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%74, %cst_3, %76 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%78 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%79 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%78 : memref<128xf32, #hivm.address_space<ub>>) outs(%22 : memref<128xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %80 = arith.muli %33, %c128_i32 : i32
    %81 = arith.index_cast %80 : i32 to index
    %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%81], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_13 = memref.subview %22[0] [%38] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    %subview_14 = memref.subview %reinterpret_cast_12[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_13 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_14 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %20]
    scf.if %31 {
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %20]
      memref.store %63, %21[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %82 = arith.index_cast %32 : i32 to index
      %83 = arith.index_cast %27 : i32 to index
      %84 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%82, %83]
      %reinterpret_cast_15 = memref.reinterpret_cast %arg5 to offset: [%84], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%21 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %20]
    } {pipeline.veconly}
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
