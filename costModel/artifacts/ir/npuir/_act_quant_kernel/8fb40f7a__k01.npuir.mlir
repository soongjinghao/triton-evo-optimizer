// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c7136_i64 = arith.constant 7136 : i64
  %c6080_i64 = arith.constant 6080 : i64
  %c6048_i64 = arith.constant 6048 : i64
  %c6624_i64 = arith.constant 6624 : i64
  %c5536_i64 = arith.constant 5536 : i64
  %c5472_i64 = arith.constant 5472 : i64
  %c4960_i64 = arith.constant 4960 : i64
  %c4448_i64 = arith.constant 4448 : i64
  %c4416_i64 = arith.constant 4416 : i64
  %c4384_i64 = arith.constant 4384 : i64
  %c3872_i64 = arith.constant 3872 : i64
  %c3808_i64 = arith.constant 3808 : i64
  %c3296_i64 = arith.constant 3296 : i64
  %c2784_i64 = arith.constant 2784 : i64
  %c2752_i64 = arith.constant 2752 : i64
  %c2720_i64 = arith.constant 2720 : i64
  %c2688_i64 = arith.constant 2688 : i64
  %c2656_i64 = arith.constant 2656 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2528_i64 = arith.constant 2528 : i64
  %c2496_i64 = arith.constant 2496 : i64
  %c2464_i64 = arith.constant 2464 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c2400_i64 = arith.constant 2400 : i64
  %c2144_i64 = arith.constant 2144 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c6112_i64 = arith.constant 6112 : i64
  %c1088_i64 = arith.constant 1088 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c544_i64 = arith.constant 544 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.00223214296 : f32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 9.99999974E-5 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c128_i32 = arith.constant 128 : i32
  %cst_2 = arith.constant -4.480000e+02 : f32
  %cst_3 = arith.constant 4.480000e+02 : f32
  %c128 = arith.constant 128 : index
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
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst_0, %5[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %6 = hivm.hir.pointer_cast(%c32_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%6 : memref<128xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c544_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%7 : memref<128xf32, #hivm.address_space<ub>>)
  %8 = arith.index_cast %arg7 : i32 to index
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vcmp ins(%5, %5 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%9 : memref<1xi1, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%9 : memref<1xi1, #hivm.address_space<ub>>) outs(%10 : memref<1xi1, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c1056_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst_5, %11[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %12 = hivm.hir.pointer_cast(%c1056_i64) : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vln ins(%11 : memref<1xf32, #hivm.address_space<ub>>) outs(%12 : memref<1xf32, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%6, %6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%13 : memref<128xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%13 : memref<128xi1, #hivm.address_space<ub>>) outs(%14 : memref<128xi1, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%7, %7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%15 : memref<128xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%15 : memref<128xi1, #hivm.address_space<ub>>) outs(%16 : memref<128xi1, #hivm.address_space<ub>>)
  %17 = arith.divsi %arg7, %c128_i32 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
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
    %c0_i64_7 = arith.constant 0 : i64
    %c1_i64_8 = arith.constant 1 : i64
    %26 = arith.select %25, %c0_i64_7, %c1_i64_8 : i64
    %27 = hivm.hir.pointer_cast(%c6080_i64, %c7136_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c5536_i64, %c6624_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c1088_i64, %c6112_i64) : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %30 = arith.divsi %arg11, %arg10 : i32
    %31 = arith.remsi %30, %arg9 : i32
    %32 = arith.divsi %arg11, %4 : i32
    %33 = arith.remsi %32, %arg8 : i32
    %34 = arith.muli %31, %c128_i32 : i32
    %35 = arith.muli %33, %arg7 : i32
    %36 = arith.index_cast %35 : i32 to index
    %37 = arith.index_cast %34 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%36, %37]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%37, %8]
    %40 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%39, %37]
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %37]
    %42 = arith.cmpi slt, %41, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%41] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %29[0] [%41] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %42 {
      hivm.hir.vbrc ins(%cst_4 : f32) outs(%29 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %24]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %43 = hivm.hir.pointer_cast(%c1600_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%29 : memref<128xf32, #hivm.address_space<ub>>) outs(%43 : memref<128xf32, #hivm.address_space<ub>>)
    %44 = hivm.hir.pointer_cast(%c2112_i64) : memref<1xf32, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c2144_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%43 : memref<128xf32, #hivm.address_space<ub>>) outs(%44 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%45 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %46 = memref.load %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c2400_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %46, %47[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %48 = hivm.hir.pointer_cast(%c2432_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%47, %47 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%48 : memref<1xi1, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c2432_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%48 : memref<1xi1, #hivm.address_space<ub>>) outs(%49 : memref<1xi1, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c2464_i64) : memref<1xf32, #hivm.address_space<ub>>
    %51 = arith.maximumf %46, %cst_0 : f32
    memref.store %51, %50[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %52 = hivm.hir.pointer_cast(%c2496_i64) : memref<1xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c2528_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%49, %cst_0, %50 : memref<1xi1, #hivm.address_space<ub>>, f32, memref<1xf32, #hivm.address_space<ub>>) outs(%52 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<16xf32, #hivm.address_space<ub>>)
    %54 = hivm.hir.pointer_cast(%c2592_i64) : memref<1xf32, #hivm.address_space<ub>>
    %55 = hivm.hir.pointer_cast(%c2624_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%10, %47, %52 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %57 = arith.mulf %56, %cst : f32
    %58 = hivm.hir.pointer_cast(%c2656_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %57, %58[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %59 = hivm.hir.pointer_cast(%c2656_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%58 : memref<1xf32, #hivm.address_space<ub>>) outs(%59 : memref<1xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c2656_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%59, %12 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%60 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %61 = memref.load %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c2688_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %61, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %63 = hivm.hir.pointer_cast(%c2688_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%62 : memref<1xf32, #hivm.address_space<ub>>) outs(%63 : memref<1xf32, #hivm.address_space<ub>>) round_mode = <ceil>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %64 = memref.load %63[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %65 = hivm.hir.pointer_cast(%c2720_i64) : memref<1xf32, #hivm.address_space<ub>>
    %66 = arith.mulf %64, %cst_6 : f32
    memref.store %66, %65[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %67 = hivm.hir.pointer_cast(%c2720_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%65 : memref<1xf32, #hivm.address_space<ub>>) outs(%67 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %68 = memref.load %67[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %69 = arith.divf %cst_1, %68 : f32
    hivm.hir.vmul ins(%29, %69 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%29 : memref<128xf32, #hivm.address_space<ub>>)
    %70 = hivm.hir.pointer_cast(%c2752_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%29, %29 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%70 : memref<128xi1, #hivm.address_space<ub>>)
    %71 = hivm.hir.pointer_cast(%c2752_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%70 : memref<128xi1, #hivm.address_space<ub>>) outs(%71 : memref<128xi1, #hivm.address_space<ub>>)
    %72 = hivm.hir.pointer_cast(%c2784_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%29, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%72 : memref<128xf32, #hivm.address_space<ub>>)
    %73 = hivm.hir.pointer_cast(%c3296_i64) : memref<128xf32, #hivm.address_space<ub>>
    %74 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%71, %cst_2, %72 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%73 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%74 : memref<16xf32, #hivm.address_space<ub>>)
    %75 = hivm.hir.pointer_cast(%c3872_i64) : memref<128xf32, #hivm.address_space<ub>>
    %76 = hivm.hir.pointer_cast(%c4384_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%14, %29, %73 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%75 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%76 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %24]
    %77 = hivm.hir.pointer_cast(%c4416_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%75, %75 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%77 : memref<128xi1, #hivm.address_space<ub>>)
    %78 = hivm.hir.pointer_cast(%c4416_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%77 : memref<128xi1, #hivm.address_space<ub>>) outs(%78 : memref<128xi1, #hivm.address_space<ub>>)
    %79 = hivm.hir.pointer_cast(%c4448_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%75, %cst_3 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%79 : memref<128xf32, #hivm.address_space<ub>>)
    %80 = hivm.hir.pointer_cast(%c4960_i64) : memref<128xf32, #hivm.address_space<ub>>
    %81 = hivm.hir.pointer_cast(%c5472_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%78, %cst_3, %79 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%80 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%81 : memref<16xf32, #hivm.address_space<ub>>)
    %82 = hivm.hir.pointer_cast(%c5536_i64, %c6624_i64) : memref<128xf32, #hivm.address_space<ub>>
    %83 = hivm.hir.pointer_cast(%c6048_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %26]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%16, %75, %80 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%82 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%83 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%82 : memref<128xf32, #hivm.address_space<ub>>) outs(%28 : memref<128xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_11 = memref.subview %28[0] [%41] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    %subview_12 = memref.subview %reinterpret_cast_10[0] [%41] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_11 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %26]
    %84 = arith.muli %33, %17 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %26]
    memref.store %68, %27[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %85 = arith.index_cast %84 : i32 to index
    %86 = arith.index_cast %31 : i32 to index
    %87 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%85, %86]
    %reinterpret_cast_13 = memref.reinterpret_cast %arg5 to offset: [%87], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%27 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_13 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %26]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
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
