// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c7008_i64 = arith.constant 7008 : i64
  %c5952_i64 = arith.constant 5952 : i64
  %c5920_i64 = arith.constant 5920 : i64
  %c6496_i64 = arith.constant 6496 : i64
  %c5408_i64 = arith.constant 5408 : i64
  %c5344_i64 = arith.constant 5344 : i64
  %c4832_i64 = arith.constant 4832 : i64
  %c4320_i64 = arith.constant 4320 : i64
  %c4288_i64 = arith.constant 4288 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c3744_i64 = arith.constant 3744 : i64
  %c3680_i64 = arith.constant 3680 : i64
  %c3168_i64 = arith.constant 3168 : i64
  %c2656_i64 = arith.constant 2656 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c2496_i64 = arith.constant 2496 : i64
  %c2464_i64 = arith.constant 2464 : i64
  %c2432_i64 = arith.constant 2432 : i64
  %c2400_i64 = arith.constant 2400 : i64
  %c2368_i64 = arith.constant 2368 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c1568_i64 = arith.constant 1568 : i64
  %c5984_i64 = arith.constant 5984 : i64
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
  %11 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%6, %6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%11 : memref<128xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%11 : memref<128xi1, #hivm.address_space<ub>>) outs(%12 : memref<128xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%7, %7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%13 : memref<128xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%13 : memref<128xi1, #hivm.address_space<ub>>) outs(%14 : memref<128xi1, #hivm.address_space<ub>>)
  %15 = arith.divsi %arg7, %c128_i32 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %16 = arith.index_cast %arg11 : i32 to index
    %17 = arith.index_cast %3 : i32 to index
    %18 = arith.index_cast %1 : i32 to index
    %19 = arith.index_cast %c48_i32 : i32 to index
    %20 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%16, %17, %19]
    %21 = arith.index_cast %20 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %22 = arith.select %21, %c1_i64, %c2_i64 : i64
    %23 = arith.index_cast %20 : index to i1
    %c0_i64_5 = arith.constant 0 : i64
    %c1_i64_6 = arith.constant 1 : i64
    %24 = arith.select %23, %c0_i64_5, %c1_i64_6 : i64
    %25 = hivm.hir.pointer_cast(%c5952_i64, %c7008_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c5408_i64, %c6496_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c1056_i64, %c5984_i64) : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %28 = arith.divsi %arg11, %arg10 : i32
    %29 = arith.remsi %28, %arg9 : i32
    %30 = arith.divsi %arg11, %4 : i32
    %31 = arith.remsi %30, %arg8 : i32
    %32 = arith.muli %29, %c128_i32 : i32
    %33 = arith.muli %31, %arg7 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = arith.index_cast %32 : i32 to index
    %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%34, %35]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %37 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%35, %8]
    %38 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%37, %35]
    %39 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %35]
    %40 = arith.cmpi slt, %39, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%39] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %27[0] [%39] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %40 {
      hivm.hir.vbrc ins(%cst_4 : f32) outs(%27 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %22]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_4 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %41 = hivm.hir.pointer_cast(%c1568_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%27 : memref<128xf32, #hivm.address_space<ub>>) outs(%41 : memref<128xf32, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c2080_i64) : memref<1xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c2112_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%41 : memref<128xf32, #hivm.address_space<ub>>) outs(%42 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%43 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %44 = memref.load %42[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c2368_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %44, %45[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %46 = hivm.hir.pointer_cast(%c2400_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%45, %45 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%46 : memref<1xi1, #hivm.address_space<ub>>)
    %47 = hivm.hir.pointer_cast(%c2400_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%46 : memref<1xi1, #hivm.address_space<ub>>) outs(%47 : memref<1xi1, #hivm.address_space<ub>>)
    %48 = hivm.hir.pointer_cast(%c2432_i64) : memref<1xf32, #hivm.address_space<ub>>
    %49 = arith.maximumf %44, %cst_0 : f32
    memref.store %49, %48[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %50 = hivm.hir.pointer_cast(%c2464_i64) : memref<1xf32, #hivm.address_space<ub>>
    %51 = hivm.hir.pointer_cast(%c2496_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%47, %cst_0, %48 : memref<1xi1, #hivm.address_space<ub>>, f32, memref<1xf32, #hivm.address_space<ub>>) outs(%50 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<16xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c2560_i64) : memref<1xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c2592_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%10, %45, %50 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%52 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %54 = memref.load %52[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %55 = arith.mulf %54, %cst : f32
    %56 = arith.divf %cst_1, %55 : f32
    hivm.hir.vmul ins(%27, %56 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%27 : memref<128xf32, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%27, %27 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%57 : memref<128xi1, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%57 : memref<128xi1, #hivm.address_space<ub>>) outs(%58 : memref<128xi1, #hivm.address_space<ub>>)
    %59 = hivm.hir.pointer_cast(%c2656_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%27, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%59 : memref<128xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c3680_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%58, %cst_2, %59 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%60 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<16xf32, #hivm.address_space<ub>>)
    %62 = hivm.hir.pointer_cast(%c3744_i64) : memref<128xf32, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c4256_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%12, %27, %60 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%62 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%63 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %22]
    %64 = hivm.hir.pointer_cast(%c4288_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%62, %62 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%64 : memref<128xi1, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c4288_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%64 : memref<128xi1, #hivm.address_space<ub>>) outs(%65 : memref<128xi1, #hivm.address_space<ub>>)
    %66 = hivm.hir.pointer_cast(%c4320_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%62, %cst_3 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%66 : memref<128xf32, #hivm.address_space<ub>>)
    %67 = hivm.hir.pointer_cast(%c4832_i64) : memref<128xf32, #hivm.address_space<ub>>
    %68 = hivm.hir.pointer_cast(%c5344_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%65, %cst_3, %66 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%67 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%68 : memref<16xf32, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c5408_i64, %c6496_i64) : memref<128xf32, #hivm.address_space<ub>>
    %70 = hivm.hir.pointer_cast(%c5920_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %24]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%14, %62, %67 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%69 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%70 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%69 : memref<128xf32, #hivm.address_space<ub>>) outs(%26 : memref<128xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_9 = memref.subview %26[0] [%39] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    %subview_10 = memref.subview %reinterpret_cast_8[0] [%39] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_9 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %24]
    %71 = arith.muli %31, %15 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %24]
    memref.store %55, %25[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %72 = arith.index_cast %71 : i32 to index
    %73 = arith.index_cast %29 : i32 to index
    %74 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%72, %73]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg5 to offset: [%74], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%25 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %24]
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
