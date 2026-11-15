// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_silu_mul_fp8_quant_deep_gemm(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c8576_i64 = arith.constant 8576 : i64
  %c7520_i64 = arith.constant 7520 : i64
  %c7456_i64 = arith.constant 7456 : i64
  %c8064_i64 = arith.constant 8064 : i64
  %c6944_i64 = arith.constant 6944 : i64
  %c6912_i64 = arith.constant 6912 : i64
  %c6400_i64 = arith.constant 6400 : i64
  %c5888_i64 = arith.constant 5888 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c5824_i64 = arith.constant 5824 : i64
  %c5312_i64 = arith.constant 5312 : i64
  %c5248_i64 = arith.constant 5248 : i64
  %c4736_i64 = arith.constant 4736 : i64
  %c4224_i64 = arith.constant 4224 : i64
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
  %c7808_i64 = arith.constant 7808 : i64
  %c1824_i64 = arith.constant 1824 : i64
  %c1312_i64 = arith.constant 1312 : i64
  %c7552_i64 = arith.constant 7552 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c544_i64 = arith.constant 544 : i64
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 1.000000e-10 : f32
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_2 = arith.constant 4.480000e+02 : f32
  %cst_3 = arith.constant -4.480000e+02 : f32
  %c1_i64 = arith.constant 1 : i64
  %cst_4 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg13, %arg14 : i32
  %1 = arith.muli %0, %arg15 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg15, %arg14 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst, %5[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %6 = hivm.hir.pointer_cast(%c32_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%6 : memref<128xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c544_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : memref<128xf32, #hivm.address_space<ub>>)
  %8 = arith.extsi %arg7 : i32 to i64
  %9 = arith.extsi %arg9 : i32 to i64
  %10 = arith.extsi %arg11 : i32 to i64
  %11 = arith.extsi %arg8 : i32 to i64
  %12 = arith.extsi %arg10 : i32 to i64
  %13 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vcmp ins(%5, %5 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%13 : memref<1xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%13 : memref<1xi1, #hivm.address_space<ub>>) outs(%14 : memref<1xi1, #hivm.address_space<ub>>)
  %15 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%7, %7 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%15 : memref<128xi1, #hivm.address_space<ub>>)
  %16 = hivm.hir.pointer_cast(%c544_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%15 : memref<128xi1, #hivm.address_space<ub>>) outs(%16 : memref<128xi1, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%6, %6 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%17 : memref<128xi1, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c32_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%17 : memref<128xi1, #hivm.address_space<ub>>) outs(%18 : memref<128xi1, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg16 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg16, %4 : i32
    %20 = arith.remsi %19, %arg13 : i32
    %21 = arith.extsi %20 : i32 to i64
    %22 = arith.index_cast %20 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = arith.extsi %23 : i32 to i64
    %25 = arith.muli %21, %8 : i64
    %26 = arith.addi %25, %c128_i64 : i64
    %27 = arith.muli %21, %9 : i64
    %28 = arith.muli %21, %10 : i64
    %29 = arith.index_cast %25 : i64 to index
    %30 = arith.index_cast %26 : i64 to index
    %31 = arith.index_cast %27 : i64 to index
    scf.for %arg17 = %c0_i64 to %24 step %c1_i64  : i64 {
      %32 = arith.index_cast %arg17 : i64 to index
      %33 = arith.index_cast %c0_i64 : i64 to index
      %34 = arith.index_cast %24 : i64 to index
      %35 = arith.index_cast %c1_i64 : i64 to index
      %36 = arith.index_cast %arg16 : i32 to index
      %37 = arith.index_cast %3 : i32 to index
      %38 = arith.index_cast %1 : i32 to index
      %39 = arith.index_cast %c48_i32 : i32 to index
      %40 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%32, %33, %34, %35, %36, %37, %39]
      %41 = arith.index_cast %40 : index to i1
      %c0_i64_5 = arith.constant 0 : i64
      %c1_i64_6 = arith.constant 1 : i64
      %42 = arith.select %41, %c0_i64_5, %c1_i64_6 : i64
      %43 = hivm.hir.pointer_cast(%c7520_i64, %c8576_i64) : memref<1xf32, #hivm.address_space<ub>>
      annotation.mark %43 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
      %44 = hivm.hir.pointer_cast(%c6944_i64, %c8064_i64) : memref<128xf16, #hivm.address_space<ub>>
      annotation.mark %44 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
      %45 = hivm.hir.pointer_cast(%c1824_i64, %c7808_i64) : memref<128xf16, #hivm.address_space<ub>>
      annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
      %46 = hivm.hir.pointer_cast(%c1056_i64, %c7552_i64) : memref<128xf16, #hivm.address_space<ub>>
      annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
      %47 = arith.muli %arg17, %11 : i64
      %48 = arith.index_cast %47 : i64 to index
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %48]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%reinterpret_cast_7 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%46 : memref<128xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %50 = hivm.hir.pointer_cast(%c1312_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%46 : memref<128xf16, #hivm.address_space<ub>>) outs(%50 : memref<128xf32, #hivm.address_space<ub>>)
      %51 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %48]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%reinterpret_cast_8 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%45 : memref<128xf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %52 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%50, %cst_4 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%52 : memref<128xf32, #hivm.address_space<ub>>)
      %53 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%52, %cst_0 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%53 : memref<128xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vexp ins(%53 : memref<128xf32, #hivm.address_space<ub>>) outs(%54 : memref<128xf32, #hivm.address_space<ub>>)
      %55 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%54, %cst_1 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%55 : memref<128xf32, #hivm.address_space<ub>>)
      %56 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vdiv ins(%50, %55 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%56 : memref<128xf32, #hivm.address_space<ub>>)
      %57 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%45 : memref<128xf16, #hivm.address_space<ub>>) outs(%57 : memref<128xf32, #hivm.address_space<ub>>)
      %58 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%56, %57 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%58 : memref<128xf32, #hivm.address_space<ub>>)
      %59 = hivm.hir.pointer_cast(%c3104_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vabs ins(%58 : memref<128xf32, #hivm.address_space<ub>>) outs(%59 : memref<128xf32, #hivm.address_space<ub>>)
      %60 = hivm.hir.pointer_cast(%c3616_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = hivm.hir.pointer_cast(%c3648_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <max> ins(%59 : memref<128xf32, #hivm.address_space<ub>>) outs(%60 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %62 = memref.load %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %63 = hivm.hir.pointer_cast(%c3904_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %62, %63[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %64 = hivm.hir.pointer_cast(%c3936_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcmp ins(%63, %63 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%64 : memref<1xi1, #hivm.address_space<ub>>)
      %65 = hivm.hir.pointer_cast(%c3936_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%64 : memref<1xi1, #hivm.address_space<ub>>) outs(%65 : memref<1xi1, #hivm.address_space<ub>>)
      %66 = hivm.hir.pointer_cast(%c3968_i64) : memref<1xf32, #hivm.address_space<ub>>
      %67 = arith.maximumf %62, %cst : f32
      memref.store %67, %66[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %68 = hivm.hir.pointer_cast(%c4000_i64) : memref<1xf32, #hivm.address_space<ub>>
      %69 = hivm.hir.pointer_cast(%c4032_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%65, %cst, %66 : memref<1xi1, #hivm.address_space<ub>>, f32, memref<1xf32, #hivm.address_space<ub>>) outs(%68 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<16xf32, #hivm.address_space<ub>>)
      %70 = hivm.hir.pointer_cast(%c4096_i64) : memref<1xf32, #hivm.address_space<ub>>
      %71 = hivm.hir.pointer_cast(%c4128_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%14, %63, %68 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%70 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%71 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %72 = memref.load %70[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %73 = arith.divf %72, %cst_2 : f32
      %74 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xf32, #hivm.address_space<ub>>
      %75 = hivm.hir.pointer_cast(%c4160_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.vdiv ins(%58, %73 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%74 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%75 : memref<8xf32, #hivm.address_space<ub>>)
      %76 = hivm.hir.pointer_cast(%c4192_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%74, %74 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%76 : memref<128xi1, #hivm.address_space<ub>>)
      %77 = hivm.hir.pointer_cast(%c4192_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%76 : memref<128xi1, #hivm.address_space<ub>>) outs(%77 : memref<128xi1, #hivm.address_space<ub>>)
      %78 = hivm.hir.pointer_cast(%c4224_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmin ins(%74, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%78 : memref<128xf32, #hivm.address_space<ub>>)
      %79 = hivm.hir.pointer_cast(%c4736_i64) : memref<128xf32, #hivm.address_space<ub>>
      %80 = hivm.hir.pointer_cast(%c5248_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%77, %cst_2, %78 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%79 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%80 : memref<16xf32, #hivm.address_space<ub>>)
      %81 = hivm.hir.pointer_cast(%c5312_i64) : memref<128xf32, #hivm.address_space<ub>>
      %82 = hivm.hir.pointer_cast(%c5824_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%16, %74, %79 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%81 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%82 : memref<8xf32, #hivm.address_space<ub>>)
      %83 = hivm.hir.pointer_cast(%c5856_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%81, %81 : memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%83 : memref<128xi1, #hivm.address_space<ub>>)
      %84 = hivm.hir.pointer_cast(%c5856_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%83 : memref<128xi1, #hivm.address_space<ub>>) outs(%84 : memref<128xi1, #hivm.address_space<ub>>)
      %85 = hivm.hir.pointer_cast(%c5888_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vmax ins(%81, %cst_3 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%85 : memref<128xf32, #hivm.address_space<ub>>)
      %86 = hivm.hir.pointer_cast(%c6400_i64) : memref<128xf32, #hivm.address_space<ub>>
      %87 = hivm.hir.pointer_cast(%c6912_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%18, %81, %85 : memref<128xi1, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>, memref<128xf32, #hivm.address_space<ub>>) outs(%86 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%87 : memref<8xf32, #hivm.address_space<ub>>)
      %88 = hivm.hir.pointer_cast(%c6944_i64, %c8064_i64) : memref<128xf32, #hivm.address_space<ub>>
      %89 = hivm.hir.pointer_cast(%c7456_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %42]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%84, %cst_3, %86 : memref<128xi1, #hivm.address_space<ub>>, f32, memref<128xf32, #hivm.address_space<ub>>) outs(%88 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%89 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%88 : memref<128xf32, #hivm.address_space<ub>>) outs(%44 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %90 = arith.muli %arg17, %12 : i64
      %91 = arith.index_cast %90 : i64 to index
      %92 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%31, %91]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%92], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%44 : memref<128xf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %42]
      %93 = arith.addi %28, %arg17 : i64
      %94 = arith.index_cast %93 : i64 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %42]
      memref.store %73, %43[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%94], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%43 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %42]
    } {pipeline.veconly, tt.num_stages = 4 : i32}
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
