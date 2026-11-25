#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 + 64)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 + 128)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 + 256)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1 + 512)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1 + 1024)>
#map7 = affine_map<()[s0, s1] -> (s0 + s1 + 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fused_qkvzba_split_reshape_cat_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fused_qkvzba_split_reshape_cat_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c8_i32 = arith.constant 8 : i32
    %c384_i32 = arith.constant 384 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3072_i32 = arith.constant 3072 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1600_i64 = arith.constant 1600 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1856_i64 = arith.constant 1856 : i64
    %c256_i64 = arith.constant 256 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c512_i64 = arith.constant 512 : i64
    %c2624_i64 = arith.constant 2624 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c3136_i64 = arith.constant 3136 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %c1568_i64 = arith.constant 1568 : i64
    %c3168_i64 = arith.constant 3168 : i64
    %0 = hivm.hir.pointer_cast(%c1568_i64) : memref<2xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c3168_i64) : memref<2xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1536_i64) : memref<2xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c3136_i64) : memref<2xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c2112_i64) : memref<128xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c256_i64) : memref<64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1856_i64) : memref<64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c1600_i64) : memref<64xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg9, %arg10 : i32
    %13 = arith.muli %12, %arg11 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg11, %arg10 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
    scf.for %arg12 = %15 to %13 step %c48_i32  : i32 {
      %17 = arith.index_cast %arg12 : i32 to index
      %18 = arith.index_cast %15 : i32 to index
      %19 = affine.apply #map()[%17, %18]
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      %22 = arith.select %21, %1, %0 : memref<2xf32, #hivm.address_space<ub>>
      %23 = arith.index_cast %arg12 : i32 to index
      %24 = arith.index_cast %15 : i32 to index
      %25 = affine.apply #map()[%23, %24]
      %26 = arith.index_cast %25 : index to i64
      %27 = arith.cmpi eq, %26, %c1_i64 : i64
      %28 = arith.select %27, %3, %2 : memref<2xf32, #hivm.address_space<ub>>
      %29 = arith.index_cast %arg12 : i32 to index
      %30 = arith.index_cast %15 : i32 to index
      %31 = affine.apply #map()[%29, %30]
      %32 = arith.index_cast %31 : index to i64
      %33 = arith.cmpi eq, %32, %c1_i64 : i64
      %34 = arith.select %33, %5, %4 : memref<128xf32, #hivm.address_space<ub>>
      %35 = arith.index_cast %arg12 : i32 to index
      %36 = arith.index_cast %15 : i32 to index
      %37 = affine.apply #map()[%35, %36]
      %38 = arith.index_cast %37 : index to i64
      %39 = arith.cmpi eq, %38, %c1_i64 : i64
      %40 = arith.select %39, %7, %6 : memref<128xf32, #hivm.address_space<ub>>
      %41 = arith.index_cast %arg12 : i32 to index
      %42 = arith.index_cast %15 : i32 to index
      %43 = affine.apply #map()[%41, %42]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.cmpi eq, %44, %c1_i64 : i64
      %46 = arith.select %45, %9, %8 : memref<64xf32, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg12 : i32 to index
      %48 = arith.index_cast %15 : i32 to index
      %49 = affine.apply #map()[%47, %48]
      %50 = arith.index_cast %49 : index to i64
      %51 = arith.cmpi eq, %50, %c1_i64 : i64
      %52 = arith.select %51, %11, %10 : memref<64xf32, #hivm.address_space<ub>>
      %53 = arith.index_cast %arg12 : i32 to index
      %54 = arith.index_cast %15 : i32 to index
      %55 = affine.apply #map()[%53, %54]
      %56 = arith.index_cast %55 : index to i1
      %57 = arith.select %56, %c6_i64, %c7_i64 : i64
      %58 = arith.index_cast %55 : index to i1
      %59 = arith.select %58, %c4_i64, %c5_i64 : i64
      %60 = arith.index_cast %55 : index to i1
      %61 = arith.select %60, %c2_i64, %c3_i64 : i64
      %62 = arith.index_cast %55 : index to i1
      %63 = arith.xori %62, %true : i1
      %64 = arith.extui %63 : i1 to i64
      hivm.hir.set_mask_norm
      %65 = arith.divsi %arg12, %16 : i32
      %66 = arith.remsi %65, %arg9 : i32
      %67 = arith.divsi %66, %c8_i32 : i32
      %68 = arith.remsi %66, %c8_i32 : i32
      %69 = arith.muli %67, %c3072_i32 : i32
      %70 = arith.muli %68, %c384_i32 : i32
      %71 = arith.index_cast %69 : i32 to index
      %72 = arith.index_cast %70 : i32 to index
      %73 = affine.apply #map1()[%71, %72]
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%73], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %64]
      %cast = memref.cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_0 = memref.cast %52 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast, %cast_0, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
      %74 = affine.apply #map2()[%71, %72]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%74], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_3 = memref.cast %46 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_2, %cast_3, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
      %75 = affine.apply #map3()[%71, %72]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%75], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_5 = memref.cast %reinterpret_cast_4 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_6 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_5, %cast_6, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      %76 = affine.apply #map4()[%71, %72]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%76], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %61]
      %cast_8 = memref.cast %reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_9 = memref.cast %34 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_8, %cast_9, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %77 = arith.muli %67, %c2048_i32 : i32
      %78 = arith.muli %68, %c64_i32 : i32
      %79 = arith.muli %68, %c128_i32 : i32
      %80 = arith.index_cast %77 : i32 to index
      %81 = arith.index_cast %78 : i32 to index
      %82 = affine.apply #map1()[%80, %81]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%82], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_11 = memref.cast %52 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %reinterpret_cast_10 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_11, %cast_12, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      %83 = affine.apply #map5()[%81, %80]
      %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_14 = memref.cast %46 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %reinterpret_cast_13 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_14, %cast_15, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      %84 = arith.index_cast %79 : i32 to index
      %85 = affine.apply #map6()[%84, %80]
      %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_17 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %reinterpret_cast_16 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_17, %cast_18, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %64]
      %86 = arith.muli %67, %c1024_i32 : i32
      %87 = arith.index_cast %86 : i32 to index
      %88 = affine.apply #map1()[%87, %84]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%88], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_20 = memref.cast %34 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %reinterpret_cast_19 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_20, %cast_21, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %61]
      %89 = arith.muli %67, %c32_i32 : i32
      %90 = arith.muli %68, %c4_i32 : i32
      %91 = arith.index_cast %89 : i32 to index
      %92 = arith.index_cast %90 : i32 to index
      %93 = affine.apply #map1()[%91, %92]
      %reinterpret_cast_22 = memref.reinterpret_cast %arg8 to offset: [%93], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %59]
      %cast_23 = memref.cast %reinterpret_cast_22 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_24 = memref.cast %28 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_23, %cast_24, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      %94 = affine.apply #map7()[%91, %92]
      %reinterpret_cast_25 = memref.reinterpret_cast %arg8 to offset: [%94], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %57]
      %cast_26 = memref.cast %reinterpret_cast_25 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_27 = memref.cast %22 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_26, %cast_27, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %95 = arith.muli %67, %c16_i32 : i32
      %96 = arith.muli %68, %c2_i32 : i32
      %97 = arith.index_cast %95 : i32 to index
      %98 = arith.index_cast %96 : i32 to index
      %99 = affine.apply #map1()[%97, %98]
      %reinterpret_cast_28 = memref.reinterpret_cast %arg5 to offset: [%99], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      %cast_29 = memref.cast %28 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %reinterpret_cast_28 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_29, %cast_30, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %59]
      %reinterpret_cast_31 = memref.reinterpret_cast %arg6 to offset: [%99], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_32 = memref.cast %22 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_33 = memref.cast %reinterpret_cast_31 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_32, %cast_33, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %57]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
