#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (s0 + 512)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_dequantize_k_cache_fast_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bfloat16_t_to_float_1d_with_mode(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_bfloat16_t_to_1d(bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_dequantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c3104_i64 = arith.constant 3104 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3616_i64 = arith.constant 3616 : i64
    %c0_i64 = arith.constant 0 : i64
    %c6688_i64 = arith.constant 6688 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c4640_i64 = arith.constant 4640 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c3360_i64 = arith.constant 3360 : i64
    %c6720_i64 = arith.constant 6720 : i64
    %0 = hivm.hir.pointer_cast(%c3360_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c6720_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c4640_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c3072_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c6688_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c0_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c3616_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c4640_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c4640_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg11, %arg12 : i32
    %13 = arith.muli %12, %arg13 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg13, %arg12 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    scf.for %arg14 = %15 to %13 step %c48_i32  : i32 {
      %17 = arith.index_cast %arg14 : i32 to index
      %18 = arith.index_cast %15 : i32 to index
      %19 = affine.apply #map()[%17, %18]
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      %22 = arith.select %21, %1, %0 : memref<128xbf16, #hivm.address_space<ub>>
      %23 = arith.index_cast %arg14 : i32 to index
      %24 = arith.index_cast %15 : i32 to index
      %25 = affine.apply #map()[%23, %24]
      %26 = arith.index_cast %25 : index to i64
      %27 = arith.cmpi eq, %26, %c1_i64 : i64
      %28 = arith.select %27, %3, %2 : memref<4x128xbf16, #hivm.address_space<ub>>
      %29 = arith.index_cast %arg14 : i32 to index
      %30 = arith.index_cast %15 : i32 to index
      %31 = affine.apply #map()[%29, %30]
      %32 = arith.index_cast %31 : index to i64
      %33 = arith.cmpi eq, %32, %c1_i64 : i64
      %34 = arith.select %33, %5, %4 : memref<4x1xf32, #hivm.address_space<ub>>
      %35 = arith.index_cast %arg14 : i32 to index
      %36 = arith.index_cast %15 : i32 to index
      %37 = affine.apply #map()[%35, %36]
      %38 = arith.index_cast %37 : index to i64
      %39 = arith.cmpi eq, %38, %c1_i64 : i64
      %40 = arith.select %39, %7, %6 : memref<4x128xbf16, #hivm.address_space<ub>>
      %41 = arith.index_cast %arg14 : i32 to index
      %42 = arith.index_cast %15 : i32 to index
      %43 = affine.apply #map()[%41, %42]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.cmpi eq, %44, %c1_i64 : i64
      %46 = arith.select %45, %9, %8 : memref<4x128xf32, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg14 : i32 to index
      %48 = arith.index_cast %15 : i32 to index
      %49 = affine.apply #map()[%47, %48]
      %50 = arith.index_cast %49 : index to i64
      %51 = arith.cmpi eq, %50, %c1_i64 : i64
      %52 = arith.select %51, %11, %10 : memref<4x128xf32, #hivm.address_space<ub>>
      %53 = arith.index_cast %arg14 : i32 to index
      %54 = arith.index_cast %15 : i32 to index
      %55 = affine.apply #map()[%53, %54]
      %56 = arith.index_cast %55 : index to i1
      %57 = arith.select %56, %c2_i64, %c3_i64 : i64
      %58 = arith.index_cast %55 : index to i1
      %59 = arith.xori %58, %true : i1
      %60 = arith.extui %59 : i1 to i64
      hivm.hir.set_mask_norm
      %61 = arith.divsi %arg14, %16 : i32
      %62 = arith.remsi %61, %arg11 : i32
      %63 = arith.muli %62, %arg8 : i32
      %64 = arith.index_cast %63 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [4, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_1 = memref.collapse_shape %40 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
      %cast = memref.cast %collapse_shape : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %collapse_shape_1 : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast, %cast_2, %c0_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_3 = memref.collapse_shape %46 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %60]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_4 = memref.cast %collapse_shape_1 : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %collapse_shape_3 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_4, %cast_5, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %65 = arith.muli %62, %arg9 : i32
      %66 = arith.index_cast %65 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%66], sizes: [4, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_7 = memref.collapse_shape %reinterpret_cast_6 [[0, 1]] : memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_8 = memref.collapse_shape %34 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
      %cast_9 = memref.cast %collapse_shape_7 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_10 = memref.cast %collapse_shape_8 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_9, %cast_10, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %67 = hivm.hir.pointer_cast(%c3104_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_11 = memref.cast %46 : memref<4x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %34 : memref<4x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %52 : memref<4x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %67 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_11, %cast_12, %cast_13, %cast_14) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %68 = arith.muli %62, %arg7 : i32
      %69 = arith.index_cast %68 : i32 to index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [4, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_16 = memref.collapse_shape %52 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_17 = memref.collapse_shape %28 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_18 = memref.cast %collapse_shape_16 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %collapse_shape_17 : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_18, %cast_19, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %collapse_shape_20 = memref.collapse_shape %reinterpret_cast_15 [[0, 1]] : memref<4x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_21 = memref.cast %collapse_shape_17 : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %collapse_shape_20 : memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_21, %cast_22, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %60]
      %70 = arith.muli %62, %arg10 : i32
      %71 = arith.index_cast %70 : i32 to index
      %reinterpret_cast_23 = memref.reinterpret_cast %arg6 to offset: [%71], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %72 = affine.apply #map1()[%69]
      %reinterpret_cast_24 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast_23[0] [64] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_25 = memref.subview %22[0] [64] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %57]
      %cast_26 = memref.cast %22 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_26) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_27 = memref.cast %subview : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_28 = memref.cast %subview_25 : memref<64xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_27, %cast_28, %c2_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_29 = memref.subview %reinterpret_cast_24[0] [64] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_30 = memref.cast %subview_25 : memref<64xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %subview_29 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_30, %cast_31, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %57]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
