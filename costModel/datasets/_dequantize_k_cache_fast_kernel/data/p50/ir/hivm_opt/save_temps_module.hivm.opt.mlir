#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map2 = affine_map<()[s0] -> (4, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 4, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
#map5 = affine_map<()[s0] -> (512, s0)>
#map6 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map7 = affine_map<()[s0, s1] -> (4, s0 - s1)>
#map8 = affine_map<()[s0] -> (0, s0)>
#map9 = affine_map<()[s0, s1] -> (128, s0 - s1)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2 + 512)>
#map11 = affine_map<()[s0] -> (64, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_dequantize_k_cache_fast_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_bfloat16_t_to_1d(bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bfloat16_t_to_float_1d_with_mode(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_dequantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %c3200_i64 = arith.constant 3200 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4480_i64 = arith.constant 4480 : i64
    %c0_i64 = arith.constant 0 : i64
    %c7552_i64 = arith.constant 7552 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c5504_i64 = arith.constant 5504 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c3456_i64 = arith.constant 3456 : i64
    %c7680_i64 = arith.constant 7680 : i64
    %0 = hivm.hir.pointer_cast(%c3456_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c7680_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c5504_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c3072_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c7552_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c0_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c4480_i64) : memref<4x128xbf16, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c5504_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c5504_i64) : memref<4x128xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg11, %arg12 : i32
    %13 = arith.muli %12, %arg13 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg13, %arg12 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg14 = %15 to %13 step %c48_i32  : i32 {
      %17 = arith.index_cast %arg14 : i32 to index
      %18 = arith.index_cast %15 : i32 to index
      %19 = affine.apply #map()[%17, %18]
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      %22 = arith.select %21, %1, %0 : memref<4x128xbf16, #hivm.address_space<ub>>
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
      %34 = arith.select %33, %5, %4 : memref<4x8x1xf32, #hivm.address_space<ub>>
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
      %57 = arith.xori %56, %true : i1
      %58 = arith.extui %57 : i1 to i64
      %59 = arith.index_cast %55 : index to i1
      %60 = arith.select %59, %c2_i64, %c3_i64 : i64
      %61 = arith.index_cast %55 : index to i1
      %62 = arith.select %61, %c1_i64, %c2_i64 : i64
      hivm.hir.set_mask_norm
      %63 = arith.divsi %arg14, %arg13 : i32
      %64 = arith.remsi %63, %arg12 : i32
      %65 = arith.divsi %arg14, %16 : i32
      %66 = arith.remsi %65, %arg11 : i32
      %67 = arith.muli %66, %c4_i32 : i32
      %68 = arith.cmpi slt, %64, %c4_i32 : i32
      scf.if %68 {
        %69 = arith.muli %64, %c128_i32 : i32
        %70 = arith.index_cast %67 : i32 to index
        %71 = arith.index_cast %arg8 : i32 to index
        %72 = arith.index_cast %69 : i32 to index
        %73 = affine.apply #map1()[%72, %70, %71]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%73], sizes: [4, 128], strides: [%71, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %74 = arith.index_cast %arg9 : i32 to index
        %75 = arith.index_cast %64 : i32 to index
        %76 = affine.apply #map1()[%75, %70, %74]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%76], sizes: [4, 1], strides: [%74, %74] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %77 = affine.max #map2()[%70]
        %78 = affine.min #map3()[%77, %70]
        %79 = affine.apply #map4()[%78, %70]
        %80 = affine.max #map5()[%72]
        %81 = affine.min #map6()[%80, %72]
        %82 = affine.min #map7()[%78, %70]
        %83 = affine.max #map8()[%82]
        %84 = affine.min #map9()[%81, %72]
        %85 = affine.max #map8()[%84]
        %86 = arith.cmpi slt, %83, %c4 : index
        %87 = arith.cmpi slt, %85, %c128 : index
        %88 = arith.ori %86, %87 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%83, %85] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %40[0, 0] [%83, %85] [1, 1] : memref<4x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
        scf.if %88 {
          %collapse_shape_26 = memref.collapse_shape %40 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
          %cast_27 = memref.cast %collapse_shape_26 : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_27) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %60]
        %cast = memref.cast %subview : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_3 = memref.cast %subview_2 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast, %cast_3, %c2_i32, %cst_0, %c0) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %40 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
        %collapse_shape_4 = memref.collapse_shape %46 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %62]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_5 = memref.cast %collapse_shape : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %collapse_shape_4 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_5, %cast_6, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %60]
        %subview_7 = memref.subview %34[0, 0, 0] [4, 1, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %89 = arith.cmpi slt, %79, %c4 : index
        %subview_8 = memref.subview %reinterpret_cast_1[0, 0] [%79, 1] [1, 1] : memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %subview_7[0, 0] [%79, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        scf.if %89 {
          %collapse_shape_26 = memref.collapse_shape %subview_7 [[0, 1]] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
          %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %collapse_shape_26 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
          %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %cast_28 = memref.cast %reinterpret_cast_27 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_2d(%cst, %cast_28) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
        } {hivm.unlikely_condition}
        %collapse_shape_10 = memref.collapse_shape %subview_8 [[0, 1]] : memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %58]
        %cast_12 = memref.cast %collapse_shape_11 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%collapse_shape_10, %cast_12, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %90 = hivm.hir.pointer_cast(%c3200_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_13 = memref.cast %46 : memref<4x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_14 = memref.cast %subview_7 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_15 = memref.cast %52 : memref<4x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_16 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_13, %cast_14, %cast_15, %cast_16) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %58]
        %collapse_shape_17 = memref.collapse_shape %52 [[0, 1]] : memref<4x128xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        %collapse_shape_18 = memref.collapse_shape %28 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_19 = memref.cast %collapse_shape_17 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_20 = memref.cast %collapse_shape_18 : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_19, %cast_20, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %91 = arith.index_cast %arg7 : i32 to index
        %92 = affine.apply #map1()[%72, %70, %91]
        %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [4, 128], strides: [%91, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_22 = memref.subview %28[0, 0] [%83, %85] [1, 1] : memref<4x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
        %subview_23 = memref.subview %reinterpret_cast_21[0, 0] [%83, %85] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_24 = memref.cast %subview_22 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %subview_23 : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_bfloat16_t(%cast_24, %cast_25, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %62]
      } else {
        %69 = arith.subi %64, %c4_i32 : i32
        %70 = arith.muli %69, %c128_i32 : i32
        %71 = arith.index_cast %67 : i32 to index
        %72 = arith.index_cast %arg10 : i32 to index
        %73 = arith.index_cast %70 : i32 to index
        %74 = affine.apply #map1()[%73, %71, %72]
        %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%74], sizes: [4, 128], strides: [%72, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %75 = arith.index_cast %arg7 : i32 to index
        %76 = affine.apply #map10()[%73, %71, %75]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%76], sizes: [4, 128], strides: [%75, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %77 = affine.max #map2()[%71]
        %78 = affine.min #map3()[%77, %71]
        %79 = affine.max #map11()[%73]
        %80 = affine.min #map6()[%79, %73]
        %81 = affine.min #map7()[%78, %71]
        %82 = affine.max #map8()[%81]
        %83 = affine.min #map9()[%80, %73]
        %84 = affine.max #map8()[%83]
        %85 = arith.cmpi slt, %82, %c4 : index
        %86 = arith.cmpi slt, %84, %c128 : index
        %87 = arith.ori %85, %86 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%82, %84] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %22[0, 0] [%82, %84] [1, 1] : memref<4x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
        scf.if %87 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %22 [[0, 1]] : memref<4x128xbf16, #hivm.address_space<ub>> into memref<512xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_7 = memref.cast %collapse_shape : memref<512xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_7) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %58]
        %cast = memref.cast %subview : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_3 = memref.cast %subview_2 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_bfloat16_t(%cast, %cast_3, %c2_i32, %cst_0, %c0) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_4 = memref.subview %reinterpret_cast_1[0, 0] [%82, %84] [1, 1] : memref<4x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_5 = memref.cast %subview_2 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %subview_4 : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_bfloat16_t(%cast_5, %cast_6, %c0_i32) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %58]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
