#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<()[s0] -> (4, s0)>
#map2 = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
#map4 = affine_map<()[s0, s1] -> (s1, s0)>
#map5 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map6 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map7 = affine_map<()[s0] -> (1, s0)>
#map8 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @copy_all_layer_kv_cache_tiled_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_int64_t_to_1d(i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int8_t(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int8_t_to_half_2d_with_mode(memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f16) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_int8_t_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_int8_t(memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @copy_all_layer_kv_cache_tiled(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1344_i64 = arith.constant 1344 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1376_i64 = arith.constant 1376 : i64
    %c544_i64 = arith.constant 544 : i64
    %c1408_i64 = arith.constant 1408 : i64
    %c576_i64 = arith.constant 576 : i64
    %c832_i64 = arith.constant 832 : i64
    %c1664_i64 = arith.constant 1664 : i64
    %0 = hivm.hir.pointer_cast(%c832_i64) : memref<4x64xi8, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1664_i64) : memref<4x64xi8, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c576_i64) : memref<4x64xi8, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c1408_i64) : memref<4x64xi8, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c544_i64) : memref<4xi64, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c1376_i64) : memref<4xi64, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c512_i64) : memref<4xi64, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c1344_i64) : memref<4xi64, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c832_i64) : memref<4x64xf16, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1664_i64) : memref<4x64xf16, #hivm.address_space<ub>>
    %10 = arith.muli %arg8, %arg9 : i32
    %11 = arith.muli %10, %arg10 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg10, %arg9 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [4], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[1]>, #hivm.address_space<gm>>
    %15 = arith.index_cast %arg7 : i32 to index
    %16 = affine.max #map()[%15]
    %17 = affine.min #map1()[%16]
    %18 = arith.cmpi slt, %17, %c4 : index
    %subview = memref.subview %reinterpret_cast[0] [%17] [1] : memref<4xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [4], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[1]>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %reinterpret_cast_0[0] [%17] [1] : memref<4xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
    %19 = affine.max #map()[%17]
    %20 = affine.min #map1()[%19]
    %21 = hivm.hir.pointer_cast(%c0_i64) : memref<4x64xf16, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %21 [[0, 1]] : memref<4x64xf16, #hivm.address_space<ub>> into memref<256xf16, #hivm.address_space<ub>>
    %cast = memref.cast %collapse_shape : memref<256xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_half_to_1d(%cst, %cast) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg11 = %13 to %11 step %c48_i32  : i32 {
      %22 = arith.index_cast %arg11 : i32 to index
      %23 = arith.index_cast %13 : i32 to index
      %24 = affine.apply #map2()[%22, %23]
      %25 = arith.index_cast %24 : index to i64
      %26 = arith.cmpi eq, %25, %c1_i64 : i64
      %27 = arith.select %26, %1, %0 : memref<4x64xi8, #hivm.address_space<ub>>
      %28 = arith.index_cast %arg11 : i32 to index
      %29 = arith.index_cast %13 : i32 to index
      %30 = affine.apply #map2()[%28, %29]
      %31 = arith.index_cast %30 : index to i64
      %32 = arith.cmpi eq, %31, %c1_i64 : i64
      %33 = arith.select %32, %3, %2 : memref<4x64xi8, #hivm.address_space<ub>>
      %34 = arith.index_cast %arg11 : i32 to index
      %35 = arith.index_cast %13 : i32 to index
      %36 = affine.apply #map2()[%34, %35]
      %37 = arith.index_cast %36 : index to i64
      %38 = arith.cmpi eq, %37, %c1_i64 : i64
      %39 = arith.select %38, %5, %4 : memref<4xi64, #hivm.address_space<ub>>
      %40 = arith.index_cast %arg11 : i32 to index
      %41 = arith.index_cast %13 : i32 to index
      %42 = affine.apply #map2()[%40, %41]
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.cmpi eq, %43, %c1_i64 : i64
      %45 = arith.select %44, %7, %6 : memref<4xi64, #hivm.address_space<ub>>
      %46 = arith.index_cast %arg11 : i32 to index
      %47 = arith.index_cast %13 : i32 to index
      %48 = affine.apply #map2()[%46, %47]
      %49 = arith.index_cast %48 : index to i64
      %50 = arith.cmpi eq, %49, %c1_i64 : i64
      %51 = arith.select %50, %9, %8 : memref<4x64xf16, #hivm.address_space<ub>>
      %52 = arith.index_cast %arg11 : i32 to index
      %53 = arith.index_cast %13 : i32 to index
      %54 = affine.apply #map2()[%52, %53]
      %55 = arith.index_cast %54 : index to i1
      %56 = arith.xori %55, %true : i1
      %57 = arith.extui %56 : i1 to i64
      %58 = arith.index_cast %54 : index to i1
      %59 = arith.select %58, %c2_i64, %c3_i64 : i64
      hivm.hir.set_mask_norm
      %60 = arith.divsi %arg11, %arg10 : i32
      %61 = arith.remsi %60, %arg9 : i32
      %62 = arith.divsi %arg11, %14 : i32
      %63 = arith.remsi %62, %arg8 : i32
      %64 = arith.index_cast %63 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %65 = memref.load %reinterpret_cast_2[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%64], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %66 = memref.load %reinterpret_cast_3[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %67 = arith.muli %61, %c64_i32 : i32
      %subview_4 = memref.subview %45[0] [%17] [1] : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      scf.if %18 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_23 = memref.cast %45 : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_int64_t_to_1d(%c0_i64, %cast_23) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %59]
      %cast_5 = memref.cast %subview : memref<?xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_6 = memref.cast %subview_4 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int64_t(%cast_5, %cast_6, %c2_i32, %c0_i64, %c0) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
      %subview_7 = memref.subview %39[0] [%17] [1] : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      scf.if %18 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_23 = memref.cast %39 : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_int64_t_to_1d(%c0_i64, %cast_23) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %57]
      %cast_8 = memref.cast %subview_1 : memref<?xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_9 = memref.cast %subview_7 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int64_t(%cast_8, %cast_9, %c2_i32, %c0_i64, %c0) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %68 = arith.index_cast %67 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
      scf.for %arg12 = %c0 to %20 step %c1 {
        %74 = memref.load %45[%arg12] {DiscreteMemAccess} : memref<4xi64, #hivm.address_space<ub>>
        %75 = arith.muli %74, %65 : i64
        %76 = arith.index_cast %75 : i64 to index
        %77 = affine.apply #map3()[%76, %68]
        %78 = arith.index_cast %77 : index to i64
        %79 = arith.addi %66, %78 : i64
        %80 = hivm.hir.pointer_cast(%79) [%c64] : memref<?xi8, #hivm.address_space<gm>>
        %reinterpret_cast_23 = memref.reinterpret_cast %80 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_24 = memref.subview %33[%arg12, 0] [1, 64] [1, 1] : memref<4x64xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_25 = memref.collapse_shape %reinterpret_cast_23 [[0, 1]] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_26 = memref.collapse_shape %subview_24 [[0, 1]] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> into memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %cast_27 = memref.cast %collapse_shape_25 : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_28 = memref.cast %collapse_shape_26 : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int8_t(%cast_27, %cast_28, %c0_i32, %c0_i8, %c0) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) -> ()
      } {ExtractedLoadOrStore, hivm.parallel_loop, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %59]
      %69 = arith.index_cast %65 : i64 to index
      %70 = affine.max #map4()[%68, %69]
      %71 = affine.min #map5()[%70, %68]
      %72 = affine.min #map6()[%71, %68]
      %73 = affine.max #map()[%72]
      %subview_10 = memref.subview %33[0, 0] [%19, %73] [1, 1] : memref<4x64xi8, #hivm.address_space<ub>> to memref<?x?xi8, strided<[64, 1]>, #hivm.address_space<ub>>
      %subview_11 = memref.subview %21[0, 0] [%19, %73] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_12 = memref.cast %subview_10 : memref<?x?xi8, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %subview_11 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_int8_t_to_half_2d_with_mode(%cast_12, %cast_13, %c0_i32) : (memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %collapse_shape_14 = memref.collapse_shape %51 [[0, 1]] : memref<4x64xf16, #hivm.address_space<ub>> into memref<256xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_15 = memref.cast %collapse_shape : memref<256xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %collapse_shape_14 : memref<256xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_half(%cast_15, %cast_16, %c0_i32, %cst) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16) -> ()
      %subview_17 = memref.subview %51[0, 0] [%19, %73] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_18 = memref.cast %subview_11 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %subview_17 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_2d_half(%cast_18, %cast_19, %c0_i32, %cst) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f16) -> ()
      %collapse_shape_20 = memref.collapse_shape %27 [[0, 1]] : memref<4x64xi8, #hivm.address_space<ub>> into memref<256xi8, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_21 = memref.cast %collapse_shape_14 : memref<256xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %collapse_shape_20 : memref<256xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_half_to_int8_t_1d_with_mode(%cast_21, %cast_22, %c4_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.for %arg12 = %c0 to %20 step %c1 {
        %74 = memref.load %39[%arg12] {DiscreteMemAccess} : memref<4xi64, #hivm.address_space<ub>>
        %75 = arith.muli %74, %65 : i64
        %76 = arith.index_cast %75 : i64 to index
        %77 = affine.apply #map3()[%76, %68]
        %78 = arith.index_cast %77 : index to i64
        %79 = arith.addi %66, %78 : i64
        %80 = hivm.hir.pointer_cast(%79) [%c64] : memref<?xi8, #hivm.address_space<gm>>
        %reinterpret_cast_23 = memref.reinterpret_cast %80 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_24 = memref.subview %27[%arg12, 0] [1, 64] [1, 1] {DiscreteMemAccess} : memref<4x64xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
        %81 = arith.index_cast %arg12 : index to i32
        %82 = arith.cmpi slt, %81, %arg7 : i32
        %83 = arith.index_castui %82 : i1 to index
        %84 = affine.min #map7()[%83]
        %85 = affine.max #map()[%84]
        %86 = affine.min #map8()[%83, %71, %68]
        %87 = affine.max #map()[%86]
        %subview_25 = memref.subview %subview_24[0, 0] [%85, %87] [1, 1] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_26 = memref.subview %reinterpret_cast_23[0, 0] [%85, %87] [1, 1] : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_27 = memref.cast %subview_25 : memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_28 = memref.cast %subview_26 : memref<?x?xi8, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_int8_t(%cast_27, %cast_28, %c0_i32) : (memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {ExtractedLoadOrStore, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %57]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
