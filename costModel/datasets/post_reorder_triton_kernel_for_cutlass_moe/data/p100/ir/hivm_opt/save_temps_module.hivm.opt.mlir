#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 128) mod 2)>
#map2 = affine_map<()[s0, s1] -> (s1, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @post_reorder_triton_kernel_for_cutlass_moe_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @post_reorder_triton_kernel_for_cutlass_moe(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %c256_i64 = arith.constant 256 : i64
    %c768_i64 = arith.constant 768 : i64
    %0 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c768_i64) : memref<128xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.muli %arg13, %arg12 : i32
    %7 = arith.divsi %5, %6 : i32
    %8 = arith.remsi %7, %arg11 : i32
    %9 = arith.divsi %8, %arg8 : i32
    %10 = arith.remsi %8, %arg8 : i32
    %11 = arith.muli %9, %arg8 : i32
    %12 = arith.muli %9, %arg10 : i32
    %13 = arith.index_cast %11 : i32 to index
    %14 = arith.index_cast %10 : i32 to index
    %15 = affine.apply #map()[%13, %14]
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg14 = %c0_i32 to %arg10 step %c128_i32  : i32 {
      %16 = arith.index_cast %arg14 : i32 to index
      %17 = affine.apply #map1()[%16]
      %18 = arith.index_cast %17 : index to i64
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      %20 = arith.select %19, %1, %0 : memref<128xf16, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg14 : i32 to index
      %22 = affine.apply #map1()[%21]
      %23 = arith.index_cast %22 : index to i64
      %24 = arith.cmpi eq, %23, %c1_i64 : i64
      %25 = arith.select %24, %3, %2 : memref<128xf16, #hivm.address_space<ub>>
      %26 = arith.index_cast %arg14 : i32 to index
      %27 = affine.apply #map1()[%26]
      %28 = arith.index_cast %27 : index to i1
      %29 = arith.xori %28, %true : i1
      %30 = arith.extui %29 : i1 to i64
      %31 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %32 = arith.cmpi ne, %31, %arg9 : i32
      scf.if %32 {
        %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %34 = arith.extsi %33 : i32 to i64
        %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%15], sizes: [1], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %35 = memref.load %reinterpret_cast_1[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %36 = arith.extsi %arg10 : i32 to i64
        %37 = arith.muli %34, %36 : i64
        %38 = arith.index_cast %37 : i64 to index
        %39 = arith.index_cast %arg14 : i32 to index
        %40 = affine.apply #map()[%38, %39]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %41 = arith.index_cast %arg10 : i32 to index
        %42 = affine.max #map2()[%39, %41]
        %43 = affine.min #map3()[%42, %39]
        %44 = affine.apply #map4()[%43, %39]
        %45 = arith.cmpi slt, %44, %c128 : index
        %subview = memref.subview %reinterpret_cast_2[0] [%44] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %25[0] [%44] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %45 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_13 = memref.cast %25 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_half_to_1d(%cst, %cast_13) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %30]
        %cast = memref.cast %subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_3 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_half(%cast, %cast_4, %c2_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %30]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %46 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
        %cast_5 = memref.cast %25 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %20 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %46 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_half(%cast_5, %35, %cast_6, %cast_7) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %30]
        %47 = arith.index_cast %12 : i32 to index
        %48 = affine.apply #map()[%47, %39]
        %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%44] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %20[0] [%44] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_11 = memref.cast %subview_10 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_12 = memref.cast %subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_11, %cast_12, %c1_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %30]
      } {pipeline.veconly}
    }
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
