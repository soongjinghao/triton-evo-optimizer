#map = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @reshape_and_cache_kernel_flash_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @reshape_and_cache_kernel_flash(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: i64 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c8_i64 = arith.constant 8 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c48_i32 = arith.constant 48 : i32
    %0 = arith.muli %arg14, %arg15 : i32
    %1 = arith.muli %0, %arg16 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg16, %arg15 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %5 = arith.divsi %arg17, %arg16 : i32
      %6 = arith.remsi %5, %arg15 : i32
      %7 = arith.divsi %arg17, %4 : i32
      %8 = arith.remsi %7, %arg14 : i32
      %9 = arith.index_cast %8 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%9], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %10 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %11 = arith.cmpi slt, %10, %c0_i64 : i64
      scf.if %11 {
      } else {
        %12 = arith.muli %6, %c128_i32 : i32
        %13 = arith.divsi %10, %c8_i64 : i64
        %14 = arith.remsi %10, %c8_i64 : i64
        %15 = arith.muli %8, %c128_i32 : i32
        %16 = arith.muli %13, %arg12 : i64
        %17 = arith.muli %14, %arg13 : i64
        %18 = arith.addi %16, %17 : i64
        %19 = arith.index_cast %15 : i32 to index
        %20 = arith.index_cast %12 : i32 to index
        %21 = affine.apply #map()[%19, %20]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %22 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        %cast = memref.cast %reinterpret_cast_0 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_1 = memref.cast %22 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_half(%cast, %cast_1, %c0_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %23 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_3 = memref.cast %reinterpret_cast_2 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %23 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_half(%cast_3, %cast_4, %c0_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %24 = arith.index_cast %18 : i64 to index
        %25 = affine.apply #map()[%24, %20]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %cast_6 = memref.cast %22 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %reinterpret_cast_5 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_6, %cast_7, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_9 = memref.cast %23 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %reinterpret_cast_8 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_9, %cast_10, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
