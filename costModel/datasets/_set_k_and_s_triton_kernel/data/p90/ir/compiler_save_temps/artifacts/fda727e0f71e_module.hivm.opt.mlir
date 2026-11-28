#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_set_k_and_s_triton_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_set_k_and_s_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c0 = arith.constant 0 : index
    %c6_i64 = arith.constant 6 : i64
    %c63_i64 = arith.constant 63 : i64
    %c8448_i64 = arith.constant 8448 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c288_i64 = arith.constant 288 : i64
    %c0_i64 = arith.constant 0 : i64
    %c256_i64 = arith.constant 256 : i64
    %c544_i64 = arith.constant 544 : i64
    %0 = hivm.hir.pointer_cast(%c256_i64) : memref<1xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c544_i64) : memref<1xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c288_i64) : memref<128xf16, #hivm.address_space<ub>>
    %4 = arith.muli %arg9, %arg10 : i32
    %5 = arith.muli %4, %arg11 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg11, %arg10 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg12 = %7 to %5 step %c48_i32  : i32 {
      %9 = arith.index_cast %arg12 : i32 to index
      %10 = arith.index_cast %7 : i32 to index
      %11 = affine.apply #map()[%9, %10]
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.cmpi eq, %12, %c1_i64 : i64
      %14 = arith.select %13, %1, %0 : memref<1xf32, #hivm.address_space<ub>>
      %15 = arith.index_cast %arg12 : i32 to index
      %16 = arith.index_cast %7 : i32 to index
      %17 = affine.apply #map()[%15, %16]
      %18 = arith.index_cast %17 : index to i64
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      %20 = arith.select %19, %3, %2 : memref<128xf16, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg12 : i32 to index
      %22 = arith.index_cast %7 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i1
      %25 = arith.xori %24, %true : i1
      %26 = arith.extui %25 : i1 to i64
      hivm.hir.set_mask_norm
      %27 = arith.divsi %arg12, %8 : i32
      %28 = arith.remsi %27, %arg9 : i32
      %29 = arith.index_cast %28 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %30 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.muli %28, %arg8 : i32
      %32 = arith.index_cast %31 : i32 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%32], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
      %cast = memref.cast %reinterpret_cast_0 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_1 = memref.cast %20 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_half(%cast, %cast_1, %c0_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%29], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %33 = memref.load %reinterpret_cast_2[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = arith.shrsi %30, %c6_i64 : i64
      %35 = arith.andi %30, %c63_i64 : i64
      %36 = arith.muli %34, %c8448_i64 : i64
      %37 = arith.muli %35, %c128_i64 : i64
      %38 = arith.addi %36, %37 : i64
      %39 = arith.divsi %36, %c4_i64 : i64
      %40 = arith.addi %39, %c2048_i64 : i64
      %41 = arith.addi %40, %35 : i64
      %42 = arith.index_cast %38 : i64 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_4 = memref.cast %20 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %reinterpret_cast_3 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_half(%cast_4, %cast_5, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %26]
      memref.store %33, %14[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %43 = arith.index_cast %41 : i64 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_7 = memref.cast %14 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %reinterpret_cast_6 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_7, %cast_8, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %26]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
