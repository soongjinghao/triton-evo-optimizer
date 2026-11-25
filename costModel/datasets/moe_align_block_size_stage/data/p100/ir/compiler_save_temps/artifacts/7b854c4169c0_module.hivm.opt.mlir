#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + ((s1 - s2) floordiv 48) * 4 - ((s0 + ((s1 - s2) floordiv 48) * 4 - 1) floordiv 2) * 2 - 1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @moe_align_block_size_stage1_stage2_fused_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @moe_align_block_size_stage1_stage2_fused(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c1_i64 = arith.constant 1 : i64
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c96_i64 = arith.constant 96 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<1xi32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c32_i64) : memref<1xi32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c96_i64) : memref<1xi32, #hivm.address_space<ub>>
    %4 = arith.muli %arg6, %arg7 : i32
    %5 = arith.muli %4, %arg8 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg8, %arg7 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg9 = %7 to %5 step %c48_i32  : i32 {
      %9 = arith.index_cast %arg9 : i32 to index
      %10 = arith.index_cast %7 : i32 to index
      %11 = affine.apply #map()[%9, %10]
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.cmpi eq, %12, %c1_i64 : i64
      %14 = arith.select %13, %1, %0 : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_mask_norm
      %15 = arith.divsi %arg9, %arg8 : i32
      %16 = arith.remsi %15, %arg7 : i32
      %17 = arith.divsi %arg9, %8 : i32
      %18 = arith.remsi %17, %arg6 : i32
      %19 = arith.muli %18, %c4_i32 : i32
      %20 = arith.addi %19, %16 : i32
      %21 = arith.cmpi slt, %20, %arg5 : i32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.if %21 {
        %24 = arith.index_cast %20 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %25 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %26 = arith.addi %18, %c1_i32 : i32
        %27 = arith.muli %26, %c4_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %25 : i32 to index
        %30 = affine.apply #map1()[%28, %29]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%30], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %31 = memref.load %reinterpret_cast_0[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = arith.addi %31, %c1_i32 : i32
        memref.store %32, %14[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast = memref.cast %14 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_1 = memref.cast %reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      %22 = arith.index_cast %16 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %23 = scf.for %arg10 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %c0_i32) -> (i32)  : i32 {
        %24 = arith.index_cast %arg10 : i32 to index
        %25 = arith.index_cast %arg9 : i32 to index
        %26 = arith.index_cast %7 : i32 to index
        %27 = affine.apply #map2()[%24, %25, %26]
        %28 = arith.index_cast %27 : index to i64
        %29 = arith.cmpi eq, %28, %c1_i64 : i64
        %30 = arith.select %29, %3, %2 : memref<1xi32, #hivm.address_space<ub>>
        %31 = arith.muli %arg10, %c4_i32 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = affine.apply #map1()[%32, %22]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
        %34 = memref.load %reinterpret_cast[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %35 = arith.addi %arg11, %34 : i32
        memref.store %35, %30[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast = memref.cast %30 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_0 = memref.cast %reinterpret_cast : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_0, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
        scf.yield %35 : i32
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
