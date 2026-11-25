#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @assign_draft_cache_locs_page_size_1_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @assign_draft_cache_locs_page_size_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c15_i32 = arith.constant 15 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1024_i64) : memref<256xi32, #hivm.address_space<ub>>
    %2 = arith.muli %arg7, %arg8 : i32
    %3 = arith.muli %2, %arg9 : i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.muli %arg9, %arg8 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    scf.for %arg10 = %5 to %3 step %c48_i32  : i32 {
      %7 = arith.index_cast %arg10 : i32 to index
      %8 = arith.index_cast %5 : i32 to index
      %9 = affine.apply #map()[%7, %8]
      %10 = arith.index_cast %9 : index to i64
      %11 = arith.cmpi eq, %10, %c1_i64 : i64
      %12 = arith.select %11, %1, %0 : memref<256xi32, #hivm.address_space<ub>>
      %13 = arith.index_cast %arg10 : i32 to index
      %14 = arith.index_cast %5 : i32 to index
      %15 = affine.apply #map()[%13, %14]
      %16 = arith.index_cast %15 : index to i1
      %17 = arith.xori %16, %true : i1
      %18 = arith.extui %17 : i1 to i64
      hivm.hir.set_mask_norm
      %19 = arith.divsi %arg10, %6 : i32
      %20 = arith.remsi %19, %arg7 : i32
      %21 = arith.muli %20, %c15_i32 : i32
      %22 = arith.index_cast %20 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %23 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %24 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %25 = arith.muli %24, %c1024_i32 : i32
      %26 = arith.addi %25, %23 : i32
      %27 = arith.index_cast %26 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast_1[0] [15] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<15xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %12[0] [15] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<15xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %18]
      %cast = memref.cast %12 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%c0_i32, %cast) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_3 = memref.cast %subview : memref<15xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_4 = memref.cast %subview_2 : memref<15xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int32_t(%cast_3, %cast_4, %c2_i32, %c0_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %28 = arith.index_cast %21 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%28], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [15] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<15xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_7 = memref.cast %subview_2 : memref<15xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %subview_6 : memref<15xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_int32_t(%cast_7, %cast_8, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %18]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
