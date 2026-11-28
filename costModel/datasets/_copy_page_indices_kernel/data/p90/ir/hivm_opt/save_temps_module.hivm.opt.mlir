#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 256 + ((s2 - s3) floordiv 48) * ((s1 + 255) floordiv 256)) mod 2)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1)>
#map3 = affine_map<()[s0] -> (0, s0)>
#map4 = affine_map<()[s0] -> (256, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_copy_page_indices_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_copy_page_indices_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
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
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg10 = %5 to %3 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %7 = arith.divsi %arg10, %6 : i32
      %8 = arith.remsi %7, %arg7 : i32
      %9 = arith.muli %8, %arg5 : i32
      %10 = arith.index_cast %8 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %11 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %12 = affine.apply #map()[%10]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %13 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %14 = arith.subi %13, %11 : i32
      %15 = arith.index_cast %9 : i32 to index
      %16 = arith.index_cast %11 : i32 to index
      scf.for %arg11 = %c0_i32 to %14 step %c256_i32  : i32 {
        %17 = arith.index_cast %arg11 : i32 to index
        %18 = arith.index_cast %14 : i32 to index
        %19 = arith.index_cast %arg10 : i32 to index
        %20 = arith.index_cast %5 : i32 to index
        %21 = affine.apply #map1()[%17, %18, %19, %20]
        %22 = arith.index_cast %21 : index to i64
        %23 = arith.cmpi eq, %22, %c1_i64 : i64
        %24 = arith.select %23, %1, %0 : memref<256xi32, #hivm.address_space<ub>>
        %25 = arith.index_cast %arg11 : i32 to index
        %26 = arith.index_cast %14 : i32 to index
        %27 = arith.index_cast %arg10 : i32 to index
        %28 = arith.index_cast %5 : i32 to index
        %29 = affine.apply #map1()[%25, %26, %27, %28]
        %30 = arith.index_cast %29 : index to i1
        %31 = arith.xori %30, %true : i1
        %32 = arith.extui %31 : i1 to i64
        %33 = arith.subi %14, %arg11 : i32
        %34 = arith.index_cast %arg11 : i32 to index
        %35 = affine.apply #map2()[%15, %34]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %36 = arith.index_cast %33 : i32 to index
        %37 = affine.max #map3()[%36]
        %38 = affine.min #map4()[%37]
        %39 = arith.cmpi slt, %38, %c256 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%38] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %24[0] [%38] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %39 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_8 = memref.cast %24 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_int32_t_to_1d(%c0_i32, %cast_8) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %32]
        %cast = memref.cast %subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_3 = memref.cast %subview_2 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int32_t(%cast, %cast_3, %c2_i32, %c0_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %40 = affine.apply #map2()[%16, %34]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %reinterpret_cast_4[0] [%38] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_6 = memref.cast %subview_2 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %subview_5 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_6, %cast_7, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %32]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
