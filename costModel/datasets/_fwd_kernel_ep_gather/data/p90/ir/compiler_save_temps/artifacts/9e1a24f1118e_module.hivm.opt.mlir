#map = affine_map<()[s0] -> (s0 mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_fwd_kernel_ep_gather_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_fwd_kernel_ep_gather(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, false, true, false, true, false, true, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %0 = hivm.hir.pointer_cast(%c4096_i64) : memref<512xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c8192_i64) : memref<512xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c2048_i64) : memref<512xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c6144_i64) : memref<512xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg12, %arg13 : i32
    %5 = arith.muli %4, %arg14 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg14, %arg13 : i32
    %9 = hivm.hir.pointer_cast(%c0_i64) : memref<512xf32, #hivm.address_space<ub>>
    %cast = memref.cast %9 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    scf.for %arg15 = %7 to %5 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %10 = arith.divsi %arg15, %8 : i32
      %11 = arith.remsi %10, %arg12 : i32
      %12 = arith.muli %11, %c4_i32 : i32
      scf.for %arg16 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %13 = arith.index_cast %arg16 : i32 to index
        %14 = affine.apply #map()[%13]
        %15 = arith.index_cast %14 : index to i64
        %16 = arith.cmpi eq, %15, %c1_i64 : i64
        %17 = arith.select %16, %1, %0 : memref<512xf32, #hivm.address_space<ub>>
        %18 = arith.index_cast %arg16 : i32 to index
        %19 = affine.apply #map()[%18]
        %20 = arith.index_cast %19 : index to i1
        %21 = arith.xori %20, %true : i1
        %22 = arith.extui %21 : i1 to i64
        %23 = arith.addi %12, %arg16 : i32
        %24 = arith.cmpi slt, %23, %arg3 : i32
        scf.if %24 {
          %25 = arith.muli %23, %arg7 : i32
          %26 = arith.muli %23, %arg9 : i32
          %27 = arith.extsi %arg5 : i32 to i64
          %28 = arith.index_cast %25 : i32 to index
          %29 = arith.index_cast %26 : i32 to index
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %22]
          %cast_0 = memref.cast %9 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_1 = memref.cast %17 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_0, %cast_1, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          %30 = scf.for %arg17 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg18 = %17) -> (memref<512xf32, #hivm.address_space<ub>>)  : i32 {
            %33 = arith.index_cast %arg17 : i32 to index
            %34 = affine.apply #map()[%33]
            %35 = arith.index_cast %34 : index to i64
            %36 = arith.cmpi eq, %35, %c1_i64 : i64
            %37 = arith.select %36, %3, %2 : memref<512xf32, #hivm.address_space<ub>>
            %38 = arith.index_cast %arg17 : i32 to index
            %39 = affine.apply #map()[%38]
            %40 = arith.index_cast %39 : index to i1
            %41 = arith.xori %40, %true : i1
            %42 = arith.extui %41 : i1 to i64
            %43 = arith.index_cast %arg17 : i32 to index
            %44 = affine.apply #map1()[%28, %43]
            %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%44], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %45 = memref.load %reinterpret_cast_4[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %46 = affine.apply #map1()[%29, %43]
            %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%46], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %47 = memref.load %reinterpret_cast_5[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %48 = arith.muli %45, %27 : i64
            %49 = arith.index_cast %48 : i64 to index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %42]
            %cast_7 = memref.cast %reinterpret_cast_6 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_8 = memref.cast %37 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_float(%cast_7, %cast_8, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %50 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_9 = memref.cast %37 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_10 = memref.cast %37 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_11 = memref.cast %50 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_9, %47, %cast_10, %cast_11) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.pipe_barrier[<PIPE_V>]
            %51 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_12 = memref.cast %arg18 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_13 = memref.cast %37 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_14 = memref.cast %17 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_15 = memref.cast %51 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vadd_1d_float(%cast_12, %cast_13, %cast_14, %cast_15) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %42]
            scf.yield %17 : memref<512xf32, #hivm.address_space<ub>>
          }
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %31 = arith.muli %23, %arg11 : i32
          %32 = arith.index_cast %31 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%32], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_2 = memref.cast %30 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_3 = memref.cast %reinterpret_cast : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_2, %cast_3, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %22]
        } {pipeline.veconly}
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
