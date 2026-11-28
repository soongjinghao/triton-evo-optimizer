#map = affine_map<()[s0] -> (s0 mod 2)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_fwd_kernel_ep_gather_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_fwd_kernel_ep_gather(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, false, true, false, true, false, true, false, true, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %0 = hivm.hir.pointer_cast(%c6144_i64) : memref<512xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c12288_i64) : memref<512xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c4096_i64) : memref<512xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c10240_i64) : memref<512xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c2048_i64) : memref<512xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c8192_i64) : memref<512xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg14, %arg15 : i32
    %7 = arith.muli %6, %arg16 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg16, %arg15 : i32
    %11 = hivm.hir.pointer_cast(%c0_i64) : memref<512xf32, #hivm.address_space<ub>>
    %cast = memref.cast %11 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg17 = %9 to %7 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %12 = arith.divsi %arg17, %10 : i32
      %13 = arith.remsi %12, %arg14 : i32
      %14 = arith.muli %13, %c4_i32 : i32
      scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %15 = arith.index_cast %arg18 : i32 to index
        %16 = affine.apply #map()[%15]
        %17 = arith.index_cast %16 : index to i64
        %18 = arith.cmpi eq, %17, %c1_i64 : i64
        %19 = arith.select %18, %1, %0 : memref<512xf32, #hivm.address_space<ub>>
        %20 = arith.index_cast %arg18 : i32 to index
        %21 = affine.apply #map()[%20]
        %22 = arith.index_cast %21 : index to i64
        %23 = arith.cmpi eq, %22, %c1_i64 : i64
        %24 = arith.select %23, %3, %2 : memref<512xf32, #hivm.address_space<ub>>
        %25 = arith.index_cast %arg18 : i32 to index
        %26 = affine.apply #map()[%25]
        %27 = arith.index_cast %26 : index to i64
        %28 = arith.cmpi eq, %27, %c1_i64 : i64
        %29 = arith.select %28, %5, %4 : memref<512xf32, #hivm.address_space<ub>>
        %30 = arith.index_cast %arg18 : i32 to index
        %31 = affine.apply #map()[%30]
        %32 = arith.index_cast %31 : index to i1
        %33 = arith.xori %32, %true : i1
        %34 = arith.extui %33 : i1 to i64
        %35 = arith.addi %14, %arg18 : i32
        %36 = arith.cmpi slt, %35, %arg3 : i32
        scf.if %36 {
          %37 = arith.muli %35, %arg7 : i32
          %38 = arith.index_cast %37 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %39 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %40 = affine.apply #map1()[%38]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%40], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %41 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %42 = arith.cmpi sge, %39, %c0_i64 : i64
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %34]
          %43 = arith.select %42, %29, %11 : memref<512xf32, #hivm.address_space<ub>>
          scf.if %42 {
            %49 = arith.muli %35, %arg11 : i32
            %50 = arith.index_cast %49 : i32 to index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg10 to offset: [%50], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %51 = memref.load %reinterpret_cast_8[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %52 = arith.muli %35, %arg9 : i32
            %53 = arith.index_cast %52 : i32 to index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%53], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %54 = memref.load %reinterpret_cast_9[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %55 = arith.extsi %arg5 : i32 to i64
            %56 = arith.muli %51, %55 : i64
            %57 = arith.index_cast %56 : i64 to index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%57], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %cast_11 = memref.cast %reinterpret_cast_10 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_12 = memref.cast %29 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_float(%cast_11, %cast_12, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %58 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_13 = memref.cast %29 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_14 = memref.cast %29 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_15 = memref.cast %58 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_13, %54, %cast_14, %cast_15) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          }
          %44 = arith.cmpi sge, %41, %c0_i64 : i64
          %45 = arith.select %44, %24, %11 : memref<512xf32, #hivm.address_space<ub>>
          scf.if %44 {
            %49 = arith.muli %35, %arg11 : i32
            %50 = arith.index_cast %49 : i32 to index
            %51 = affine.apply #map1()[%50]
            %reinterpret_cast_8 = memref.reinterpret_cast %arg10 to offset: [%51], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %52 = memref.load %reinterpret_cast_8[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %53 = arith.muli %35, %arg9 : i32
            %54 = arith.index_cast %53 : i32 to index
            %55 = affine.apply #map1()[%54]
            %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%55], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %56 = memref.load %reinterpret_cast_9[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %57 = arith.extsi %arg5 : i32 to i64
            %58 = arith.muli %52, %57 : i64
            %59 = arith.index_cast %58 : i64 to index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %cast_11 = memref.cast %reinterpret_cast_10 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_12 = memref.cast %24 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_float(%cast_11, %cast_12, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %60 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_13 = memref.cast %24 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_14 = memref.cast %24 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_15 = memref.cast %60 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_13, %56, %cast_14, %cast_15) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          }
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %34]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %46 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_1 = memref.cast %43 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_2 = memref.cast %45 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_3 = memref.cast %19 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_4 = memref.cast %46 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vadd_1d_float(%cast_1, %cast_2, %cast_3, %cast_4) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %34]
          %47 = arith.muli %35, %arg13 : i32
          %48 = arith.index_cast %47 : i32 to index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg12 to offset: [%48], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_6 = memref.cast %19 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_7 = memref.cast %reinterpret_cast_5 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_6, %cast_7, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %34]
        } {pipeline.veconly}
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
