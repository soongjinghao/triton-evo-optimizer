#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + ((s1 - s2) floordiv 48) * 4 - ((s0 + ((s1 - s2) floordiv 48) * 4 - 1) floordiv 2) * 2 - 1)>
#map2 = affine_map<()[s0] -> (s0 + 15)>
#map3 = affine_map<()[s0] -> (s0 + 1)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @moe_align_block_size_stage3_stage4_fused_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @moe_align_block_size_stage3_stage4_fused(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg8: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c31_i32 = arith.constant 31 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c160_i64 = arith.constant 160 : i64
    %c0_i64 = arith.constant 0 : i64
    %c192_i64 = arith.constant 192 : i64
    %c32_i64 = arith.constant 32 : i64
    %c224_i64 = arith.constant 224 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i64 = arith.constant 256 : i64
    %c96_i64 = arith.constant 96 : i64
    %c128_i64 = arith.constant 128 : i64
    %c288_i64 = arith.constant 288 : i64
    %0 = hivm.hir.pointer_cast(%c128_i64) : memref<1xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c288_i64) : memref<1xi32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c96_i64) : memref<1xi32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c256_i64) : memref<1xi32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c64_i64) : memref<1xi32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c224_i64) : memref<1xi32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c32_i64) : memref<1xi32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c192_i64) : memref<1xi32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c160_i64) : memref<1xi32, #hivm.address_space<ub>>
    %10 = arith.muli %arg10, %arg11 : i32
    %11 = arith.muli %10, %arg12 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg12, %arg11 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    scf.for %arg13 = %13 to %11 step %c48_i32  : i32 {
      %15 = arith.index_cast %arg13 : i32 to index
      %16 = arith.index_cast %13 : i32 to index
      %17 = affine.apply #map()[%15, %16]
      %18 = arith.index_cast %17 : index to i64
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      %20 = arith.select %19, %1, %0 : memref<1xi32, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg13 : i32 to index
      %22 = arith.index_cast %13 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.cmpi eq, %24, %c1_i64 : i64
      %26 = arith.select %25, %3, %2 : memref<1xi32, #hivm.address_space<ub>>
      %27 = arith.index_cast %arg13 : i32 to index
      %28 = arith.index_cast %13 : i32 to index
      %29 = affine.apply #map()[%27, %28]
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.cmpi eq, %30, %c1_i64 : i64
      %32 = arith.select %31, %5, %4 : memref<1xi32, #hivm.address_space<ub>>
      %33 = arith.index_cast %arg13 : i32 to index
      %34 = arith.index_cast %13 : i32 to index
      %35 = affine.apply #map()[%33, %34]
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.cmpi eq, %36, %c1_i64 : i64
      %38 = arith.select %37, %7, %6 : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_mask_norm
      %39 = arith.divsi %arg13, %arg12 : i32
      %40 = arith.remsi %39, %arg11 : i32
      %41 = arith.divsi %arg13, %14 : i32
      %42 = arith.remsi %41, %arg10 : i32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %43 = scf.for %arg14 = %c1_i32 to %c5_i32 step %c1_i32 iter_args(%arg15 = %c0_i32) -> (i32)  : i32 {
        %51 = arith.index_cast %arg14 : i32 to index
        %52 = arith.index_cast %arg13 : i32 to index
        %53 = arith.index_cast %13 : i32 to index
        %54 = affine.apply #map1()[%51, %52, %53]
        %55 = arith.index_cast %54 : index to i64
        %56 = arith.cmpi eq, %55, %c1_i64 : i64
        %57 = arith.select %56, %9, %8 : memref<1xi32, #hivm.address_space<ub>>
        %58 = arith.index_cast %arg14 : i32 to index
        %59 = arith.index_cast %arg13 : i32 to index
        %60 = arith.index_cast %13 : i32 to index
        %61 = affine.apply #map1()[%58, %59, %60]
        %62 = arith.index_cast %61 : index to i1
        %63 = arith.select %62, %c1_i64, %c2_i64 : i64
        %64 = arith.index_cast %arg14 : i32 to index
        %65 = affine.apply #map2()[%64]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %66 = memref.load %reinterpret_cast_3[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %67 = arith.addi %66, %c31_i32 : i32
        %68 = arith.divsi %67, %c32_i32 : i32
        %69 = arith.muli %68, %c32_i32 : i32
        %70 = arith.addi %arg15, %69 : i32
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %63]
        memref.store %70, %57[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%64], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_5 = memref.cast %57 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %reinterpret_cast_4 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_5, %cast_6, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %63]
        scf.yield %70 : i32
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      memref.store %43, %38[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast = memref.cast %38 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_0 = memref.cast %reinterpret_cast : memref<1xi32, strided<[1]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_0, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      %44 = arith.index_cast %40 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %45 = memref.load %reinterpret_cast_1[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %46 = affine.apply #map3()[%44]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%46], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %47 = memref.load %reinterpret_cast_2[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.store %40, %32[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.for %arg14 = %45 to %47 step %c32_i32  : i32 {
        %51 = arith.divsi %arg14, %c32_i32 : i32
        %52 = arith.index_cast %51 : i32 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_4 = memref.cast %32 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_5 = memref.cast %reinterpret_cast_3 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_4, %cast_5, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {pipeline.veconly}
      %48 = arith.muli %42, %c4_i32 : i32
      %49 = arith.addi %48, %40 : i32
      %50 = arith.cmpi slt, %49, %arg9 : i32
      scf.if %50 {
        %51 = arith.index_cast %49 : i32 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %52 = memref.load %reinterpret_cast_3[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %53 = arith.index_cast %48 : i32 to index
        %54 = arith.index_cast %52 : i32 to index
        %55 = affine.apply #map4()[%53, %54]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%55], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %56 = memref.load %reinterpret_cast_4[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%54], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %57 = memref.load %reinterpret_cast_5[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %58 = arith.addi %56, %57 : i32
        memref.store %49, %26[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %59 = arith.index_cast %58 : i32 to index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_7 = memref.cast %26 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %reinterpret_cast_6 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_7, %cast_8, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        %60 = arith.addi %56, %c1_i32 : i32
        memref.store %60, %20[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_9 = memref.cast %20 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %reinterpret_cast_4 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_9, %cast_10, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
