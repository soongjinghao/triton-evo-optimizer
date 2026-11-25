#map = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 4 + ((s2 - s3) floordiv 48) * ((s1 + 3) floordiv 4)) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @moe_align_block_size_stage1_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @moe_align_block_size_stage1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c1_i64 = arith.constant 1 : i64
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %c160_i64 = arith.constant 160 : i64
    %c32_i64 = arith.constant 32 : i64
    %c192_i64 = arith.constant 192 : i64
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
    %c224_i64 = arith.constant 224 : i64
    %0 = hivm.hir.pointer_cast(%c96_i64) : memref<1xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c224_i64) : memref<1xi32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c64_i64) : memref<1xi32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c192_i64) : memref<1xi32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c32_i64) : memref<1xi32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c160_i64) : memref<1xi32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c0_i64) : memref<1xi32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c128_i64) : memref<1xi32, #hivm.address_space<ub>>
    %8 = arith.muli %arg7, %arg8 : i32
    %9 = arith.muli %8, %arg9 : i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = arith.muli %arg9, %arg8 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg10 = %11 to %9 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %13 = arith.divsi %arg10, %12 : i32
      %14 = arith.remsi %13, %arg7 : i32
      %15 = arith.muli %14, %arg6 : i32
      %16 = arith.addi %14, %c1_i32 : i32
      %17 = arith.muli %16, %c4_i32 : i32
      scf.for %arg11 = %c0_i32 to %arg6 step %c4_i32  : i32 {
        %18 = arith.index_cast %arg11 : i32 to index
        %19 = arith.index_cast %arg6 : i32 to index
        %20 = arith.index_cast %arg10 : i32 to index
        %21 = arith.index_cast %11 : i32 to index
        %22 = affine.apply #map()[%18, %19, %20, %21]
        %23 = arith.index_cast %22 : index to i64
        %24 = arith.cmpi eq, %23, %c1_i64 : i64
        %25 = arith.select %24, %1, %0 : memref<1xi32, #hivm.address_space<ub>>
        %26 = arith.index_cast %arg11 : i32 to index
        %27 = arith.index_cast %arg6 : i32 to index
        %28 = arith.index_cast %arg10 : i32 to index
        %29 = arith.index_cast %11 : i32 to index
        %30 = affine.apply #map()[%26, %27, %28, %29]
        %31 = arith.index_cast %30 : index to i64
        %32 = arith.cmpi eq, %31, %c1_i64 : i64
        %33 = arith.select %32, %3, %2 : memref<1xi32, #hivm.address_space<ub>>
        %34 = arith.index_cast %arg11 : i32 to index
        %35 = arith.index_cast %arg6 : i32 to index
        %36 = arith.index_cast %arg10 : i32 to index
        %37 = arith.index_cast %11 : i32 to index
        %38 = affine.apply #map()[%34, %35, %36, %37]
        %39 = arith.index_cast %38 : index to i64
        %40 = arith.cmpi eq, %39, %c1_i64 : i64
        %41 = arith.select %40, %5, %4 : memref<1xi32, #hivm.address_space<ub>>
        %42 = arith.index_cast %arg11 : i32 to index
        %43 = arith.index_cast %arg6 : i32 to index
        %44 = arith.index_cast %arg10 : i32 to index
        %45 = arith.index_cast %11 : i32 to index
        %46 = affine.apply #map()[%42, %43, %44, %45]
        %47 = arith.index_cast %46 : index to i64
        %48 = arith.cmpi eq, %47, %c1_i64 : i64
        %49 = arith.select %48, %7, %6 : memref<1xi32, #hivm.address_space<ub>>
        %50 = arith.addi %arg11, %c1_i32 : i32
        %51 = arith.addi %arg11, %c2_i32 : i32
        %52 = arith.addi %arg11, %c3_i32 : i32
        %53 = arith.addi %15, %arg11 : i32
        %54 = arith.cmpi slt, %53, %arg5 : i32
        %55 = arith.addi %15, %50 : i32
        %56 = arith.cmpi slt, %55, %arg5 : i32
        %57 = arith.addi %15, %51 : i32
        %58 = arith.cmpi slt, %57, %arg5 : i32
        %59 = arith.addi %15, %52 : i32
        %60 = arith.cmpi slt, %59, %arg5 : i32
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
        scf.if %54 {
          %61 = arith.index_cast %15 : i32 to index
          %62 = arith.index_cast %arg11 : i32 to index
          %63 = affine.apply #map1()[%61, %62]
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %64 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %65 = arith.index_cast %17 : i32 to index
          %66 = arith.index_cast %64 : i32 to index
          %67 = affine.apply #map1()[%65, %66]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %68 = memref.load %reinterpret_cast_0[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %69 = arith.addi %68, %c1_i32 : i32
          memref.store %69, %49[%c0] : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast = memref.cast %49 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_1 = memref.cast %reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
        scf.if %56 {
          %61 = arith.index_cast %15 : i32 to index
          %62 = arith.index_cast %50 : i32 to index
          %63 = affine.apply #map1()[%61, %62]
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %64 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %65 = arith.index_cast %17 : i32 to index
          %66 = arith.index_cast %64 : i32 to index
          %67 = affine.apply #map1()[%65, %66]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %68 = memref.load %reinterpret_cast_0[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %69 = arith.addi %68, %c1_i32 : i32
          memref.store %69, %41[%c0] : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast = memref.cast %41 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_1 = memref.cast %reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
        scf.if %58 {
          %61 = arith.index_cast %15 : i32 to index
          %62 = arith.index_cast %51 : i32 to index
          %63 = affine.apply #map1()[%61, %62]
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %64 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %65 = arith.index_cast %17 : i32 to index
          %66 = arith.index_cast %64 : i32 to index
          %67 = affine.apply #map1()[%65, %66]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %68 = memref.load %reinterpret_cast_0[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %69 = arith.addi %68, %c1_i32 : i32
          memref.store %69, %33[%c0] : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast = memref.cast %33 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_1 = memref.cast %reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
        scf.if %60 {
          %61 = arith.index_cast %15 : i32 to index
          %62 = arith.index_cast %52 : i32 to index
          %63 = affine.apply #map1()[%61, %62]
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %64 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %65 = arith.index_cast %17 : i32 to index
          %66 = arith.index_cast %64 : i32 to index
          %67 = affine.apply #map1()[%65, %66]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %68 = memref.load %reinterpret_cast_0[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %69 = arith.addi %68, %c1_i32 : i32
          memref.store %69, %25[%c0] : memref<1xi32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast = memref.cast %25 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_1 = memref.cast %reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
