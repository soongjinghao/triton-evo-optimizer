#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_trtllm_prefill_attn_kvfp8_dequant_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_int8_t(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_uint8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_float_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c0 = arith.constant 0 : index
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c18432_i64 = arith.constant 18432 : i64
    %c30720_i64 = arith.constant 30720 : i64
    %c0_i64 = arith.constant 0 : i64
    %c22528_i64 = arith.constant 22528 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %0 = hivm.hir.pointer_cast(%c22528_i64) : memref<2048xbf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c32768_i64) : memref<2048xbf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<2048xi8, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c30720_i64) : memref<2048xi8, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c22528_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c32768_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c22528_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c32768_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %8 = arith.muli %arg9, %arg10 : i32
    %9 = arith.muli %8, %arg11 : i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = arith.muli %arg11, %arg10 : i32
    %13 = arith.extsi %arg5 : i32 to i64
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg12 = %11 to %9 step %c48_i32  : i32 {
      %14 = arith.index_cast %arg12 : i32 to index
      %15 = arith.index_cast %11 : i32 to index
      %16 = affine.apply #map()[%14, %15]
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.cmpi eq, %17, %c1_i64 : i64
      %19 = arith.select %18, %1, %0 : memref<2048xbf16, #hivm.address_space<ub>>
      %20 = arith.index_cast %arg12 : i32 to index
      %21 = arith.index_cast %11 : i32 to index
      %22 = affine.apply #map()[%20, %21]
      %23 = arith.index_cast %22 : index to i64
      %24 = arith.cmpi eq, %23, %c1_i64 : i64
      %25 = arith.select %24, %3, %2 : memref<2048xi8, #hivm.address_space<ub>>
      %26 = arith.index_cast %arg12 : i32 to index
      %27 = arith.index_cast %11 : i32 to index
      %28 = affine.apply #map()[%26, %27]
      %29 = arith.index_cast %28 : index to i64
      %30 = arith.cmpi eq, %29, %c1_i64 : i64
      %31 = arith.select %30, %5, %4 : memref<2048xf32, #hivm.address_space<ub>>
      %32 = arith.index_cast %arg12 : i32 to index
      %33 = arith.index_cast %11 : i32 to index
      %34 = affine.apply #map()[%32, %33]
      %35 = arith.index_cast %34 : index to i64
      %36 = arith.cmpi eq, %35, %c1_i64 : i64
      %37 = arith.select %36, %7, %6 : memref<2048xf32, #hivm.address_space<ub>>
      %38 = arith.index_cast %arg12 : i32 to index
      %39 = arith.index_cast %11 : i32 to index
      %40 = affine.apply #map()[%38, %39]
      %41 = arith.index_cast %40 : index to i1
      %42 = arith.xori %41, %true : i1
      %43 = arith.extui %42 : i1 to i64
      hivm.hir.set_mask_norm
      %44 = arith.divsi %arg12, %arg11 : i32
      %45 = arith.remsi %44, %arg10 : i32
      %46 = arith.divsi %arg12, %12 : i32
      %47 = arith.remsi %46, %arg9 : i32
      %48 = arith.extsi %47 : i32 to i64
      %49 = arith.extsi %45 : i32 to i64
      %50 = arith.muli %48, %13 : i64
      %51 = arith.addi %50, %49 : i64
      %52 = arith.index_cast %51 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %53 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %54 = arith.extsi %53 : i32 to i64
      %55 = arith.cmpi sle, %54, %c0_i64 : i64
      scf.if %55 {
      } else {
        %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %56 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %57 = memref.load %reinterpret_cast_1[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %58 = arith.muli %54, %c2048_i64 : i64
        %59 = arith.index_cast %58 : i64 to index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [2048], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<2048xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %43]
        %cast = memref.cast %reinterpret_cast_2 : memref<2048xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_3 = memref.cast %25 : memref<2048xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int8_t(%cast, %cast_3, %c0_i32, %c0_i8, %c0) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %60 = hivm.hir.pointer_cast(%c2048_i64) : memref<2048xf32, #hivm.address_space<ub>>
        %cast_4 = memref.cast %60 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%56, %cast_4) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %61 = hivm.hir.pointer_cast(%c10240_i64) : memref<2048xf32, #hivm.address_space<ub>>
        %cast_5 = memref.cast %61 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%57, %cast_5) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %subview = memref.subview %60[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %61[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_7 = memref.cast %subview : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %subview_6 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_7, %cast_8, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %62 = hivm.hir.pointer_cast(%c18432_i64) : memref<2048xf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_9 = memref.cast %25 : memref<2048xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %62 : memref<2048xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_uint8_t_to_half_1d_with_mode(%cast_9, %cast_10, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %43]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %43]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_11 = memref.cast %62 : memref<2048xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_12 = memref.cast %31 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_half_to_float_1d_with_mode(%cast_11, %cast_12, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %63 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_13 = memref.cast %31 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_14 = memref.cast %61 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_15 = memref.cast %37 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_16 = memref.cast %63 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_1d_float(%cast_13, %cast_14, %cast_15, %cast_16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %64 = arith.addi %51, %c1_i64 : i64
        %65 = arith.muli %64, %c2048_i64 : i64
        %66 = arith.index_cast %65 : i64 to index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg6 to offset: [%66], sizes: [2048], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<2048xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_18 = memref.cast %37 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_19 = memref.cast %19 : memref<2048xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_18, %cast_19, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_20 = memref.cast %19 : memref<2048xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_21 = memref.cast %reinterpret_cast_17 : memref<2048xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_20, %cast_21, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %43]
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
