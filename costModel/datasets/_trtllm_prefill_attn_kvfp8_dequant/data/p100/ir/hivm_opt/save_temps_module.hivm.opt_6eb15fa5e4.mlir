#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 2, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0] -> (s0 mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_trtllm_prefill_attn_kvfp8_dequant_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func @_trtllm_prefill_attn_kvfp8_dequant_infer_sync_block_lock_num_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_num_function>} {
    %c16_i64 = arith.constant 16 : i64
    return %c16_i64 : i64
  }
  func.func @_trtllm_prefill_attn_kvfp8_dequant_infer_sync_block_lock_init_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_init_function>} {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  func.func private @broadcast_scalar_int64_t_to_1d(i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int32_t_to_int64_t_1d_with_mode(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_ne_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vand_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmps_eq_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_half(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_int8_t_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_uint8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_float_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @sync_block_lock(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @sync_block_unlock(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @free_lock_var(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8192 = arith.constant 8192 : index
    %c16384_i64 = arith.constant 16384 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c64 = arith.constant 64 : index
    %c96_i64 = arith.constant 96 : i64
    %c131264_i64 = arith.constant 131264 : i64
    %c131296_i64 = arith.constant 131296 : i64
    %c131328_i64 = arith.constant 131328 : i64
    %c128_i64 = arith.constant 128 : i64
    %c131360_i64 = arith.constant 131360 : i64
    %c32896_i64 = arith.constant 32896 : i64
    %c164128_i64 = arith.constant 164128 : i64
    %c65664_i64 = arith.constant 65664 : i64
    %c98432_i64 = arith.constant 98432 : i64
    %c98496_i64 = arith.constant 98496 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = hivm.hir.pointer_cast(%c32_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c32_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c64_i64) : memref<1xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.divsi %5, %arg11 : i32
    %7 = arith.remsi %6, %arg10 : i32
    %8 = arith.muli %arg11, %arg10 : i32
    %9 = arith.divsi %5, %8 : i32
    %10 = arith.remsi %9, %arg9 : i32
    %11 = hivm.hir.pointer_cast(%c32_i64) : memref<2xi64, #hivm.address_space<ub>>
    %cast = memref.cast %11 : memref<2xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_int64_t_to_1d(%c0_i64, %cast) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %12 = arith.extsi %10 : i32 to i64
    %13 = arith.extsi %7 : i32 to i64
    %14 = hivm.hir.pointer_cast(%c96_i64) : memref<2xi32, #hivm.address_space<ub>>
    %cast_0 = memref.cast %14 : memref<2xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_0, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %15 = hivm.hir.pointer_cast(%c131264_i64) : memref<2xi64, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_1 = memref.cast %14 : memref<2xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_2 = memref.cast %15 : memref<2xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_1, %cast_2, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %16 = arith.muli %13, %c2_i64 : i64
    %17 = hivm.hir.pointer_cast(%c131264_i64) : memref<2xi64, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %15[%arg12] : memref<2xi64, #hivm.address_space<ub>>
      %68 = arith.addi %67, %16 : i64
      memref.store %68, %17[%arg12] : memref<2xi64, #hivm.address_space<ub>>
    }
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = hivm.hir.pointer_cast(%c64_i64) : memref<2xi64, #hivm.address_space<ub>>
    %cast_3 = memref.cast %19 : memref<2xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_int64_t_to_1d(%18, %cast_3) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %20 = hivm.hir.pointer_cast(%c131296_i64) : memref<2xi1, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c64_i64) : memref<2xi8, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %17[%arg12] : memref<2xi64, #hivm.address_space<ub>>
      %68 = memref.load %19[%arg12] : memref<2xi64, #hivm.address_space<ub>>
      %69 = arith.cmpi slt, %67, %68 : i64
      %70 = arith.extui %69 : i1 to i8
      memref.store %70, %21[%arg12] : memref<2xi8, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %22 = hivm.hir.pointer_cast(%c131296_i64) : memref<2xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %cast_4 = memref.cast %21 : memref<2xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_5 = memref.cast %22 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int8_t_to_half_1d_with_mode(%cast_4, %cast_5, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %23 = hivm.hir.pointer_cast(%c96_i64) : memref<2xf16, #hivm.address_space<ub>>
    %cast_6 = memref.cast %23 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_half_to_1d(%cst, %cast_6) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_7 = memref.cast %22 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_8 = memref.cast %23 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %20 : memref<2xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_ne_1d_half(%cast_7, %cast_8, %cast_9) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %24 = arith.muli %12, %18 : i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = arith.index_cast %16 : i64 to index
    %27 = affine.apply #map()[%25, %26]
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [2], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<2xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %28 = hivm.hir.pointer_cast(%c0_i64) : memref<2xi32, #hivm.address_space<ub>>
    %29 = arith.index_cast %arg5 : i32 to index
    %30 = affine.max #map1()[%26, %29]
    %31 = affine.min #map2()[%30, %26]
    %32 = affine.apply #map3()[%31, %26]
    %33 = arith.cmpi slt, %32, %c2 : index
    %subview = memref.subview %reinterpret_cast[0] [%32] [1] : memref<2xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %28[0] [%32] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %33 {
      %cast_88 = memref.cast %28 : memref<2xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%c0_i32, %cast_88) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %cast_11 = memref.cast %subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_12 = memref.cast %subview_10 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t(%cast_11, %cast_12, %c2_i32, %c0_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %34 = hivm.hir.pointer_cast(%c131328_i64) : memref<2xi64, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %cast_13 = memref.cast %28 : memref<2xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_14 = memref.cast %34 : memref<2xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_13, %cast_14, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %35 = hivm.hir.pointer_cast(%c128_i64) : memref<2xi1, #hivm.address_space<ub>>
    %36 = hivm.hir.pointer_cast(%c32_i64) : memref<2xi8, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %34[%arg12] : memref<2xi64, #hivm.address_space<ub>>
      %68 = memref.load %11[%arg12] : memref<2xi64, #hivm.address_space<ub>>
      %69 = arith.cmpi sgt, %67, %68 : i64
      %70 = arith.extui %69 : i1 to i8
      memref.store %70, %36[%arg12] : memref<2xi8, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %37 = hivm.hir.pointer_cast(%c96_i64) : memref<2xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %cast_15 = memref.cast %36 : memref<2xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_16 = memref.cast %37 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int8_t_to_half_1d_with_mode(%cast_15, %cast_16, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %38 = hivm.hir.pointer_cast(%c128_i64) : memref<2xf16, #hivm.address_space<ub>>
    %cast_17 = memref.cast %38 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_half_to_1d(%cst, %cast_17) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_18 = memref.cast %37 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_19 = memref.cast %38 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_20 = memref.cast %35 : memref<2xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_ne_1d_half(%cast_18, %cast_19, %cast_20) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %39 = hivm.hir.pointer_cast(%c131296_i64) : memref<2xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %40 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_21 = memref.cast %20 : memref<2xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_22 = memref.cast %35 : memref<2xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_23 = memref.cast %39 : memref<2xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_24 = memref.cast %40 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vand_1d_bool(%cast_21, %cast_22, %cast_23, %cast_24) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %expand_shape = memref.expand_shape %34 [[0, 1]] output_shape [2, 1] : memref<2xi64, #hivm.address_space<ub>> into memref<2x1xi64, #hivm.address_space<ub>>
    %expand_shape_25 = memref.expand_shape %17 [[0, 1]] output_shape [2, 1] : memref<2xi64, #hivm.address_space<ub>> into memref<2x1xi64, #hivm.address_space<ub>>
    %expand_shape_26 = memref.expand_shape %39 [[0, 1]] output_shape [2, 1] : memref<2xi1, #hivm.address_space<ub>> into memref<2x1xi1, #hivm.address_space<ub>>
    %41 = hivm.hir.pointer_cast(%c131360_i64) : memref<8192xi32, #hivm.address_space<ub>>
    %cast_27 = memref.cast %41 : memref<8192xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_27, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %expand_shape_28 = memref.expand_shape %41 [[0, 1]] output_shape [1, 8192] : memref<8192xi32, #hivm.address_space<ub>> into memref<1x8192xi32, #hivm.address_space<ub>>
    %reinterpret_cast_29 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
    %42 = memref.load %reinterpret_cast_29[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_30 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
    %43 = memref.load %reinterpret_cast_30[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
    %44 = hivm.hir.pointer_cast(%c96_i64) : memref<2xf16, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c128_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_31 = memref.cast %39 : memref<2xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_32 = memref.cast %44 : memref<2xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_33 = memref.cast %45 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_bool_to_half_1d_with_temp(%cast_31, %cast_32, %cast_33, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %expand_shape_34 = memref.expand_shape %44 [[0, 1]] output_shape [2, 1] : memref<2xf16, #hivm.address_space<ub>> into memref<2x1xf16, #hivm.address_space<ub>>
    %46 = hivm.hir.pointer_cast(%c128_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c32896_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_35 = memref.cast %expand_shape_34 : memref<2x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_36 = memref.cast %46 : memref<2x8192xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_37 = memref.cast %47 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_last_axis_align_2d_half(%cast_35, %cast_36, %cast_37) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %48 = hivm.hir.pointer_cast(%c128_i64) : memref<2x8192xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %46 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    %collapse_shape_38 = memref.collapse_shape %48 [[0, 1]] : memref<2x8192xi1, #hivm.address_space<ub>> into memref<16384xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_39 = memref.cast %collapse_shape : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_40 = memref.cast %collapse_shape_38 : memref<16384xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmps_eq_1d_half(%cast_39, %cst, %cast_40) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %49 = hivm.hir.pointer_cast(%c128_i64) : memref<2x8192xi1, #hivm.address_space<ub>>
    %collapse_shape_41 = memref.collapse_shape %49 [[0, 1]] : memref<2x8192xi1, #hivm.address_space<ub>> into memref<16384xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %50 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_42 = memref.cast %collapse_shape_38 : memref<16384xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_43 = memref.cast %collapse_shape_41 : memref<16384xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_44 = memref.cast %50 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_42, %cast_43, %cast_44) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %51 = hivm.hir.pointer_cast(%c164128_i64) : memref<2x8192xi8, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %expand_shape[%arg12, %c0] {DiscreteMemAccess} : memref<2x1xi64, #hivm.address_space<ub>>
      %68 = arith.muli %67, %c16384_i64 : i64
      scf.for %arg13 = %c0 to %c8192 step %c1 {
        %69 = memref.load %expand_shape_28[%c0, %arg13] {DiscreteMemAccess} : memref<1x8192xi32, #hivm.address_space<ub>>
        %70 = arith.extsi %69 : i32 to i64
        %71 = arith.addi %68, %70 : i64
        %72 = arith.index_cast %71 : i64 to index
        %reinterpret_cast_88 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [1], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %73 = memref.load %reinterpret_cast_88[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %73, %51[%arg12, %arg13] : memref<2x8192xi8, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
    } {ExtractedLoadOrStore}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %52 = hivm.hir.pointer_cast(%c32896_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %51 [[0, 1]] : memref<2x8192xi8, #hivm.address_space<ub>> into memref<16384xi8, #hivm.address_space<ub>>
    %collapse_shape_46 = memref.collapse_shape %52 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %cast_47 = memref.cast %collapse_shape_45 : memref<16384xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_48 = memref.cast %collapse_shape_46 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int8_t_to_half_1d_with_mode(%cast_47, %cast_48, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
    %53 = hivm.hir.pointer_cast(%c65664_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %collapse_shape_49 = memref.collapse_shape %53 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c98432_i64) : memref<32xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_50 = memref.cast %collapse_shape_41 : memref<16384xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_51 = memref.cast %collapse_shape_46 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_52 = memref.cast %collapse_shape_49 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_53 = memref.cast %54 : memref<32xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vsel_vs_1d_half(%cast_50, %cast_51, %cst, %cast_52, %cast_53) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %55 = hivm.hir.pointer_cast(%c65664_i64) : memref<2x8192xi8, #hivm.address_space<ub>>
    %collapse_shape_54 = memref.collapse_shape %55 [[0, 1]] : memref<2x8192xi8, #hivm.address_space<ub>> into memref<16384xi8, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_55 = memref.cast %collapse_shape_49 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_56 = memref.cast %collapse_shape_54 : memref<16384xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_half_to_int8_t_1d_with_mode(%cast_55, %cast_56, %c4_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %56 = hivm.hir.pointer_cast(%c32896_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %collapse_shape_57 = memref.collapse_shape %56 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_58 = memref.cast %collapse_shape_54 : memref<16384xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_59 = memref.cast %collapse_shape_57 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_uint8_t_to_half_1d_with_mode(%cast_58, %cast_59, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %57 = hivm.hir.pointer_cast(%c65664_i64) : memref<2x8192xf32, #hivm.address_space<ub>>
    %collapse_shape_60 = memref.collapse_shape %57 [[0, 1]] : memref<2x8192xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_61 = memref.cast %collapse_shape_57 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_62 = memref.cast %collapse_shape_60 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_half_to_float_1d_with_mode(%cast_61, %cast_62, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %view = memref.view %arg1[%c64][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
    %cast_63 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_lock(%cast_63) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %58 = hivm.hir.pointer_cast(%c164128_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
    memref.store %42, %58[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %expand_shape_25[%arg12, %c0] {DiscreteMemAccess} : memref<2x1xi64, #hivm.address_space<ub>>
      %68 = arith.addi %24, %67 : i64
      %69 = arith.addi %68, %c1_i64 : i64
      %70 = arith.muli %69, %c16384_i64 : i64
      %71 = memref.load %expand_shape_26[%arg12, %c0] {DiscreteMemAccess} : memref<2x1xi1, #hivm.address_space<ub>>
      scf.for %arg13 = %c0 to %c8192 step %c1 {
        %72 = affine.apply #map4()[%arg13]
        %73 = arith.index_cast %72 : index to i64
        %74 = arith.cmpi eq, %73, %c1_i64 : i64
        %75 = arith.select %74, %1, %0 : memref<1xbf16, #hivm.address_space<ub>>
        %76 = memref.load %expand_shape_28[%c0, %arg13] {DiscreteMemAccess} : memref<1x8192xi32, #hivm.address_space<ub>>
        %77 = arith.extsi %76 : i32 to i64
        %78 = arith.addi %70, %77 : i64
        %79 = memref.load %57[%arg12, %arg13] {DiscreteMemAccess} : memref<2x8192xf32, #hivm.address_space<ub>>
        %80 = memref.load %58[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %81 = arith.mulf %79, %80 : f32
        %82 = hivm.hir.pointer_cast(%c32_i64) : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
        memref.store %81, %82[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %83 = hivm.hir.pointer_cast(%c32_i64) : memref<1xbf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_88 = memref.cast %82 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_89 = memref.cast %83 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_88, %cast_89, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %84 = memref.load %83[%c0] : memref<1xbf16, #hivm.address_space<ub>>
        scf.if %71 {
          memref.store %84, %75[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %85 = arith.index_cast %78 : i64 to index
          %reinterpret_cast_90 = memref.reinterpret_cast %arg6 to offset: [%85], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_91 = memref.cast %75 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_92 = memref.cast %reinterpret_cast_90 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_91, %cast_92, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      } {ExtractedLoadOrStore, pipeline.veconly}
    } {ExtractedLoadOrStore}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    %cast_64 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_unlock(%cast_64) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %59 = hivm.hir.pointer_cast(%c164128_i64) : memref<2x8192xi8, #hivm.address_space<ub>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %expand_shape[%arg12, %c0] {DiscreteMemAccess} : memref<2x1xi64, #hivm.address_space<ub>>
      %68 = arith.muli %67, %c16384_i64 : i64
      scf.for %arg13 = %c0 to %c8192 step %c1 {
        %69 = memref.load %expand_shape_28[%c0, %arg13] {DiscreteMemAccess} : memref<1x8192xi32, #hivm.address_space<ub>>
        %70 = arith.extsi %69 : i32 to i64
        %71 = arith.addi %68, %70 : i64
        %72 = arith.addi %71, %c8192_i64 : i64
        %73 = arith.index_cast %72 : i64 to index
        %reinterpret_cast_88 = memref.reinterpret_cast %arg3 to offset: [%73], sizes: [1], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %74 = memref.load %reinterpret_cast_88[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %74, %59[%arg12, %arg13] : memref<2x8192xi8, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
    } {ExtractedLoadOrStore}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %60 = hivm.hir.pointer_cast(%c32896_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %collapse_shape_65 = memref.collapse_shape %59 [[0, 1]] : memref<2x8192xi8, #hivm.address_space<ub>> into memref<16384xi8, #hivm.address_space<ub>>
    %collapse_shape_66 = memref.collapse_shape %60 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %cast_67 = memref.cast %collapse_shape_65 : memref<16384xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_68 = memref.cast %collapse_shape_66 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int8_t_to_half_1d_with_mode(%cast_67, %cast_68, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %61 = hivm.hir.pointer_cast(%c65664_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %collapse_shape_69 = memref.collapse_shape %61 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c98432_i64) : memref<32xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_70 = memref.cast %collapse_shape_41 : memref<16384xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_71 = memref.cast %collapse_shape_66 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_72 = memref.cast %collapse_shape_69 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_73 = memref.cast %62 : memref<32xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vsel_vs_1d_half(%cast_70, %cast_71, %cst, %cast_72, %cast_73) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %63 = hivm.hir.pointer_cast(%c65664_i64) : memref<2x8192xi8, #hivm.address_space<ub>>
    %collapse_shape_74 = memref.collapse_shape %63 [[0, 1]] : memref<2x8192xi8, #hivm.address_space<ub>> into memref<16384xi8, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_75 = memref.cast %collapse_shape_69 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_76 = memref.cast %collapse_shape_74 : memref<16384xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_half_to_int8_t_1d_with_mode(%cast_75, %cast_76, %c4_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %64 = hivm.hir.pointer_cast(%c98496_i64) : memref<2x8192xf16, #hivm.address_space<ub>>
    %collapse_shape_77 = memref.collapse_shape %64 [[0, 1]] : memref<2x8192xf16, #hivm.address_space<ub>> into memref<16384xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_78 = memref.cast %collapse_shape_74 : memref<16384xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_79 = memref.cast %collapse_shape_77 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_uint8_t_to_half_1d_with_mode(%cast_78, %cast_79, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %65 = hivm.hir.pointer_cast(%c96_i64) : memref<2x8192xf32, #hivm.address_space<ub>>
    %collapse_shape_80 = memref.collapse_shape %65 [[0, 1]] : memref<2x8192xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_81 = memref.cast %collapse_shape_77 : memref<16384xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_82 = memref.cast %collapse_shape_80 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_half_to_float_1d_with_mode(%cast_81, %cast_82, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %view_83 = memref.view %arg1[%c0][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
    %cast_84 = memref.cast %view_83 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_lock(%cast_84) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %66 = hivm.hir.pointer_cast(%c131328_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %43, %66[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %67 = memref.load %expand_shape_25[%arg12, %c0] {DiscreteMemAccess} : memref<2x1xi64, #hivm.address_space<ub>>
      %68 = arith.addi %24, %67 : i64
      %69 = arith.addi %68, %c1_i64 : i64
      %70 = arith.muli %69, %c16384_i64 : i64
      %71 = memref.load %expand_shape_26[%arg12, %c0] {DiscreteMemAccess} : memref<2x1xi1, #hivm.address_space<ub>>
      scf.for %arg13 = %c0 to %c8192 step %c1 {
        %72 = affine.apply #map4()[%arg13]
        %73 = arith.index_cast %72 : index to i64
        %74 = arith.cmpi eq, %73, %c1_i64 : i64
        %75 = arith.select %74, %3, %2 : memref<1xbf16, #hivm.address_space<ub>>
        %76 = memref.load %expand_shape_28[%c0, %arg13] {DiscreteMemAccess} : memref<1x8192xi32, #hivm.address_space<ub>>
        %77 = arith.extsi %76 : i32 to i64
        %78 = arith.addi %70, %77 : i64
        %79 = arith.addi %78, %c8192_i64 : i64
        %80 = memref.load %65[%arg12, %arg13] {DiscreteMemAccess} : memref<2x8192xf32, #hivm.address_space<ub>>
        %81 = memref.load %66[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %82 = arith.mulf %80, %81 : f32
        %83 = hivm.hir.pointer_cast(%c32_i64) : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
        memref.store %82, %83[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %84 = hivm.hir.pointer_cast(%c32_i64) : memref<1xbf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_88 = memref.cast %83 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_89 = memref.cast %84 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_88, %cast_89, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %85 = memref.load %84[%c0] : memref<1xbf16, #hivm.address_space<ub>>
        scf.if %71 {
          memref.store %85, %75[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %86 = arith.index_cast %79 : i64 to index
          %reinterpret_cast_90 = memref.reinterpret_cast %arg6 to offset: [%86], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_91 = memref.cast %75 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_92 = memref.cast %reinterpret_cast_90 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_91, %cast_92, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      } {ExtractedLoadOrStore, pipeline.veconly}
    } {ExtractedLoadOrStore}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    %cast_85 = memref.cast %view_83 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_unlock(%cast_85) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %cast_86 = memref.cast %view_83 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @free_lock_var(%cast_86) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %cast_87 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @free_lock_var(%cast_87) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    return
  }
}
