#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 8, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
#map5 = affine_map<()[s0] -> (512, s0)>
#map6 = affine_map<()[s0, s1] -> (s1 + 512, s0)>
#map7 = affine_map<()[s0, s1] -> (512, s0 - s1)>
#map8 = affine_map<()[s0, s1, s2] -> ((s0 + s2 * s1) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @reshape_and_cache_flash_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vand_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_int32_t(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmps_eq_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @reshape_and_cache_flash_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: f32, %arg15: f32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<20xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c2144_i64 = arith.constant 2144 : i64
    %c2208_i64 = arith.constant 2208 : i64
    %c2240_i64 = arith.constant 2240 : i64
    %c2272_i64 = arith.constant 2272 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c2368_i64 = arith.constant 2368 : i64
    %c4416_i64 = arith.constant 4416 : i64
    %c6464_i64 = arith.constant 6464 : i64
    %c6496_i64 = arith.constant 6496 : i64
    %c6592_i64 = arith.constant 6592 : i64
    %c14784_i64 = arith.constant 14784 : i64
    %c15040_i64 = arith.constant 15040 : i64
    %c16064_i64 = arith.constant 16064 : i64
    %c16160_i64 = arith.constant 16160 : i64
    %c40736_i64 = arith.constant 40736 : i64
    %c57120_i64 = arith.constant 57120 : i64
    %c73568_i64 = arith.constant 73568 : i64
    %c89952_i64 = arith.constant 89952 : i64
    %c1_i64 = arith.constant 1 : i64
    %c122912_i64 = arith.constant 122912 : i64
    %c90048_i64 = arith.constant 90048 : i64
    %c122880_i64 = arith.constant 122880 : i64
    %c90016_i64 = arith.constant 90016 : i64
    %c90080_i64 = arith.constant 90080 : i64
    %c2176_i64 = arith.constant 2176 : i64
    %c90112_i64 = arith.constant 90112 : i64
    %c24352_i64 = arith.constant 24352 : i64
    %c57184_i64 = arith.constant 57184 : i64
    %c106496_i64 = arith.constant 106496 : i64
    %0 = hivm.hir.pointer_cast(%c57184_i64) : memref<8x512xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c106496_i64) : memref<8x512xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c24352_i64) : memref<8x512xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c90112_i64) : memref<8x512xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c2176_i64) : memref<8xi32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c90080_i64) : memref<8xi32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c90016_i64) : memref<1xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c122880_i64) : memref<1xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c90048_i64) : memref<1xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c122912_i64) : memref<1xf32, #hivm.address_space<ub>>
    %10 = arith.muli %arg17, %arg18 : i32
    %11 = arith.muli %10, %arg19 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg19, %arg18 : i32
    %15 = hivm.hir.pointer_cast(%c0_i64) : memref<8xi32, #hivm.address_space<ub>>
    %cast = memref.cast %15 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_int32_t_to_1d(%c0_i32, %cast) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %16 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi32, #hivm.address_space<ub>>
    %cast_1 = memref.cast %16 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_1, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %17 = arith.index_cast %arg16 : i32 to index
    %18 = hivm.hir.pointer_cast(%c64_i64) : memref<512xi32, #hivm.address_space<ub>>
    %cast_2 = memref.cast %18 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_2, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    scf.for %arg20 = %13 to %11 step %c48_i32  : i32 {
      %19 = arith.index_cast %arg20 : i32 to index
      %20 = arith.index_cast %13 : i32 to index
      %21 = affine.apply #map()[%19, %20]
      %22 = arith.index_cast %21 : index to i64
      %23 = arith.cmpi eq, %22, %c1_i64 : i64
      %24 = arith.select %23, %1, %0 : memref<8x512xf32, #hivm.address_space<ub>>
      %25 = arith.index_cast %arg20 : i32 to index
      %26 = arith.index_cast %13 : i32 to index
      %27 = affine.apply #map()[%25, %26]
      %28 = arith.index_cast %27 : index to i64
      %29 = arith.cmpi eq, %28, %c1_i64 : i64
      %30 = arith.select %29, %3, %2 : memref<8x512xf32, #hivm.address_space<ub>>
      %31 = arith.index_cast %arg20 : i32 to index
      %32 = arith.index_cast %13 : i32 to index
      %33 = affine.apply #map()[%31, %32]
      %34 = arith.index_cast %33 : index to i64
      %35 = arith.cmpi eq, %34, %c1_i64 : i64
      %36 = arith.select %35, %5, %4 : memref<8xi32, #hivm.address_space<ub>>
      %37 = arith.index_cast %arg20 : i32 to index
      %38 = arith.index_cast %13 : i32 to index
      %39 = affine.apply #map()[%37, %38]
      %40 = arith.index_cast %39 : index to i1
      %41 = arith.xori %40, %true : i1
      %42 = arith.extui %41 : i1 to i64
      hivm.hir.set_mask_norm
      %43 = arith.divsi %arg20, %arg19 : i32
      %44 = arith.remsi %43, %arg18 : i32
      %45 = arith.divsi %arg20, %14 : i32
      %46 = arith.remsi %45, %arg17 : i32
      %47 = arith.muli %46, %c8_i32 : i32
      %48 = hivm.hir.pointer_cast(%c2112_i64) : memref<8xi32, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_3 = memref.cast %16 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_4 = memref.cast %48 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %49 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_3, %47, %cast_4, %cast_5) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %50 = hivm.hir.pointer_cast(%c2144_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %51 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_6 = memref.cast %48 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %50 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %51 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_6, %arg16, %cast_7, %cast_8) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %52 = hivm.hir.pointer_cast(%c2144_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_9 = memref.cast %50 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %48 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %52 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_9, %cast_10, %cast_11) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %53 = hivm.hir.pointer_cast(%c2144_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %54 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_12 = memref.cast %52 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %53 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %54 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_12, %cast_13, %cast_14) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %55 = arith.index_cast %47 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%55], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %56 = affine.max #map1()[%55, %17]
      %57 = affine.min #map2()[%56, %55]
      %58 = affine.apply #map3()[%57, %55]
      %59 = arith.cmpi slt, %58, %c8 : index
      %subview = memref.subview %reinterpret_cast[0] [%58] [1] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_15 = memref.subview %36[0] [%58] [1] : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %59 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
        %cast_94 = memref.cast %36 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_int32_t_to_1d(%c-1_i32, %cast_94) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %42]
      %cast_16 = memref.cast %subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_17 = memref.cast %subview_15 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int32_t(%cast_16, %cast_17, %c2_i32, %c-1_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %60 = hivm.hir.pointer_cast(%c2208_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %61 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_18 = memref.cast %36 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %60 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %61 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_18, %c0_i32, %cast_19, %cast_20) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %62 = hivm.hir.pointer_cast(%c2208_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_21 = memref.cast %60 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %36 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %62 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_21, %cast_22, %cast_23) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %63 = hivm.hir.pointer_cast(%c2144_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %64 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_24 = memref.cast %53 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_25 = memref.cast %62 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %63 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %64 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vand_1d_bool(%cast_24, %cast_25, %cast_26, %cast_27) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_28 = memref.subview %48[0] [%58] [1] : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %65 = hivm.hir.pointer_cast(%c2240_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %cast_29 = memref.cast %15 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %65 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_int32_t(%cast_29, %cast_30, %c0_i32, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32) -> ()
      %subview_31 = memref.subview %65[0] [%58] [1] : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_32 = memref.cast %subview_28 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_33 = memref.cast %subview_31 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_int32_t(%cast_32, %cast_33, %c0_i32, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %66 = hivm.hir.pointer_cast(%c2272_i64) : memref<8xi32, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c2304_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %cast_34 = memref.cast %63 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %36 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %66 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %67 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_int32_t(%cast_34, %cast_35, %c0_i32, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %68 = arith.muli %44, %c512_i32 : i32
      %69 = hivm.hir.pointer_cast(%c2368_i64) : memref<512xi32, #hivm.address_space<ub>>
      %70 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_38 = memref.cast %18 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_39 = memref.cast %69 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %70 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_38, %68, %cast_39, %cast_40) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %71 = hivm.hir.pointer_cast(%c4416_i64) : memref<512xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %72 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_41 = memref.cast %69 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %71 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %72 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_41, %c512_i32, %cast_42, %cast_43) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %73 = hivm.hir.pointer_cast(%c2368_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_44 = memref.cast %71 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %69 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %73 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_44, %cast_45, %cast_46) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %74 = hivm.hir.pointer_cast(%c2368_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %75 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_47 = memref.cast %73 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %74 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %75 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_47, %cast_48, %cast_49) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %76 = hivm.hir.pointer_cast(%c6464_i64) : memref<8xf16, #hivm.address_space<ub>>
      %77 = hivm.hir.pointer_cast(%c6496_i64) : memref<48xf16, #hivm.address_space<ub>>
      %cast_50 = memref.cast %63 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_51 = memref.cast %76 : memref<8xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_52 = memref.cast %77 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_half_1d_with_temp(%cast_50, %cast_51, %cast_52, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %expand_shape = memref.expand_shape %76 [[0, 1]] output_shape [8, 1] : memref<8xf16, #hivm.address_space<ub>> into memref<8x1xf16, #hivm.address_space<ub>>
      %78 = hivm.hir.pointer_cast(%c6592_i64) : memref<8x512xf16, #hivm.address_space<ub>>
      %79 = hivm.hir.pointer_cast(%c14784_i64) : memref<128xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_53 = memref.cast %expand_shape : memref<8x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_54 = memref.cast %78 : memref<8x512xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %79 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_last_axis_align_2d_half(%cast_53, %cast_54, %cast_55) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %80 = hivm.hir.pointer_cast(%c6592_i64) : memref<8x512xi1, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %78 [[0, 1]] : memref<8x512xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
      %collapse_shape_56 = memref.collapse_shape %80 [[0, 1]] : memref<8x512xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_57 = memref.cast %collapse_shape : memref<4096xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_58 = memref.cast %collapse_shape_56 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmps_eq_1d_half(%cast_57, %cst, %cast_58) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %81 = hivm.hir.pointer_cast(%c6592_i64) : memref<8x512xi1, #hivm.address_space<ub>>
      %collapse_shape_59 = memref.collapse_shape %81 [[0, 1]] : memref<8x512xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %82 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_60 = memref.cast %collapse_shape_56 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %collapse_shape_59 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_62 = memref.cast %82 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_60, %cast_61, %cast_62) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %83 = hivm.hir.pointer_cast(%c15040_i64) : memref<512xf16, #hivm.address_space<ub>>
      %84 = hivm.hir.pointer_cast(%c16064_i64) : memref<48xf16, #hivm.address_space<ub>>
      %cast_63 = memref.cast %74 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_64 = memref.cast %83 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_65 = memref.cast %84 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_half_1d_with_temp(%cast_63, %cast_64, %cast_65, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %expand_shape_66 = memref.expand_shape %83 [[0, 1]] output_shape [1, 512] : memref<512xf16, #hivm.address_space<ub>> into memref<1x512xf16, #hivm.address_space<ub>>
      %85 = hivm.hir.pointer_cast(%c16160_i64) : memref<8x512xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_67 = memref.cast %expand_shape_66 : memref<1x512xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_68 = memref.cast %85 : memref<8x512xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_half(%cast_67, %cast_68) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %86 = hivm.hir.pointer_cast(%c16160_i64) : memref<8x512xi1, #hivm.address_space<ub>>
      %collapse_shape_69 = memref.collapse_shape %85 [[0, 1]] : memref<8x512xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
      %collapse_shape_70 = memref.collapse_shape %86 [[0, 1]] : memref<8x512xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_71 = memref.cast %collapse_shape_69 : memref<4096xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_72 = memref.cast %collapse_shape_70 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmps_eq_1d_half(%cast_71, %cst, %cast_72) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %87 = hivm.hir.pointer_cast(%c16160_i64) : memref<8x512xi1, #hivm.address_space<ub>>
      %collapse_shape_73 = memref.collapse_shape %87 [[0, 1]] : memref<8x512xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %88 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_74 = memref.cast %collapse_shape_70 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_75 = memref.cast %collapse_shape_73 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_76 = memref.cast %88 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_74, %cast_75, %cast_76) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %89 = arith.index_cast %68 : i32 to index
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg21 = %c0 to %c8 step %c1 {
        %99 = memref.load %65[%arg21] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %100 = arith.muli %99, %arg9 : i32
        %101 = arith.index_cast %100 : i32 to index
        %102 = affine.apply #map4()[%101, %89]
        %reinterpret_cast_94 = memref.reinterpret_cast %arg3 to offset: [%102], sizes: [1, 512], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_95 = memref.subview %30[%arg21, 0] [1, 512] [1, 1] : memref<8x512xf32, #hivm.address_space<ub>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_96 = memref.collapse_shape %reinterpret_cast_94 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_97 = memref.collapse_shape %subview_95 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %cast_98 = memref.cast %collapse_shape_96 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_99 = memref.cast %collapse_shape_97 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_98, %cast_99, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      } {ExtractedLoadOrStore, hivm.parallel_loop, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %90 = hivm.hir.pointer_cast(%c6592_i64) : memref<8x512xi1, #hivm.address_space<ub>>
      %collapse_shape_77 = memref.collapse_shape %90 [[0, 1]] : memref<8x512xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %91 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_78 = memref.cast %collapse_shape_73 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_79 = memref.cast %collapse_shape_59 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_80 = memref.cast %collapse_shape_77 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_81 = memref.cast %91 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vand_1d_bool(%cast_78, %cast_79, %cast_80, %cast_81) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %92 = hivm.hir.pointer_cast(%c40736_i64) : memref<8x512xf32, #hivm.address_space<ub>>
      %collapse_shape_82 = memref.collapse_shape %30 [[0, 1]] : memref<8x512xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_83 = memref.collapse_shape %92 [[0, 1]] : memref<8x512xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %93 = hivm.hir.pointer_cast(%c57120_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_84 = memref.cast %collapse_shape_77 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %collapse_shape_82 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_86 = memref.cast %collapse_shape_83 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_87 = memref.cast %93 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_84, %cast_85, %cst_0, %cast_86, %cast_87) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
      scf.for %arg21 = %c0 to %c8 step %c1 {
        %99 = memref.load %65[%arg21] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %100 = arith.muli %99, %arg10 : i32
        %101 = arith.index_cast %100 : i32 to index
        %102 = affine.apply #map4()[%101, %89]
        %reinterpret_cast_94 = memref.reinterpret_cast %arg4 to offset: [%102], sizes: [1, 512], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_95 = memref.subview %24[%arg21, 0] [1, 512] [1, 1] : memref<8x512xf32, #hivm.address_space<ub>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_96 = memref.collapse_shape %reinterpret_cast_94 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_97 = memref.collapse_shape %subview_95 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %cast_98 = memref.cast %collapse_shape_96 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_99 = memref.cast %collapse_shape_97 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_98, %cast_99, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      } {ExtractedLoadOrStore, hivm.parallel_loop, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %94 = hivm.hir.pointer_cast(%c73568_i64) : memref<8x512xf32, #hivm.address_space<ub>>
      %collapse_shape_88 = memref.collapse_shape %24 [[0, 1]] : memref<8x512xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_89 = memref.collapse_shape %94 [[0, 1]] : memref<8x512xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %95 = hivm.hir.pointer_cast(%c89952_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_90 = memref.cast %collapse_shape_77 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_91 = memref.cast %collapse_shape_88 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_92 = memref.cast %collapse_shape_89 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_93 = memref.cast %95 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_90, %cast_91, %cst_0, %cast_92, %cast_93) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %96 = affine.max #map5()[%89]
      %97 = affine.min #map6()[%96, %89]
      %98 = affine.min #map7()[%97, %89]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
      scf.for %arg21 = %c0 to %c8 step %c1 {
        %99 = memref.load %66[%arg21] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %100 = arith.remsi %99, %arg13 : i32
        %101 = arith.muli %100, %c512_i32 : i32
        %102 = arith.index_cast %arg21 : index to i32
        %103 = arith.addi %47, %102 : i32
        %104 = arith.cmpi slt, %103, %arg16 : i32
        %105 = memref.load %36[%arg21] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %106 = arith.cmpi sge, %105, %c0_i32 : i32
        %107 = arith.andi %104, %106 : i1
        scf.for %arg22 = %c0 to %98 step %c1 {
          %108 = affine.apply #map8()[%arg22, %98, %arg21]
          %109 = arith.index_cast %108 : index to i64
          %110 = arith.cmpi eq, %109, %c1_i64 : i64
          %111 = arith.select %110, %7, %6 : memref<1xf32, #hivm.address_space<ub>>
          %112 = affine.apply #map8()[%arg22, %98, %arg21]
          %113 = arith.index_cast %112 : index to i1
          %114 = arith.xori %113, %true : i1
          %115 = arith.extui %114 : i1 to i64
          %116 = arith.divsi %99, %arg13 : i32
          %117 = arith.muli %116, %arg8 : i32
          %118 = arith.addi %117, %101 : i32
          %119 = arith.extsi %118 : i32 to i64
          %120 = arith.index_cast %arg22 : index to i32
          %121 = arith.addi %68, %120 : i32
          %122 = arith.extsi %121 : i32 to i64
          %123 = arith.addi %119, %122 : i64
          %124 = memref.load %92[%arg21, %arg22] {DiscreteMemAccess} : memref<8x512xf32, #hivm.address_space<ub>>
          scf.if %107 {
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %115]
            memref.store %124, %111[%c0] : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %125 = arith.index_cast %123 : i64 to index
            %reinterpret_cast_94 = memref.reinterpret_cast %arg5 to offset: [%125], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            %cast_95 = memref.cast %111 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_96 = memref.cast %reinterpret_cast_94 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_95, %cast_96, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %115]
          } {pipeline.veconly}
        } {ExtractedLoadOrStore, pipeline.veconly}
      } {ExtractedLoadOrStore, pipeline.veconly}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg21 = %c0 to %c8 step %c1 {
        %99 = memref.load %66[%arg21] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %100 = arith.remsi %99, %arg13 : i32
        %101 = arith.muli %100, %c512_i32 : i32
        %102 = arith.index_cast %arg21 : index to i32
        %103 = arith.addi %47, %102 : i32
        %104 = arith.cmpi slt, %103, %arg16 : i32
        %105 = memref.load %36[%arg21] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %106 = arith.cmpi sge, %105, %c0_i32 : i32
        %107 = arith.andi %104, %106 : i1
        scf.for %arg22 = %c0 to %98 step %c1 {
          %108 = affine.apply #map8()[%arg22, %98, %arg21]
          %109 = arith.index_cast %108 : index to i64
          %110 = arith.cmpi eq, %109, %c1_i64 : i64
          %111 = arith.select %110, %9, %8 : memref<1xf32, #hivm.address_space<ub>>
          %112 = affine.apply #map8()[%arg22, %98, %arg21]
          %113 = arith.index_cast %112 : index to i1
          %114 = arith.xori %113, %true : i1
          %115 = arith.extui %114 : i1 to i64
          %116 = arith.divsi %99, %arg13 : i32
          %117 = arith.muli %116, %arg8 : i32
          %118 = arith.addi %117, %101 : i32
          %119 = arith.extsi %118 : i32 to i64
          %120 = arith.index_cast %arg22 : index to i32
          %121 = arith.addi %68, %120 : i32
          %122 = arith.extsi %121 : i32 to i64
          %123 = arith.addi %119, %122 : i64
          %124 = memref.load %94[%arg21, %arg22] {DiscreteMemAccess} : memref<8x512xf32, #hivm.address_space<ub>>
          scf.if %107 {
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %115]
            memref.store %124, %111[%c0] : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %125 = arith.index_cast %123 : i64 to index
            %reinterpret_cast_94 = memref.reinterpret_cast %arg6 to offset: [%125], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            %cast_95 = memref.cast %111 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_96 = memref.cast %reinterpret_cast_94 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_95, %cast_96, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %115]
          } {pipeline.veconly}
        } {ExtractedLoadOrStore, pipeline.veconly}
      } {ExtractedLoadOrStore, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %42]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
