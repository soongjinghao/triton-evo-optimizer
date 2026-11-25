#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 4, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0, s1] -> (4, s0 - s1)>
#map5 = affine_map<()[s0] -> (s0 mod 2)>
#map6 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map7 = affine_map<()[s0, s1] -> (s0 * s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_fwd_kernel_ep_gather_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_int64_t_to_1d(i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmps_eq_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_int64_t_to_2d(i64, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_ne_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vand_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_fwd_kernel_ep_gather(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, false, true, false, true, false, true, false, true, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c8192_i64 = arith.constant 8192 : i64
    %c8224_i64 = arith.constant 8224 : i64
    %c8256_i64 = arith.constant 8256 : i64
    %c8288_i64 = arith.constant 8288 : i64
    %c8320_i64 = arith.constant 8320 : i64
    %c8352_i64 = arith.constant 8352 : i64
    %c8448_i64 = arith.constant 8448 : i64
    %c12544_i64 = arith.constant 12544 : i64
    %c13120_i64 = arith.constant 13120 : i64
    %c13056_i64 = arith.constant 13056 : i64
    %c13088_i64 = arith.constant 13088 : i64
    %c13152_i64 = arith.constant 13152 : i64
    %c13184_i64 = arith.constant 13184 : i64
    %c13280_i64 = arith.constant 13280 : i64
    %c17376_i64 = arith.constant 17376 : i64
    %c25824_i64 = arith.constant 25824 : i64
    %c34016_i64 = arith.constant 34016 : i64
    %c34080_i64 = arith.constant 34080 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c42528_i64 = arith.constant 42528 : i64
    %c12800_i64 = arith.constant 12800 : i64
    %c42656_i64 = arith.constant 42656 : i64
    %c12928_i64 = arith.constant 12928 : i64
    %c42784_i64 = arith.constant 42784 : i64
    %c17632_i64 = arith.constant 17632 : i64
    %c34336_i64 = arith.constant 34336 : i64
    %c50976_i64 = arith.constant 50976 : i64
    %0 = hivm.hir.pointer_cast(%c34336_i64) : memref<4x512xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c50976_i64) : memref<4x512xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c17632_i64) : memref<4x512xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c42784_i64) : memref<4x512xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c12928_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c42656_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c12800_i64) : memref<4x4xi64, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c42528_i64) : memref<4x4xi64, #hivm.address_space<ub>>
    %8 = arith.muli %arg14, %arg15 : i32
    %9 = arith.muli %8, %arg16 : i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = arith.muli %arg16, %arg15 : i32
    %13 = hivm.hir.pointer_cast(%c0_i64) : memref<4x512xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %13 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %cast = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %14 = hivm.hir.pointer_cast(%c8192_i64) : memref<4xi64, #hivm.address_space<ub>>
    %cast_1 = memref.cast %14 : memref<4xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_int64_t_to_1d(%c0_i64, %cast_1) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %15 = hivm.hir.pointer_cast(%c8224_i64) : memref<4xi32, #hivm.address_space<ub>>
    %cast_2 = memref.cast %15 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_2, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.index_cast %arg11 : i32 to index
    %18 = arith.index_cast %arg3 : i32 to index
    %19 = arith.index_cast %arg9 : i32 to index
    %20 = arith.index_cast %arg13 : i32 to index
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg17 = %11 to %9 step %c48_i32  : i32 {
      %21 = arith.index_cast %arg17 : i32 to index
      %22 = arith.index_cast %11 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.cmpi eq, %24, %c1_i64 : i64
      %26 = arith.select %25, %1, %0 : memref<4x512xf32, #hivm.address_space<ub>>
      %27 = arith.index_cast %arg17 : i32 to index
      %28 = arith.index_cast %11 : i32 to index
      %29 = affine.apply #map()[%27, %28]
      %30 = arith.index_cast %29 : index to i1
      %31 = arith.xori %30, %true : i1
      %32 = arith.extui %31 : i1 to i64
      hivm.hir.set_mask_norm
      %33 = arith.divsi %arg17, %12 : i32
      %34 = arith.remsi %33, %arg14 : i32
      %35 = arith.muli %34, %c4_i32 : i32
      %36 = hivm.hir.pointer_cast(%c8256_i64) : memref<4xi32, #hivm.address_space<ub>>
      %37 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_3 = memref.cast %15 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_4 = memref.cast %36 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %37 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_3, %35, %cast_4, %cast_5) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %38 = hivm.hir.pointer_cast(%c8288_i64) : memref<4xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %39 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_6 = memref.cast %36 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %38 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %39 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_6, %arg3, %cast_7, %cast_8) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %40 = hivm.hir.pointer_cast(%c8256_i64) : memref<4xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_9 = memref.cast %38 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %36 : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %40 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_9, %cast_10, %cast_11) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %41 = hivm.hir.pointer_cast(%c8256_i64) : memref<4xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %42 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_12 = memref.cast %40 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %41 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %42 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_12, %cast_13, %cast_14) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %43 = arith.index_cast %35 : i32 to index
      %44 = affine.max #map1()[%43, %18]
      %45 = affine.min #map2()[%44, %43]
      %46 = affine.apply #map3()[%45, %43]
      %47 = arith.cmpi slt, %46, %c4 : index
      %48 = hivm.hir.pointer_cast(%c8320_i64) : memref<4xf16, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c8352_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_15 = memref.cast %41 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %48 : memref<4xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %49 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_half_1d_with_temp(%cast_15, %cast_16, %cast_17, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %expand_shape = memref.expand_shape %48 [[0, 1]] output_shape [4, 1] : memref<4xf16, #hivm.address_space<ub>> into memref<4x1xf16, #hivm.address_space<ub>>
      %50 = hivm.hir.pointer_cast(%c8448_i64) : memref<4x512xf16, #hivm.address_space<ub>>
      %51 = hivm.hir.pointer_cast(%c12544_i64) : memref<128xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_18 = memref.cast %expand_shape : memref<4x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %50 : memref<4x512xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %51 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_last_axis_align_2d_half(%cast_18, %cast_19, %cast_20) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %52 = hivm.hir.pointer_cast(%c8448_i64) : memref<4x512xi1, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %50 [[0, 1]] : memref<4x512xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %52 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_23 = memref.cast %collapse_shape_21 : memref<2048xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_24 = memref.cast %collapse_shape_22 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmps_eq_1d_half(%cast_23, %cst, %cast_24) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %53 = hivm.hir.pointer_cast(%c8448_i64) : memref<4x512xi1, #hivm.address_space<ub>>
      %collapse_shape_25 = memref.collapse_shape %53 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %54 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_26 = memref.cast %collapse_shape_22 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %collapse_shape_25 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_28 = memref.cast %54 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_26, %cast_27, %cast_28) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %55 = affine.min #map4()[%45, %43]
      %collapse_shape_29 = memref.collapse_shape %26 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %32]
      %cast_30 = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %collapse_shape_29 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_30, %cast_31, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %56 = scf.for %arg18 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg19 = %26) -> (memref<4x512xf32, #hivm.address_space<ub>>)  : i32 {
        %58 = arith.index_cast %arg18 : i32 to index
        %59 = affine.apply #map5()[%58]
        %60 = arith.index_cast %59 : index to i64
        %61 = arith.cmpi eq, %60, %c1_i64 : i64
        %62 = arith.select %61, %3, %2 : memref<4x512xf32, #hivm.address_space<ub>>
        %63 = arith.index_cast %arg18 : i32 to index
        %64 = affine.apply #map5()[%63]
        %65 = arith.index_cast %64 : index to i64
        %66 = arith.cmpi eq, %65, %c1_i64 : i64
        %67 = arith.select %66, %5, %4 : memref<4x8x1xf32, #hivm.address_space<ub>>
        %68 = arith.index_cast %arg18 : i32 to index
        %69 = affine.apply #map5()[%68]
        %70 = arith.index_cast %69 : index to i64
        %71 = arith.cmpi eq, %70, %c1_i64 : i64
        %72 = arith.select %71, %7, %6 : memref<4x4xi64, #hivm.address_space<ub>>
        %73 = arith.index_cast %arg18 : i32 to index
        %74 = affine.apply #map5()[%73]
        %75 = arith.index_cast %74 : index to i1
        %76 = arith.select %75, %c1_i64, %c2_i64 : i64
        %77 = arith.index_cast %74 : index to i1
        %78 = arith.xori %77, %true : i1
        %79 = arith.extui %78 : i1 to i64
        %80 = arith.index_cast %arg18 : i32 to index
        %81 = affine.apply #map6()[%80, %43, %17]
        %reinterpret_cast_35 = memref.reinterpret_cast %arg10 to offset: [%81], sizes: [4], strides: [%17] : memref<?xi64, #hivm.address_space<gm>> to memref<4xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_36 = memref.subview %72[0, 0] [4, 1] [1, 1] : memref<4x4xi64, #hivm.address_space<ub>> to memref<4xi64, strided<[4]>, #hivm.address_space<ub>>
        %subview_37 = memref.subview %reinterpret_cast_35[0] [%46] [1] : memref<4xi64, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_38 = memref.subview %subview_36[0] [%46] [1] : memref<4xi64, strided<[4]>, #hivm.address_space<ub>> to memref<?xi64, strided<[4]>, #hivm.address_space<ub>>
        scf.if %47 {
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %base_buffer_101, %offset_102, %sizes_103, %strides_104 = memref.extract_strided_metadata %subview_36 : memref<4xi64, strided<[4]>, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
          %reinterpret_cast_105 = memref.reinterpret_cast %base_buffer_101 to offset: [0], sizes: [4, 1], strides: [4, 1] : memref<i64, #hivm.address_space<ub>> to memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %cast_106 = memref.cast %reinterpret_cast_105 : memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_int64_t_to_2d(%c0_i64, %cast_106) : (i64, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %79]
        %cast_39 = memref.cast %subview_38 : memref<?xi64, strided<[4]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int64_t(%subview_37, %cast_39, %c2_i32, %c0_i64, %c0) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
        %82 = affine.apply #map6()[%80, %43, %19]
        %reinterpret_cast_40 = memref.reinterpret_cast %arg8 to offset: [%82], sizes: [4, 1], strides: [%19, %19] : memref<?xf32, #hivm.address_space<gm>> to memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %subview_41 = memref.subview %67[0, 0, 0] [4, 1, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %subview_42 = memref.subview %reinterpret_cast_40[0, 0] [%46, 1] [1, 1] : memref<4x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %subview_43 = memref.subview %subview_41[0, 0] [%46, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        scf.if %47 {
          %collapse_shape_101 = memref.collapse_shape %subview_41 [[0, 1]] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
          %base_buffer_102, %offset_103, %sizes_104, %strides_105 = memref.extract_strided_metadata %collapse_shape_101 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
          %reinterpret_cast_106 = memref.reinterpret_cast %base_buffer_102 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %cast_107 = memref.cast %reinterpret_cast_106 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_2d(%cst_0, %cast_107) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %collapse_shape_44 = memref.collapse_shape %subview_42 [[0, 1]] : memref<?x1xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_45 = memref.collapse_shape %subview_43 [[0, 1]] : memref<?x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %76]
        %cast_46 = memref.cast %collapse_shape_45 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%collapse_shape_44, %cast_46, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %83 = hivm.hir.pointer_cast(%c13120_i64) : memref<4xi1, #hivm.address_space<ub>>
        %84 = hivm.hir.pointer_cast(%c13056_i64) : memref<4xi8, #hivm.address_space<ub>>
        %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_36 : memref<4xi64, strided<[4]>, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_47 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [4, 1] : memref<i64, #hivm.address_space<ub>> to memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>
        %base_buffer_48, %offset_49, %sizes_50, %strides_51 = memref.extract_strided_metadata %14 : memref<4xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_52 = memref.reinterpret_cast %base_buffer_48 to offset: [0], sizes: [4, 1], strides: [1, 1] : memref<i64, #hivm.address_space<ub>> to memref<4x1xi64, strided<[1, 1]>, #hivm.address_space<ub>>
        %base_buffer_53, %offset_54, %sizes_55, %strides_56 = memref.extract_strided_metadata %84 : memref<4xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_57 = memref.reinterpret_cast %base_buffer_53 to offset: [0], sizes: [4, 1], strides: [1, 1] : memref<i8, #hivm.address_space<ub>> to memref<4x1xi8, strided<[1, 1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg20 = %c0 to %c4 step %c1 {
          %101 = memref.load %reinterpret_cast_47[%arg20, %c0] : memref<4x1xi64, strided<[4, 1]>, #hivm.address_space<ub>>
          %102 = memref.load %reinterpret_cast_52[%arg20, %c0] : memref<4x1xi64, strided<[1, 1]>, #hivm.address_space<ub>>
          %103 = arith.cmpi sge, %101, %102 : i64
          %104 = arith.extui %103 : i1 to i8
          memref.store %104, %reinterpret_cast_57[%arg20, %c0] : memref<4x1xi8, strided<[1, 1]>, #hivm.address_space<ub>>
        }
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %85 = hivm.hir.pointer_cast(%c13088_i64) : memref<4xf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_58 = memref.cast %84 : memref<4xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_59 = memref.cast %85 : memref<4xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int8_t_to_half_1d_with_mode(%cast_58, %cast_59, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %86 = hivm.hir.pointer_cast(%c13120_i64) : memref<4xf16, #hivm.address_space<ub>>
        %cast_60 = memref.cast %86 : memref<4xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_half_to_1d(%cst, %cast_60) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_61 = memref.cast %85 : memref<4xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_62 = memref.cast %86 : memref<4xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_63 = memref.cast %83 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_ne_1d_half(%cast_61, %cast_62, %cast_63) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %87 = hivm.hir.pointer_cast(%c13152_i64) : memref<4xf16, #hivm.address_space<ub>>
        %88 = hivm.hir.pointer_cast(%c13184_i64) : memref<48xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_64 = memref.cast %83 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_65 = memref.cast %87 : memref<4xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %88 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bool_to_half_1d_with_temp(%cast_64, %cast_65, %cast_66, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %expand_shape_67 = memref.expand_shape %87 [[0, 1]] output_shape [4, 1] : memref<4xf16, #hivm.address_space<ub>> into memref<4x1xf16, #hivm.address_space<ub>>
        %89 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xf16, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c17376_i64) : memref<128xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_68 = memref.cast %expand_shape_67 : memref<4x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_69 = memref.cast %89 : memref<4x512xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %90 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_half(%cast_68, %cast_69, %cast_70) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %91 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xi1, #hivm.address_space<ub>>
        %collapse_shape_71 = memref.collapse_shape %89 [[0, 1]] : memref<4x512xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
        %collapse_shape_72 = memref.collapse_shape %91 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_73 = memref.cast %collapse_shape_71 : memref<2048xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_74 = memref.cast %collapse_shape_72 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmps_eq_1d_half(%cast_73, %cst, %cast_74) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %92 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xi1, #hivm.address_space<ub>>
        %collapse_shape_75 = memref.collapse_shape %92 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %93 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_76 = memref.cast %collapse_shape_72 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_77 = memref.cast %collapse_shape_75 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_78 = memref.cast %93 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_76, %cast_77, %cast_78) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.for %arg20 = %c0 to %55 step %c1 {
          %101 = memref.load %subview_36[%arg20] {DiscreteMemAccess} : memref<4xi64, strided<[4]>, #hivm.address_space<ub>>
          %102 = arith.muli %101, %16 : i64
          %103 = arith.index_cast %102 : i64 to index
          %reinterpret_cast_101 = memref.reinterpret_cast %arg4 to offset: [%103], sizes: [1, 512], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_102 = memref.subview %62[%arg20, 0] [1, 512] [1, 1] : memref<4x512xf32, #hivm.address_space<ub>> to memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_103 = memref.collapse_shape %reinterpret_cast_101 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_104 = memref.collapse_shape %subview_102 [[0, 1]] : memref<1x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_105 = memref.cast %collapse_shape_103 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_106 = memref.cast %collapse_shape_104 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_105, %cast_106, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        } {ExtractedLoadOrStore, hivm.parallel_loop, pipeline.veconly}
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %79]
        %94 = hivm.hir.pointer_cast(%c13280_i64) : memref<4x512xi1, #hivm.address_space<ub>>
        %collapse_shape_79 = memref.collapse_shape %94 [[0, 1]] : memref<4x512xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %95 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_80 = memref.cast %collapse_shape_25 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_81 = memref.cast %collapse_shape_75 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_82 = memref.cast %collapse_shape_79 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_83 = memref.cast %95 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vand_1d_bool(%cast_80, %cast_81, %cast_82, %cast_83) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %96 = hivm.hir.pointer_cast(%c25824_i64) : memref<4x512xf32, #hivm.address_space<ub>>
        %collapse_shape_84 = memref.collapse_shape %62 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        %collapse_shape_85 = memref.collapse_shape %96 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        %97 = hivm.hir.pointer_cast(%c34016_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_86 = memref.cast %collapse_shape_79 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_87 = memref.cast %collapse_shape_84 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_88 = memref.cast %collapse_shape_85 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_89 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vs_1d_float(%cast_86, %cast_87, %cst_0, %cast_88, %cast_89) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %98 = hivm.hir.pointer_cast(%c25824_i64) : memref<4x512xf32, #hivm.address_space<ub>>
        %99 = hivm.hir.pointer_cast(%c34080_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_90 = memref.cast %96 : memref<4x512xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_91 = memref.cast %subview_41 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_92 = memref.cast %98 : memref<4x512xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_93 = memref.cast %99 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_90, %cast_91, %cast_92, %cast_93) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %76]
        %collapse_shape_94 = memref.collapse_shape %arg19 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        %collapse_shape_95 = memref.collapse_shape %98 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        %collapse_shape_96 = memref.collapse_shape %26 [[0, 1]] : memref<4x512xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %100 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_97 = memref.cast %collapse_shape_94 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_98 = memref.cast %collapse_shape_95 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_99 = memref.cast %collapse_shape_96 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_100 = memref.cast %100 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_97, %cast_98, %cast_99, %cast_100) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %26 : memref<4x512xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %57 = affine.apply #map7()[%43, %20]
      %reinterpret_cast = memref.reinterpret_cast %arg12 to offset: [%57], sizes: [4, 512], strides: [%20, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %56[0, 0] [%46, 512] [1, 1] : memref<4x512xf32, #hivm.address_space<ub>> to memref<?x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %subview_32 = memref.subview %reinterpret_cast[0, 0] [%46, 512] [1, 1] : memref<4x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_33 = memref.cast %subview : memref<?x512xf32, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %subview_32 : memref<?x512xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_33, %cast_34, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %32]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
