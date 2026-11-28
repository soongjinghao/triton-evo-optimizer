#map = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_per_token_group_quant_8bit_colmajor_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vabs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_sv_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmins_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_int32_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int32_t_to_int8_t_1d_with_overflow(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_int8_t(memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_per_token_group_quant_8bit_colmajor_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c6_i32 = arith.constant 6 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16224_i64 = arith.constant 16224 : i64
    %c15200_i64 = arith.constant 15200 : i64
    %c11104_i64 = arith.constant 11104 : i64
    %c11040_i64 = arith.constant 11040 : i64
    %c10016_i64 = arith.constant 10016 : i64
    %c9984_i64 = arith.constant 9984 : i64
    %c8960_i64 = arith.constant 8960 : i64
    %c7936_i64 = arith.constant 7936 : i64
    %c7904_i64 = arith.constant 7904 : i64
    %c7872_i64 = arith.constant 7872 : i64
    %c6848_i64 = arith.constant 6848 : i64
    %c6784_i64 = arith.constant 6784 : i64
    %c5760_i64 = arith.constant 5760 : i64
    %c4736_i64 = arith.constant 4736 : i64
    %c4704_i64 = arith.constant 4704 : i64
    %c4448_i64 = arith.constant 4448 : i64
    %c4416_i64 = arith.constant 4416 : i64
    %c4384_i64 = arith.constant 4384 : i64
    %c4352_i64 = arith.constant 4352 : i64
    %c4288_i64 = arith.constant 4288 : i64
    %c4256_i64 = arith.constant 4256 : i64
    %c4224_i64 = arith.constant 4224 : i64
    %c4192_i64 = arith.constant 4192 : i64
    %c4160_i64 = arith.constant 4160 : i64
    %c3136_i64 = arith.constant 3136 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c2080_i64 = arith.constant 2080 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0.00787401571 : f32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant -1.280000e+02 : f32
    %cst_4 = arith.constant 1.270000e+02 : f32
    %c48_i32 = arith.constant 48 : i32
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg9, %arg8 : i32
    %5 = hivm.hir.pointer_cast(%c0_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %cast = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_3, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %6 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_5 = memref.collapse_shape %6 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %cast_6 = memref.cast %collapse_shape_5 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_4, %cast_6) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %7 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xf32, #hivm.address_space<ub>>
    %cast_7 = memref.cast %7 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%arg6, %cast_7) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %8 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_8 = memref.cast %7 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %7 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_10 = memref.cast %8 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_8, %cast_9, %cast_10) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %9 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_11 = memref.cast %8 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_12 = memref.cast %9 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_13 = memref.cast %10 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_11, %cast_12, %cast_13) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %11 = hivm.hir.pointer_cast(%c2080_i64) : memref<8xf32, #hivm.address_space<ub>>
    %cast_14 = memref.cast %11 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_2, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %12 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %12 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    %cast_16 = memref.cast %collapse_shape_5 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_17 = memref.cast %collapse_shape_5 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_18 = memref.cast %collapse_shape_15 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_16, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %13 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %13 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %14 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_20 = memref.cast %collapse_shape_15 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_21 = memref.cast %collapse_shape_19 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_22 = memref.cast %14 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_20, %cast_21, %cast_22) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %15 = hivm.hir.pointer_cast(%c0_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %15 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    %cast_24 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_25 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_26 = memref.cast %collapse_shape_23 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_24, %cast_25, %cast_26) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %16 = hivm.hir.pointer_cast(%c0_i64) : memref<8x32xi1, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %16 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %17 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_28 = memref.cast %collapse_shape_23 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_29 = memref.cast %collapse_shape_27 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_30 = memref.cast %17 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %18 = arith.divsi %arg10, %arg9 : i32
      %19 = arith.remsi %18, %arg8 : i32
      %20 = arith.divsi %arg10, %4 : i32
      %21 = arith.remsi %20, %arg7 : i32
      %22 = arith.muli %21, %c8_i32 : i32
      %23 = arith.muli %19, %c32_i32 : i32
      %24 = arith.index_cast %22 : i32 to index
      %25 = arith.index_cast %23 : i32 to index
      %26 = affine.apply #map()[%25, %24]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [8, 32], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_31 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [8, 32], strides: [128, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<8x32xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %27 = arith.muli %19, %c16_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = affine.apply #map1()[%28, %24]
      %reinterpret_cast_32 = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %30 = hivm.hir.pointer_cast(%c2112_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_33 = memref.cast %reinterpret_cast : memref<8x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_34 = memref.cast %30 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_33, %cast_34, %c0_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %31 = hivm.hir.pointer_cast(%c3136_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_35 = memref.collapse_shape %30 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_36 = memref.collapse_shape %31 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %32 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_37 = memref.cast %collapse_shape_35 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %collapse_shape_36 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_39 = memref.cast %32 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vabs_1d_float(%cast_37, %cast_38, %cast_39) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %33 = hivm.hir.pointer_cast(%c4160_i64) : memref<8x1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %34 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_40 = memref.cast %31 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %33 : memref<8x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %34 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_max_ar_float(%cast_40, %cast_41, %cast_42, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_43 = memref.collapse_shape %33 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
      %35 = hivm.hir.pointer_cast(%c4192_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_44 = memref.cast %collapse_shape_43 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %collapse_shape_43 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %35 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_44, %cast_45, %cast_46) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %36 = hivm.hir.pointer_cast(%c4192_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %37 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_47 = memref.cast %35 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %36 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %37 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_47, %cast_48, %cast_49) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %38 = hivm.hir.pointer_cast(%c4224_i64) : memref<8xf32, #hivm.address_space<ub>>
      %39 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_50 = memref.cast %collapse_shape_43 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_51 = memref.cast %38 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_52 = memref.cast %39 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_float(%cast_50, %arg6, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %40 = hivm.hir.pointer_cast(%c4256_i64) : memref<8xf32, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c4288_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_53 = memref.cast %36 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_54 = memref.cast %38 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %40 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %41 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_sv_1d_float(%cast_53, %arg6, %cast_54, %cast_55, %cast_56) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %42 = hivm.hir.pointer_cast(%c4352_i64) : memref<8xf32, #hivm.address_space<ub>>
      %43 = hivm.hir.pointer_cast(%c4384_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_57 = memref.cast %9 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_58 = memref.cast %collapse_shape_43 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_59 = memref.cast %40 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_60 = memref.cast %42 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %43 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_57, %cast_58, %cast_59, %cast_60, %cast_61) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %44 = hivm.hir.pointer_cast(%c4352_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %45 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_62 = memref.cast %42 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_63 = memref.cast %44 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_64 = memref.cast %45 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_62, %cst_1, %cast_63, %cast_64) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %46 = hivm.hir.pointer_cast(%c4416_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %47 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_65 = memref.cast %11 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_66 = memref.cast %44 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_67 = memref.cast %46 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_68 = memref.cast %47 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_1d_float(%cast_65, %cast_66, %cast_67, %cast_68) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape = memref.expand_shape %46 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
      %48 = hivm.hir.pointer_cast(%c2112_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c4448_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_69 = memref.cast %30 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_70 = memref.cast %expand_shape : memref<8x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_71 = memref.cast %48 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_72 = memref.cast %49 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_69, %cast_70, %cast_71, %cast_72) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %50 = hivm.hir.pointer_cast(%c4704_i64) : memref<8x32xi1, #hivm.address_space<ub>>
      %collapse_shape_73 = memref.collapse_shape %48 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_74 = memref.collapse_shape %50 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_75 = memref.cast %collapse_shape_73 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_76 = memref.cast %collapse_shape_73 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_77 = memref.cast %collapse_shape_74 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_75, %cast_76, %cast_77) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %51 = hivm.hir.pointer_cast(%c4704_i64) : memref<8x32xi1, #hivm.address_space<ub>>
      %collapse_shape_78 = memref.collapse_shape %51 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %52 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_79 = memref.cast %collapse_shape_74 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_80 = memref.cast %collapse_shape_78 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_81 = memref.cast %52 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_79, %cast_80, %cast_81) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %53 = hivm.hir.pointer_cast(%c4736_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_82 = memref.collapse_shape %53 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %54 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_83 = memref.cast %collapse_shape_73 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_84 = memref.cast %collapse_shape_82 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %54 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmins_vs_1d_float(%cast_83, %cst_4, %cast_84, %cast_85) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %55 = hivm.hir.pointer_cast(%c5760_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_86 = memref.collapse_shape %55 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %56 = hivm.hir.pointer_cast(%c6784_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_87 = memref.cast %collapse_shape_78 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_88 = memref.cast %collapse_shape_82 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_89 = memref.cast %collapse_shape_86 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_90 = memref.cast %56 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_sv_1d_float(%cast_87, %cst_4, %cast_88, %cast_89, %cast_90) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %57 = hivm.hir.pointer_cast(%c6848_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_91 = memref.collapse_shape %57 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %58 = hivm.hir.pointer_cast(%c7872_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_92 = memref.cast %collapse_shape_19 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_93 = memref.cast %collapse_shape_73 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_94 = memref.cast %collapse_shape_86 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_95 = memref.cast %collapse_shape_91 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_96 = memref.cast %58 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_92, %cast_93, %cast_94, %cast_95, %cast_96) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %59 = hivm.hir.pointer_cast(%c7904_i64) : memref<8x32xi1, #hivm.address_space<ub>>
      %collapse_shape_97 = memref.collapse_shape %59 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_98 = memref.cast %collapse_shape_91 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_99 = memref.cast %collapse_shape_91 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_100 = memref.cast %collapse_shape_97 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_98, %cast_99, %cast_100) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %60 = hivm.hir.pointer_cast(%c7904_i64) : memref<8x32xi1, #hivm.address_space<ub>>
      %collapse_shape_101 = memref.collapse_shape %60 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %61 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_102 = memref.cast %collapse_shape_97 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_103 = memref.cast %collapse_shape_101 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_104 = memref.cast %61 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_102, %cast_103, %cast_104) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %62 = hivm.hir.pointer_cast(%c7936_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_105 = memref.collapse_shape %62 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %63 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_106 = memref.cast %collapse_shape_91 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_107 = memref.cast %collapse_shape_105 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_108 = memref.cast %63 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_float(%cast_106, %cst_3, %cast_107, %cast_108) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %64 = hivm.hir.pointer_cast(%c8960_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_109 = memref.collapse_shape %64 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %65 = hivm.hir.pointer_cast(%c9984_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_110 = memref.cast %collapse_shape_27 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_111 = memref.cast %collapse_shape_91 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_112 = memref.cast %collapse_shape_105 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_113 = memref.cast %collapse_shape_109 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_114 = memref.cast %65 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_110, %cast_111, %cast_112, %cast_113, %cast_114) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %66 = hivm.hir.pointer_cast(%c10016_i64) : memref<8x32xf32, #hivm.address_space<ub>>
      %collapse_shape_115 = memref.collapse_shape %66 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c11040_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_116 = memref.cast %collapse_shape_101 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_117 = memref.cast %collapse_shape_109 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_118 = memref.cast %collapse_shape_115 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_119 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_sv_1d_float(%cast_116, %cst_3, %cast_117, %cast_118, %cast_119) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %68 = hivm.hir.pointer_cast(%c11104_i64) : memref<32x32xi32, #hivm.address_space<ub>>
      %subview = memref.subview %68[0, 0] [8, 32] [1, 1] : memref<32x32xi32, #hivm.address_space<ub>> to memref<8x32xi32, strided<[32, 1]>, #hivm.address_space<ub>>
      %collapse_shape_120 = memref.collapse_shape %subview [[0, 1]] : memref<8x32xi32, strided<[32, 1]>, #hivm.address_space<ub>> into memref<256xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_121 = memref.cast %collapse_shape_115 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_122 = memref.cast %collapse_shape_120 : memref<256xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_float_to_int32_t_1d_with_mode(%cast_121, %cast_122, %c4_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %69 = hivm.hir.pointer_cast(%c15200_i64) : memref<32x32xi8, #hivm.address_space<ub>>
      %subview_123 = memref.subview %69[0, 0] [8, 32] [1, 1] : memref<32x32xi8, #hivm.address_space<ub>> to memref<8x32xi8, strided<[32, 1]>, #hivm.address_space<ub>>
      %collapse_shape_124 = memref.collapse_shape %subview_123 [[0, 1]] : memref<8x32xi8, strided<[32, 1]>, #hivm.address_space<ub>> into memref<256xi8, strided<[1]>, #hivm.address_space<ub>>
      %70 = hivm.hir.pointer_cast(%c16224_i64) : memref<2048xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_125 = memref.cast %collapse_shape_120 : memref<256xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_126 = memref.cast %collapse_shape_124 : memref<256xi8, strided<[1]>, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_127 = memref.cast %70 : memref<2048xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_int32_t_to_int8_t_1d_with_overflow(%cast_125, %cast_126, %cast_127, %c6_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_128 = memref.cast %subview_123 : memref<8x32xi8, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_129 = memref.cast %reinterpret_cast_31 : memref<8x32xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_int8_t(%cast_128, %cast_129, %c0_i32) : (memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      %cast_130 = memref.cast %44 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_131 = memref.cast %reinterpret_cast_32 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_130, %cast_131, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
