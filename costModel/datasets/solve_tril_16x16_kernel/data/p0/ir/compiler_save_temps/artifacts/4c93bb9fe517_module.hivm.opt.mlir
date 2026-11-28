#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<()[s0] -> (16, s0)>
#map2 = affine_map<()[s0] -> (0, 16, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (((s0 - 1) floordiv 2 + ((s2 - s3) floordiv 48) * (s1 floordiv 2)) mod 2)>
#map6 = affine_map<()[s0, s1] -> (s0 + s1)>
#map7 = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @solve_tril_16x16_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int32_t_to_float_1d_with_mode(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_gt_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_float_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vor_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmps_eq_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-2_i32 = arith.constant -2 : i32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c192_i64 = arith.constant 192 : i64
    %c1216_i64 = arith.constant 1216 : i64
    %c1728_i64 = arith.constant 1728 : i64
    %c2752_i64 = arith.constant 2752 : i64
    %c2784_i64 = arith.constant 2784 : i64
    %c3808_i64 = arith.constant 3808 : i64
    %c4928_i64 = arith.constant 4928 : i64
    %c5952_i64 = arith.constant 5952 : i64
    %c14208_i64 = arith.constant 14208 : i64
    %c6080_i64 = arith.constant 6080 : i64
    %c7104_i64 = arith.constant 7104 : i64
    %c7616_i64 = arith.constant 7616 : i64
    %c7680_i64 = arith.constant 7680 : i64
    %c8768_i64 = arith.constant 8768 : i64
    %c9792_i64 = arith.constant 9792 : i64
    %c10304_i64 = arith.constant 10304 : i64
    %c10368_i64 = arith.constant 10368 : i64
    %c11392_i64 = arith.constant 11392 : i64
    %c11456_i64 = arith.constant 11456 : i64
    %c11520_i64 = arith.constant 11520 : i64
    %c12544_i64 = arith.constant 12544 : i64
    %c13568_i64 = arith.constant 13568 : i64
    %c13600_i64 = arith.constant 13600 : i64
    %c13696_i64 = arith.constant 13696 : i64
    %c15232_i64 = arith.constant 15232 : i64
    %c15328_i64 = arith.constant 15328 : i64
    %c16352_i64 = arith.constant 16352 : i64
    %c16864_i64 = arith.constant 16864 : i64
    %c16928_i64 = arith.constant 16928 : i64
    %c17952_i64 = arith.constant 17952 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c21024_i64 = arith.constant 21024 : i64
    %c6016_i64 = arith.constant 6016 : i64
    %c21088_i64 = arith.constant 21088 : i64
    %c8704_i64 = arith.constant 8704 : i64
    %c20000_i64 = arith.constant 20000 : i64
    %c3904_i64 = arith.constant 3904 : i64
    %c21152_i64 = arith.constant 21152 : i64
    %c15264_i64 = arith.constant 15264 : i64
    %c18976_i64 = arith.constant 18976 : i64
    %c21216_i64 = arith.constant 21216 : i64
    %0 = hivm.hir.pointer_cast(%c18976_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c21216_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c15264_i64) : memref<16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c21152_i64) : memref<16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c3904_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c20000_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c8704_i64) : memref<16xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c21088_i64) : memref<16xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c6016_i64) : memref<16xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c21024_i64) : memref<16xf32, #hivm.address_space<ub>>
    %10 = arith.muli %arg6, %arg7 : i32
    %11 = arith.muli %10, %arg8 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg8, %arg7 : i32
    %15 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast = memref.cast %15 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %16 = hivm.hir.pointer_cast(%c64_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_2 = memref.cast %15 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_3 = memref.cast %collapse_shape : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int32_t_to_float_1d_with_mode(%cast_2, %cast_3, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %17 = hivm.hir.pointer_cast(%c128_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %17 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %cast_5 = memref.cast %15 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_6 = memref.cast %collapse_shape_4 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int32_t_to_float_1d_with_mode(%cast_5, %cast_6, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %18 = hivm.hir.pointer_cast(%c192_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c1216_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_7 = memref.cast %16 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_8 = memref.cast %18 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %19 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_last_axis_align_2d_float(%cast_7, %cast_8, %cast_9) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %20 = hivm.hir.pointer_cast(%c1728_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %cast_10 = memref.cast %17 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_11 = memref.cast %20 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_first_axis_align_2d_float(%cast_10, %cast_11) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %21 = hivm.hir.pointer_cast(%c2752_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %18 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_13 = memref.collapse_shape %20 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %21 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_15 = memref.cast %collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_16 = memref.cast %collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_17 = memref.cast %collapse_shape_14 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_gt_1d_float(%cast_15, %cast_16, %cast_17) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %22 = hivm.hir.pointer_cast(%c192_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %22 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_19 = memref.cast %collapse_shape_12 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_20 = memref.cast %collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_21 = memref.cast %collapse_shape_18 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_19, %cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %23 = arith.index_cast %arg5 : i32 to index
    %24 = affine.max #map()[%23]
    %25 = affine.min #map1()[%24]
    %26 = affine.min #map2()[%24]
    %27 = affine.apply #map3()[%25, %26]
    %28 = arith.cmpi slt, %27, %c16 : index
    %29 = arith.minsi %arg5, %c16_i32 : i32
    %30 = arith.andi %29, %c-2_i32 : i32
    %31 = arith.index_cast %29 : i32 to index
    %32 = affine.max #map()[%31]
    %33 = affine.min #map1()[%32]
    %34 = arith.cmpi slt, %33, %c16 : index
    %35 = arith.cmpi slt, %30, %29 : i32
    %36 = hivm.hir.pointer_cast(%c2784_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %36 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %37 = hivm.hir.pointer_cast(%c3808_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_23 = memref.cast %collapse_shape_18 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_24 = memref.cast %collapse_shape_22 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_25 = memref.cast %37 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_bool_to_float_1d_with_temp(%cast_23, %cast_24, %cast_25, %c0_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
    scf.for %arg9 = %13 to %11 step %c48_i32  : i32 {
      %38 = arith.index_cast %arg9 : i32 to index
      %39 = arith.index_cast %13 : i32 to index
      %40 = affine.apply #map4()[%38, %39]
      %41 = arith.index_cast %40 : index to i64
      %42 = arith.cmpi eq, %41, %c1_i64 : i64
      %43 = arith.select %42, %1, %0 : memref<16x16xf32, #hivm.address_space<ub>>
      %44 = arith.index_cast %arg9 : i32 to index
      %45 = arith.index_cast %13 : i32 to index
      %46 = affine.apply #map4()[%44, %45]
      %47 = arith.index_cast %46 : index to i64
      %48 = arith.cmpi eq, %47, %c1_i64 : i64
      %49 = arith.select %48, %3, %2 : memref<16xf32, #hivm.address_space<ub>>
      %50 = arith.index_cast %arg9 : i32 to index
      %51 = arith.index_cast %13 : i32 to index
      %52 = affine.apply #map4()[%50, %51]
      %53 = arith.index_cast %52 : index to i64
      %54 = arith.cmpi eq, %53, %c1_i64 : i64
      %55 = arith.select %54, %5, %4 : memref<16x16xf32, #hivm.address_space<ub>>
      %56 = arith.index_cast %arg9 : i32 to index
      %57 = arith.index_cast %13 : i32 to index
      %58 = affine.apply #map4()[%56, %57]
      %59 = arith.index_cast %58 : index to i1
      %60 = arith.select %59, %c5_i64, %c6_i64 : i64
      %61 = arith.index_cast %58 : index to i1
      %62 = arith.xori %61, %true : i1
      %63 = arith.extui %62 : i1 to i64
      hivm.hir.set_mask_norm
      %64 = arith.divsi %arg9, %14 : i32
      %65 = arith.remsi %64, %arg6 : i32
      %66 = arith.divsi %65, %c8_i32 : i32
      %67 = arith.remsi %65, %c8_i32 : i32
      %68 = arith.muli %66, %arg5 : i32
      %69 = arith.muli %68, %c8_i32 : i32
      %70 = arith.addi %69, %67 : i32
      %71 = arith.muli %70, %c16_i32 : i32
      %72 = arith.index_cast %71 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%27, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_26 = memref.subview %55[%26, 0] [%27, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %28 {
        %collapse_shape_54 = memref.collapse_shape %55 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %cast_55 = memref.cast %collapse_shape_54 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_1, %cast_55) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %60]
      %cast_27 = memref.cast %subview : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_28 = memref.cast %subview_26 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_27, %cast_28, %c2_i32, %cst_1, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %73 = hivm.hir.pointer_cast(%c4928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_29 = memref.collapse_shape %55 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %73 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %74 = hivm.hir.pointer_cast(%c5952_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_31 = memref.cast %collapse_shape_14 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %collapse_shape_29 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_33 = memref.cast %collapse_shape_30 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %74 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_31, %cast_32, %cst_1, %cast_33, %cast_34) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %60]
      %75 = hivm.hir.pointer_cast(%c4928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_35 = memref.collapse_shape %75 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %76 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_36 = memref.cast %collapse_shape_30 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %collapse_shape_35 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %76 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_36, %cst, %cast_37, %cast_38) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %77 = hivm.hir.pointer_cast(%c14208_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_39 = memref.collapse_shape %77 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %78 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_40 = memref.cast %collapse_shape_35 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %collapse_shape_39 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %78 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_40, %cst_1, %cast_41, %cast_42) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %79 = scf.for %arg10 = %c1_i32 to %30 step %c2_i32 iter_args(%arg11 = %77) -> (memref<16x16xf32, #hivm.address_space<ub>>)  : i32 {
        %81 = arith.index_cast %arg10 : i32 to index
        %82 = arith.index_cast %30 : i32 to index
        %83 = arith.index_cast %arg9 : i32 to index
        %84 = arith.index_cast %13 : i32 to index
        %85 = affine.apply #map5()[%81, %82, %83, %84]
        %86 = arith.index_cast %85 : index to i64
        %87 = arith.cmpi eq, %86, %c1_i64 : i64
        %88 = arith.select %87, %7, %6 : memref<16xf32, #hivm.address_space<ub>>
        %89 = arith.index_cast %arg10 : i32 to index
        %90 = arith.index_cast %30 : i32 to index
        %91 = arith.index_cast %arg9 : i32 to index
        %92 = arith.index_cast %13 : i32 to index
        %93 = affine.apply #map5()[%89, %90, %91, %92]
        %94 = arith.index_cast %93 : index to i64
        %95 = arith.cmpi eq, %94, %c1_i64 : i64
        %96 = arith.select %95, %9, %8 : memref<16xf32, #hivm.address_space<ub>>
        %97 = arith.index_cast %arg10 : i32 to index
        %98 = arith.index_cast %30 : i32 to index
        %99 = arith.index_cast %arg9 : i32 to index
        %100 = arith.index_cast %13 : i32 to index
        %101 = affine.apply #map5()[%97, %98, %99, %100]
        %102 = arith.index_cast %101 : index to i1
        %103 = arith.select %102, %c3_i64, %c4_i64 : i64
        %104 = arith.muli %arg10, %c128_i32 : i32
        %105 = arith.index_cast %104 : i32 to index
        %106 = affine.apply #map6()[%72, %105]
        %reinterpret_cast_54 = memref.reinterpret_cast %arg3 to offset: [%106], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_55 = memref.subview %reinterpret_cast_54[0] [%33] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_56 = memref.subview %96[0] [%33] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %34 {
          %cast_143 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_1, %cast_143) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %103]
        %cast_57 = memref.cast %subview_55 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_58 = memref.cast %subview_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_57, %cast_58, %c2_i32, %cst_1, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %107 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_59 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_60 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_61 = memref.cast %107 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_59, %cst, %cast_60, %cast_61) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %108 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_62 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_63 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_64 = memref.cast %108 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_62, %cst_1, %cast_63, %cast_64) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape = memref.expand_shape %96 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %109 = hivm.hir.pointer_cast(%c6080_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %110 = hivm.hir.pointer_cast(%c7104_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_65 = memref.cast %expand_shape : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %arg11 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_67 = memref.cast %109 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_68 = memref.cast %110 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_65, %cast_66, %cast_67, %cast_68) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %111 = hivm.hir.pointer_cast(%c7616_i64) : memref<1x16xf32, #hivm.address_space<ub>>
        %112 = hivm.hir.pointer_cast(%c7680_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_69 = memref.cast %109 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %111 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_71 = memref.cast %112 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_69, %cast_70, %cast_71, %cst_1) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_72 = memref.collapse_shape %111 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %113 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_73 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_74 = memref.cast %collapse_shape_72 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_75 = memref.cast %96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_76 = memref.cast %113 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_73, %cast_74, %cast_75, %cast_76) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %114 = arith.addi %arg10, %c1_i32 : i32
        %115 = arith.muli %114, %c128_i32 : i32
        %116 = arith.index_cast %115 : i32 to index
        %117 = affine.apply #map6()[%72, %116]
        %reinterpret_cast_77 = memref.reinterpret_cast %arg3 to offset: [%117], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_78 = memref.subview %reinterpret_cast_77[0] [%33] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_79 = memref.subview %88[0] [%33] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %34 {
          %cast_143 = memref.cast %88 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_1, %cast_143) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        %cast_80 = memref.cast %subview_78 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_81 = memref.cast %subview_79 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_80, %cast_81, %c2_i32, %cst_1, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %118 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_82 = memref.cast %88 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_83 = memref.cast %88 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_84 = memref.cast %118 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_82, %cst, %cast_83, %cast_84) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %119 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_85 = memref.cast %88 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_86 = memref.cast %88 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_87 = memref.cast %119 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_85, %cst_1, %cast_86, %cast_87) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_88 = memref.expand_shape %88 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %120 = hivm.hir.pointer_cast(%c8768_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %121 = hivm.hir.pointer_cast(%c9792_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_89 = memref.cast %expand_shape_88 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_90 = memref.cast %arg11 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_91 = memref.cast %120 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_92 = memref.cast %121 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_89, %cast_90, %cast_91, %cast_92) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %122 = hivm.hir.pointer_cast(%c10304_i64) : memref<1x16xf32, #hivm.address_space<ub>>
        %123 = hivm.hir.pointer_cast(%c10368_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_93 = memref.cast %120 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_94 = memref.cast %122 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_95 = memref.cast %123 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_93, %cast_94, %cast_95, %cst_1) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_96 = memref.collapse_shape %122 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        %124 = hivm.hir.pointer_cast(%c10304_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %125 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_97 = memref.cast %88 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_98 = memref.cast %collapse_shape_96 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_99 = memref.cast %124 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_100 = memref.cast %125 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_97, %cast_98, %cast_99, %cast_100) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %126 = hivm.hir.pointer_cast(%c11392_i64) : memref<16xi32, #hivm.address_space<ub>>
        %cast_101 = memref.cast %126 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_int32_t_to_1d(%arg10, %cast_101) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %127 = hivm.hir.pointer_cast(%c11392_i64) : memref<16xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_102 = memref.cast %15 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_103 = memref.cast %126 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_104 = memref.cast %127 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_int32_t(%cast_102, %cast_103, %cast_104) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %128 = hivm.hir.pointer_cast(%c11456_i64) : memref<16xi32, #hivm.address_space<ub>>
        %cast_105 = memref.cast %128 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_int32_t_to_1d(%114, %cast_105) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %129 = hivm.hir.pointer_cast(%c11456_i64) : memref<16xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_106 = memref.cast %15 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_107 = memref.cast %128 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_108 = memref.cast %129 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_int32_t(%cast_106, %cast_107, %cast_108) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %130 = hivm.hir.pointer_cast(%c11392_i64) : memref<16x1xi1, #hivm.address_space<ub>>
        %collapse_shape_109 = memref.collapse_shape %130 [[0, 1]] : memref<16x1xi1, #hivm.address_space<ub>> into memref<16xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %131 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_110 = memref.cast %127 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_111 = memref.cast %129 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_112 = memref.cast %collapse_shape_109 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_113 = memref.cast %131 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vor_1d_bool(%cast_110, %cast_111, %cast_112, %cast_113) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_114 = memref.expand_shape %96 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
        %132 = hivm.hir.pointer_cast(%c11520_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %cast_115 = memref.cast %expand_shape_114 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_116 = memref.cast %132 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_float(%cast_115, %cast_116) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %103]
        %expand_shape_117 = memref.expand_shape %124 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
        %133 = hivm.hir.pointer_cast(%c12544_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %cast_118 = memref.cast %expand_shape_117 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_119 = memref.cast %133 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_float(%cast_118, %cast_119) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %134 = arith.index_cast %arg10 : i32 to index
        %135 = affine.apply #map7()[%134]
        %136 = arith.cmpi slt, %134, %c0 : index
        %137 = arith.cmpi sge, %134, %c16 : index
        %138 = arith.ori %136, %137 : i1
        scf.if %138 {
        } else {
          %subview_143 = memref.subview %133[0, 0] [%134, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_144 = memref.subview %132[0, 0] [%134, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_145 = memref.collapse_shape %subview_143 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_146 = memref.collapse_shape %subview_144 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_147 = memref.cast %collapse_shape_145 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_148 = memref.cast %collapse_shape_146 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_147, %cast_148, %c0_i32, %cst_1) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
          %subview_149 = memref.subview %132[0, 0] [%135, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_150 = memref.subview %133[0, 0] [%135, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_151 = memref.collapse_shape %subview_149 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_152 = memref.collapse_shape %subview_150 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_153 = memref.cast %collapse_shape_151 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_154 = memref.cast %collapse_shape_152 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_153, %cast_154, %c0_i32, %cst_1) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        }
        %139 = hivm.hir.pointer_cast(%c13568_i64) : memref<16x1xf16, #hivm.address_space<ub>>
        %collapse_shape_120 = memref.collapse_shape %139 [[0, 1]] : memref<16x1xf16, #hivm.address_space<ub>> into memref<16xf16, #hivm.address_space<ub>>
        %140 = hivm.hir.pointer_cast(%c13600_i64) : memref<48xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_121 = memref.cast %collapse_shape_109 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_122 = memref.cast %collapse_shape_120 : memref<16xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_123 = memref.cast %140 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bool_to_half_1d_with_temp(%cast_121, %cast_122, %cast_123, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %141 = hivm.hir.pointer_cast(%c13696_i64) : memref<16x16xf16, #hivm.address_space<ub>>
        %142 = hivm.hir.pointer_cast(%c14208_i64) : memref<0xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_124 = memref.cast %139 : memref<16x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_125 = memref.cast %141 : memref<16x16xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_126 = memref.cast %142 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_half(%cast_124, %cast_125, %cast_126) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %143 = hivm.hir.pointer_cast(%c13696_i64) : memref<16x16xi1, #hivm.address_space<ub>>
        %collapse_shape_127 = memref.collapse_shape %141 [[0, 1]] : memref<16x16xf16, #hivm.address_space<ub>> into memref<256xf16, #hivm.address_space<ub>>
        %collapse_shape_128 = memref.collapse_shape %143 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_129 = memref.cast %collapse_shape_127 : memref<256xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_130 = memref.cast %collapse_shape_128 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmps_eq_1d_half(%cast_129, %cst_0, %cast_130) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %144 = hivm.hir.pointer_cast(%c13696_i64) : memref<16x16xi1, #hivm.address_space<ub>>
        %collapse_shape_131 = memref.collapse_shape %144 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %145 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_132 = memref.cast %collapse_shape_128 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_133 = memref.cast %collapse_shape_131 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_134 = memref.cast %145 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_132, %cast_133, %cast_134) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %146 = hivm.hir.pointer_cast(%c14208_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %collapse_shape_135 = memref.collapse_shape %133 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %collapse_shape_136 = memref.collapse_shape %arg11 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %collapse_shape_137 = memref.collapse_shape %146 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %147 = hivm.hir.pointer_cast(%c15232_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_138 = memref.cast %collapse_shape_131 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_139 = memref.cast %collapse_shape_135 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_140 = memref.cast %collapse_shape_136 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_141 = memref.cast %collapse_shape_137 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_142 = memref.cast %147 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_138, %cast_139, %cast_140, %cast_141, %cast_142) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %146 : memref<16x16xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      scf.if %35 {
        %81 = arith.muli %30, %c128_i32 : i32
        %82 = arith.index_cast %81 : i32 to index
        %83 = affine.apply #map6()[%72, %82]
        %reinterpret_cast_54 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_55 = memref.subview %reinterpret_cast_54[0] [%33] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_56 = memref.subview %49[0] [%33] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %34 {
          %cast_80 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_1, %cast_80) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %cast_57 = memref.cast %subview_55 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_58 = memref.cast %subview_56 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_57, %cast_58, %c2_i32, %cst_1, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %84 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_59 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_60 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_61 = memref.cast %84 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_59, %cst, %cast_60, %cast_61) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %85 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_62 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_63 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_64 = memref.cast %85 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_62, %cst_1, %cast_63, %cast_64) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape = memref.expand_shape %49 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %86 = hivm.hir.pointer_cast(%c15328_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %87 = hivm.hir.pointer_cast(%c16352_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_65 = memref.cast %expand_shape : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %79 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_67 = memref.cast %86 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_68 = memref.cast %87 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_65, %cast_66, %cast_67, %cast_68) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %88 = hivm.hir.pointer_cast(%c16864_i64) : memref<1x16xf32, #hivm.address_space<ub>>
        %89 = hivm.hir.pointer_cast(%c16928_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_69 = memref.cast %86 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %88 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_71 = memref.cast %89 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_69, %cast_70, %cast_71, %cst_1) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_72 = memref.collapse_shape %88 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %90 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_73 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_74 = memref.cast %collapse_shape_72 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_75 = memref.cast %49 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_76 = memref.cast %90 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_73, %cast_74, %cast_75, %cast_76) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_77 = memref.expand_shape %49 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
        %91 = hivm.hir.pointer_cast(%c17952_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_78 = memref.cast %expand_shape_77 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_79 = memref.cast %91 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_float(%cast_78, %cast_79) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %92 = arith.index_cast %30 : i32 to index
        %93 = affine.apply #map7()[%92]
        %94 = arith.cmpi slt, %92, %c0 : index
        %95 = arith.cmpi sge, %92, %c16 : index
        %96 = arith.ori %94, %95 : i1
        scf.if %96 {
        } else {
          %subview_80 = memref.subview %79[0, 0] [%92, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_81 = memref.subview %91[0, 0] [%92, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_82 = memref.collapse_shape %subview_80 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_83 = memref.collapse_shape %subview_81 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_84 = memref.cast %collapse_shape_82 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_85 = memref.cast %collapse_shape_83 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_84, %cast_85, %c0_i32, %cst_1) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
          %subview_86 = memref.subview %91[0, 0] [%93, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_87 = memref.subview %79[0, 0] [%93, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_88 = memref.collapse_shape %subview_86 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_89 = memref.collapse_shape %subview_87 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_90 = memref.cast %collapse_shape_88 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_91 = memref.cast %collapse_shape_89 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_90, %cast_91, %c0_i32, %cst_1) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        }
      }
      %collapse_shape_43 = memref.collapse_shape %79 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_44 = memref.collapse_shape %43 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %63]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %80 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_45 = memref.cast %collapse_shape_43 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %collapse_shape_22 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_47 = memref.cast %collapse_shape_44 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %80 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_45, %cast_46, %cast_47, %cast_48) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_49 = memref.reinterpret_cast %arg4 to offset: [%72], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_50 = memref.subview %43[%26, 0] [%27, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_51 = memref.subview %reinterpret_cast_49[0, 0] [%27, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_52 = memref.cast %subview_50 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_53 = memref.cast %subview_51 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_52, %cast_53, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %63]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
