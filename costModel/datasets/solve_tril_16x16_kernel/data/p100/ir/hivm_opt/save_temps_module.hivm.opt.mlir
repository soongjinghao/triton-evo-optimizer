#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>
#map4 = affine_map<()[s0] -> (16, s0)>
#map5 = affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 16, 0)>
#map6 = affine_map<()[s0, s1] -> (16, s1, s0)>
#map7 = affine_map<()[s0, s1] -> (s0 - s1)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map9 = affine_map<()[s0] -> (s0 + 1)>
#map10 = affine_map<()[s0, s1] -> (0, s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @solve_tril_16x16_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmax_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
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
  func.func private @broadcast_first_axis_align_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @solve_tril_16x16_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c1600_i64 = arith.constant 1600 : i64
    %c2624_i64 = arith.constant 2624 : i64
    %c3648_i64 = arith.constant 3648 : i64
    %c4160_i64 = arith.constant 4160 : i64
    %c5184_i64 = arith.constant 5184 : i64
    %c7328_i64 = arith.constant 7328 : i64
    %c7456_i64 = arith.constant 7456 : i64
    %c8480_i64 = arith.constant 8480 : i64
    %c8992_i64 = arith.constant 8992 : i64
    %c9056_i64 = arith.constant 9056 : i64
    %c10080_i64 = arith.constant 10080 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c11104_i64 = arith.constant 11104 : i64
    %c5280_i64 = arith.constant 5280 : i64
    %c13152_i64 = arith.constant 13152 : i64
    %c7392_i64 = arith.constant 7392 : i64
    %c6304_i64 = arith.constant 6304 : i64
    %c12128_i64 = arith.constant 12128 : i64
    %0 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c12128_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c7392_i64) : memref<16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c13152_i64) : memref<16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c5280_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c11104_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg6, %arg7 : i32
    %7 = arith.muli %6, %arg8 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg8, %arg7 : i32
    %11 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast = memref.cast %11 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %expand_shape = memref.expand_shape %11 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c1088_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_1 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_2 = memref.cast %12 : memref<16x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_3 = memref.cast %13 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_last_axis_align_2d_int32_t(%cast_1, %cast_2, %cast_3) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %expand_shape_4 = memref.expand_shape %11 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi32, #hivm.address_space<ub>>
    %cast_5 = memref.cast %expand_shape_4 : memref<1x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_6 = memref.cast %14 : memref<16x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_first_axis_align_2d_int32_t(%cast_5, %cast_6) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %15 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c3648_i64) : memref<128xi32, #hivm.address_space<ub>>
    %cast_7 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_8 = memref.cast %expand_shape_4 : memref<1x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %15 : memref<16x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_10 = memref.cast %16 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vmax_2d_int32_t(%cast_7, %cast_8, %cast_9, %cast_10) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %17 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %15 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %14 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %17 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_13 = memref.cast %collapse_shape : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_14 = memref.cast %collapse_shape_11 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_15 = memref.cast %collapse_shape_12 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_int32_t(%cast_13, %cast_14, %cast_15) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %18 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %18 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %19 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_17 = memref.cast %collapse_shape_12 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_18 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_19 = memref.cast %19 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_17, %cast_18, %cast_19) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %20 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %12 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %20 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    %cast_22 = memref.cast %collapse_shape_20 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_23 = memref.cast %collapse_shape_11 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_24 = memref.cast %collapse_shape_21 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_int32_t(%cast_22, %cast_23, %cast_24) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %21 = arith.index_cast %arg5 : i32 to index
    %22 = hivm.hir.pointer_cast(%c4160_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %22 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c5184_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_26 = memref.cast %collapse_shape_21 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_27 = memref.cast %collapse_shape_25 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_28 = memref.cast %23 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_bool_to_float_1d_with_temp(%cast_26, %cast_27, %cast_28, %c0_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    scf.for %arg9 = %9 to %7 step %c48_i32  : i32 {
      %24 = arith.index_cast %arg9 : i32 to index
      %25 = arith.index_cast %9 : i32 to index
      %26 = affine.apply #map()[%24, %25]
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.cmpi eq, %27, %c1_i64 : i64
      %29 = arith.select %28, %1, %0 : memref<16x16xf32, #hivm.address_space<ub>>
      %30 = arith.index_cast %arg9 : i32 to index
      %31 = arith.index_cast %9 : i32 to index
      %32 = affine.apply #map()[%30, %31]
      %33 = arith.index_cast %32 : index to i64
      %34 = arith.cmpi eq, %33, %c1_i64 : i64
      %35 = arith.select %34, %3, %2 : memref<16xf32, #hivm.address_space<ub>>
      %36 = arith.index_cast %arg9 : i32 to index
      %37 = arith.index_cast %9 : i32 to index
      %38 = affine.apply #map()[%36, %37]
      %39 = arith.index_cast %38 : index to i64
      %40 = arith.cmpi eq, %39, %c1_i64 : i64
      %41 = arith.select %40, %5, %4 : memref<16x16xf32, #hivm.address_space<ub>>
      %42 = arith.index_cast %arg9 : i32 to index
      %43 = arith.index_cast %9 : i32 to index
      %44 = affine.apply #map()[%42, %43]
      %45 = arith.index_cast %44 : index to i1
      %46 = arith.select %45, %c3_i64, %c4_i64 : i64
      %47 = arith.index_cast %44 : index to i1
      %48 = arith.xori %47, %true : i1
      %49 = arith.extui %48 : i1 to i64
      hivm.hir.set_mask_norm
      %50 = arith.remsi %arg9, %arg8 : i32
      %51 = arith.divsi %arg9, %arg8 : i32
      %52 = arith.remsi %51, %arg7 : i32
      %53 = arith.divsi %arg9, %10 : i32
      %54 = arith.remsi %53, %arg6 : i32
      %55 = arith.divsi %52, %c8_i32 : i32
      %56 = arith.remsi %52, %c8_i32 : i32
      %57 = arith.muli %55, %arg5 : i32
      %58 = arith.muli %57, %c8_i32 : i32
      %59 = arith.addi %58, %56 : i32
      %60 = arith.muli %59, %c16_i32 : i32
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.muli %54, %c16_i32 : i32
      %63 = arith.remsi %62, %c16_i32 : i32
      %64 = arith.maxsi %62, %c0_i32 : i32
      %65 = arith.index_cast %64 : i32 to index
      %66 = arith.maxsi %63, %c0_i32 : i32
      %67 = arith.index_cast %66 : i32 to index
      %68 = affine.apply #map1()[%61, %65]
      %69 = affine.apply #map2()[%67, %61, %65]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %70 = affine.max #map3()[%21, %67, %65]
      %71 = affine.min #map4()[%70]
      %72 = affine.max #map5()[%67]
      %73 = affine.min #map4()[%72]
      %74 = arith.subi %c0_i32, %62 : i32
      %75 = arith.maxsi %74, %c0_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = affine.min #map6()[%76, %70]
      %78 = affine.apply #map7()[%71, %77]
      %79 = arith.subi %c0_i32, %63 : i32
      %80 = arith.maxsi %79, %c0_i32 : i32
      %81 = arith.index_cast %80 : i32 to index
      %82 = affine.min #map6()[%81, %72]
      %83 = affine.apply #map7()[%73, %82]
      %84 = arith.cmpi slt, %78, %c16 : index
      %85 = arith.cmpi slt, %83, %c16 : index
      %86 = arith.ori %84, %85 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%78, %83] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_29 = memref.subview %41[%77, %82] [%78, %83] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %86 {
        %collapse_shape_56 = memref.collapse_shape %41 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %cast_57 = memref.cast %collapse_shape_56 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_57) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %46]
      %cast_30 = memref.cast %subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_31 = memref.cast %subview_29 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_30, %cast_31, %c2_i32, %cst_0, %82) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_32 = memref.collapse_shape %41 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_33 = memref.collapse_shape %29 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %87 = hivm.hir.pointer_cast(%c7328_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %49]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_34 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %collapse_shape_32 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %collapse_shape_33 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %87 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_34, %cast_35, %cst_0, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %46]
      %collapse_shape_38 = memref.collapse_shape %29 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %88 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_39 = memref.cast %collapse_shape_33 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %collapse_shape_38 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %88 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_39, %cst, %cast_40, %cast_41) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_42 = memref.collapse_shape %29 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %89 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_43 = memref.cast %collapse_shape_38 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %89 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_43, %cst_0, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %90 = arith.addi %50, %c2_i32 : i32
      %91 = arith.subi %arg5, %62 : i32
      %92 = arith.minsi %91, %c16_i32 : i32
      %93 = arith.cmpi slt, %90, %92 : i32
      scf.if %93 {
        %99 = arith.addi %62, %90 : i32
        %100 = arith.muli %99, %c128_i32 : i32
        %101 = arith.index_cast %100 : i32 to index
        %102 = arith.index_cast %63 : i32 to index
        %103 = affine.apply #map8()[%102, %61, %101]
        %reinterpret_cast_56 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %49]
        %cast_57 = memref.cast %reinterpret_cast_56 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_58 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_57, %cast_58, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %104 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_59 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_60 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_61 = memref.cast %104 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_59, %cst, %cast_60, %cast_61) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %105 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_62 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_63 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_64 = memref.cast %105 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_62, %cst_0, %cast_63, %cast_64) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_65 = memref.expand_shape %35 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c7456_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        %107 = hivm.hir.pointer_cast(%c8480_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_66 = memref.cast %expand_shape_65 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_67 = memref.cast %29 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_68 = memref.cast %106 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_69 = memref.cast %107 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_66, %cast_67, %cast_68, %cast_69) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %108 = hivm.hir.pointer_cast(%c8992_i64) : memref<1x16xf32, #hivm.address_space<ub>>
        %109 = hivm.hir.pointer_cast(%c9056_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_70 = memref.cast %106 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_71 = memref.cast %108 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_72 = memref.cast %109 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_70, %cast_71, %cast_72, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_73 = memref.collapse_shape %108 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %110 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_74 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_75 = memref.cast %collapse_shape_73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_76 = memref.cast %35 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_77 = memref.cast %110 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_74, %cast_75, %cast_76, %cast_77) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_78 = memref.expand_shape %35 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
        %111 = hivm.hir.pointer_cast(%c10080_i64) : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_79 = memref.cast %expand_shape_78 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_80 = memref.cast %111 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_float(%cast_79, %cast_80) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %49]
        %112 = arith.index_cast %90 : i32 to index
        %113 = affine.apply #map9()[%112]
        %114 = arith.cmpi slt, %112, %c0 : index
        %115 = arith.cmpi sge, %112, %c16 : index
        %116 = arith.ori %114, %115 : i1
        scf.if %116 {
        } else {
          %subview_81 = memref.subview %29[0, 0] [%112, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_82 = memref.subview %111[0, 0] [%112, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_83 = memref.collapse_shape %subview_81 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_84 = memref.collapse_shape %subview_82 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_85 = memref.cast %collapse_shape_83 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_86 = memref.cast %collapse_shape_84 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_85, %cast_86, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
          %subview_87 = memref.subview %111[0, 0] [%113, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_88 = memref.subview %29[0, 0] [%113, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_89 = memref.collapse_shape %subview_87 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_90 = memref.collapse_shape %subview_88 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_91 = memref.cast %collapse_shape_89 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_92 = memref.cast %collapse_shape_90 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_91, %cast_92, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        }
      }
      %collapse_shape_46 = memref.collapse_shape %29 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %94 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_47 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %collapse_shape_25 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %collapse_shape_46 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_50 = memref.cast %94 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_47, %cast_48, %cast_49, %cast_50) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_51 = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %95 = affine.max #map10()[%21, %65]
      %96 = affine.min #map4()[%95]
      %97 = affine.min #map6()[%76, %95]
      %98 = affine.apply #map7()[%96, %97]
      %subview_52 = memref.subview %29[%97, 0] [%98, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_53 = memref.subview %reinterpret_cast_51[0, 0] [%98, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_54 = memref.cast %subview_52 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %subview_53 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_54, %cast_55, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %49]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
