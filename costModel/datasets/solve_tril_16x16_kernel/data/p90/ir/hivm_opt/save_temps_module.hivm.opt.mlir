#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>
#map4 = affine_map<()[s0] -> (16, s0)>
#map5 = affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 16, 0)>
#map6 = affine_map<()[s0, s1] -> (16, s1, s0)>
#map7 = affine_map<()[s0, s1] -> (s0 - s1)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map9 = affine_map<()[s0, s1] -> (0, s0 - s1)>
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
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c128_i32 = arith.constant 128 : i32
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c13_i32 = arith.constant 13 : i32
    %c12_i32 = arith.constant 12 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
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
    %c6304_i64 = arith.constant 6304 : i64
    %c7328_i64 = arith.constant 7328 : i64
    %c7392_i64 = arith.constant 7392 : i64
    %c8416_i64 = arith.constant 8416 : i64
    %c8544_i64 = arith.constant 8544 : i64
    %c9056_i64 = arith.constant 9056 : i64
    %c9120_i64 = arith.constant 9120 : i64
    %c10144_i64 = arith.constant 10144 : i64
    %c11168_i64 = arith.constant 11168 : i64
    %c12192_i64 = arith.constant 12192 : i64
    %c12320_i64 = arith.constant 12320 : i64
    %c12832_i64 = arith.constant 12832 : i64
    %c12896_i64 = arith.constant 12896 : i64
    %c13920_i64 = arith.constant 13920 : i64
    %c14944_i64 = arith.constant 14944 : i64
    %c15968_i64 = arith.constant 15968 : i64
    %c16096_i64 = arith.constant 16096 : i64
    %c16608_i64 = arith.constant 16608 : i64
    %c16672_i64 = arith.constant 16672 : i64
    %c17696_i64 = arith.constant 17696 : i64
    %c18720_i64 = arith.constant 18720 : i64
    %c19744_i64 = arith.constant 19744 : i64
    %c19872_i64 = arith.constant 19872 : i64
    %c20384_i64 = arith.constant 20384 : i64
    %c20448_i64 = arith.constant 20448 : i64
    %c21472_i64 = arith.constant 21472 : i64
    %c22496_i64 = arith.constant 22496 : i64
    %c23520_i64 = arith.constant 23520 : i64
    %c23648_i64 = arith.constant 23648 : i64
    %c24160_i64 = arith.constant 24160 : i64
    %c24224_i64 = arith.constant 24224 : i64
    %c25248_i64 = arith.constant 25248 : i64
    %c26272_i64 = arith.constant 26272 : i64
    %c27296_i64 = arith.constant 27296 : i64
    %c27424_i64 = arith.constant 27424 : i64
    %c27936_i64 = arith.constant 27936 : i64
    %c28000_i64 = arith.constant 28000 : i64
    %c29024_i64 = arith.constant 29024 : i64
    %c30048_i64 = arith.constant 30048 : i64
    %c31072_i64 = arith.constant 31072 : i64
    %c31200_i64 = arith.constant 31200 : i64
    %c31712_i64 = arith.constant 31712 : i64
    %c31776_i64 = arith.constant 31776 : i64
    %c32800_i64 = arith.constant 32800 : i64
    %c33824_i64 = arith.constant 33824 : i64
    %c34848_i64 = arith.constant 34848 : i64
    %c34976_i64 = arith.constant 34976 : i64
    %c35488_i64 = arith.constant 35488 : i64
    %c35552_i64 = arith.constant 35552 : i64
    %c36576_i64 = arith.constant 36576 : i64
    %c37600_i64 = arith.constant 37600 : i64
    %c38624_i64 = arith.constant 38624 : i64
    %c38752_i64 = arith.constant 38752 : i64
    %c39264_i64 = arith.constant 39264 : i64
    %c39328_i64 = arith.constant 39328 : i64
    %c40352_i64 = arith.constant 40352 : i64
    %c41376_i64 = arith.constant 41376 : i64
    %c42400_i64 = arith.constant 42400 : i64
    %c42528_i64 = arith.constant 42528 : i64
    %c43040_i64 = arith.constant 43040 : i64
    %c43104_i64 = arith.constant 43104 : i64
    %c44128_i64 = arith.constant 44128 : i64
    %c45152_i64 = arith.constant 45152 : i64
    %c46176_i64 = arith.constant 46176 : i64
    %c46304_i64 = arith.constant 46304 : i64
    %c46816_i64 = arith.constant 46816 : i64
    %c46880_i64 = arith.constant 46880 : i64
    %c47904_i64 = arith.constant 47904 : i64
    %c48928_i64 = arith.constant 48928 : i64
    %c49952_i64 = arith.constant 49952 : i64
    %c50080_i64 = arith.constant 50080 : i64
    %c50592_i64 = arith.constant 50592 : i64
    %c50656_i64 = arith.constant 50656 : i64
    %c51680_i64 = arith.constant 51680 : i64
    %c52704_i64 = arith.constant 52704 : i64
    %c53728_i64 = arith.constant 53728 : i64
    %c53856_i64 = arith.constant 53856 : i64
    %c54368_i64 = arith.constant 54368 : i64
    %c54432_i64 = arith.constant 54432 : i64
    %c55456_i64 = arith.constant 55456 : i64
    %c56480_i64 = arith.constant 56480 : i64
    %c57504_i64 = arith.constant 57504 : i64
    %c57632_i64 = arith.constant 57632 : i64
    %c58144_i64 = arith.constant 58144 : i64
    %c58208_i64 = arith.constant 58208 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c60256_i64 = arith.constant 60256 : i64
    %c5280_i64 = arith.constant 5280 : i64
    %c61280_i64 = arith.constant 61280 : i64
    %c8480_i64 = arith.constant 8480 : i64
    %c61344_i64 = arith.constant 61344 : i64
    %c12256_i64 = arith.constant 12256 : i64
    %c61408_i64 = arith.constant 61408 : i64
    %c16032_i64 = arith.constant 16032 : i64
    %c61472_i64 = arith.constant 61472 : i64
    %c19808_i64 = arith.constant 19808 : i64
    %c61536_i64 = arith.constant 61536 : i64
    %c23584_i64 = arith.constant 23584 : i64
    %c61600_i64 = arith.constant 61600 : i64
    %c27360_i64 = arith.constant 27360 : i64
    %c61664_i64 = arith.constant 61664 : i64
    %c31136_i64 = arith.constant 31136 : i64
    %c61728_i64 = arith.constant 61728 : i64
    %c34912_i64 = arith.constant 34912 : i64
    %c61792_i64 = arith.constant 61792 : i64
    %c38688_i64 = arith.constant 38688 : i64
    %c61856_i64 = arith.constant 61856 : i64
    %c42464_i64 = arith.constant 42464 : i64
    %c61920_i64 = arith.constant 61920 : i64
    %c46240_i64 = arith.constant 46240 : i64
    %c61984_i64 = arith.constant 61984 : i64
    %c50016_i64 = arith.constant 50016 : i64
    %c62048_i64 = arith.constant 62048 : i64
    %c53792_i64 = arith.constant 53792 : i64
    %c62112_i64 = arith.constant 62112 : i64
    %c57568_i64 = arith.constant 57568 : i64
    %c59232_i64 = arith.constant 59232 : i64
    %c62176_i64 = arith.constant 62176 : i64
    %0 = hivm.hir.pointer_cast(%c59232_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c62176_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c57568_i64) : memref<16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c62112_i64) : memref<16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c53792_i64) : memref<16xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c62048_i64) : memref<16xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c50016_i64) : memref<16xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c61984_i64) : memref<16xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c46240_i64) : memref<16xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c61920_i64) : memref<16xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c42464_i64) : memref<16xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c61856_i64) : memref<16xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c38688_i64) : memref<16xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c61792_i64) : memref<16xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c34912_i64) : memref<16xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c61728_i64) : memref<16xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c31136_i64) : memref<16xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c61664_i64) : memref<16xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c27360_i64) : memref<16xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c61600_i64) : memref<16xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c23584_i64) : memref<16xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c61536_i64) : memref<16xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c19808_i64) : memref<16xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c61472_i64) : memref<16xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c16032_i64) : memref<16xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c61408_i64) : memref<16xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c12256_i64) : memref<16xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c61344_i64) : memref<16xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c8480_i64) : memref<16xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c61280_i64) : memref<16xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c5280_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c60256_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %32 = arith.muli %arg6, %arg7 : i32
    %33 = arith.muli %32, %arg8 : i32
    %34 = hivm.hir.get_block_idx -> i64
    %35 = arith.trunci %34 : i64 to i32
    %36 = arith.muli %arg8, %arg7 : i32
    %37 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast = memref.cast %37 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %expand_shape = memref.expand_shape %37 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
    %38 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c1088_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_1 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_2 = memref.cast %38 : memref<16x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_3 = memref.cast %39 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_last_axis_align_2d_int32_t(%cast_1, %cast_2, %cast_3) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %expand_shape_4 = memref.expand_shape %37 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
    %40 = hivm.hir.pointer_cast(%c1600_i64) : memref<16x16xi32, #hivm.address_space<ub>>
    %cast_5 = memref.cast %expand_shape_4 : memref<1x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_6 = memref.cast %40 : memref<16x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_first_axis_align_2d_int32_t(%cast_5, %cast_6) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %41 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c3648_i64) : memref<128xi32, #hivm.address_space<ub>>
    %cast_7 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_8 = memref.cast %expand_shape_4 : memref<1x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %41 : memref<16x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_10 = memref.cast %42 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vmax_2d_int32_t(%cast_7, %cast_8, %cast_9, %cast_10) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %43 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %41 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %40 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %43 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_13 = memref.cast %collapse_shape : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_14 = memref.cast %collapse_shape_11 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_15 = memref.cast %collapse_shape_12 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_int32_t(%cast_13, %cast_14, %cast_15) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %44 = hivm.hir.pointer_cast(%c2624_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %44 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %45 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_17 = memref.cast %collapse_shape_12 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_18 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_19 = memref.cast %45 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_17, %cast_18, %cast_19) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %46 = hivm.hir.pointer_cast(%c64_i64) : memref<16x16xi1, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %38 [[0, 1]] : memref<16x16xi32, #hivm.address_space<ub>> into memref<256xi32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %46 [[0, 1]] : memref<16x16xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
    %cast_22 = memref.cast %collapse_shape_20 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_23 = memref.cast %collapse_shape_11 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_24 = memref.cast %collapse_shape_21 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_int32_t(%cast_22, %cast_23, %cast_24) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %47 = arith.index_cast %arg5 : i32 to index
    %48 = hivm.hir.pointer_cast(%c4160_i64) : memref<16x16xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %48 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c5184_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_26 = memref.cast %collapse_shape_21 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_27 = memref.cast %collapse_shape_25 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_28 = memref.cast %49 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_bool_to_float_1d_with_temp(%cast_26, %cast_27, %cast_28, %c0_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
    scf.for %arg9 = %35 to %33 step %c48_i32  : i32 {
      %50 = arith.index_cast %arg9 : i32 to index
      %51 = arith.index_cast %35 : i32 to index
      %52 = affine.apply #map()[%50, %51]
      %53 = arith.index_cast %52 : index to i64
      %54 = arith.cmpi eq, %53, %c1_i64 : i64
      %55 = arith.select %54, %1, %0 : memref<16x16xf32, #hivm.address_space<ub>>
      %56 = arith.index_cast %arg9 : i32 to index
      %57 = arith.index_cast %35 : i32 to index
      %58 = affine.apply #map()[%56, %57]
      %59 = arith.index_cast %58 : index to i64
      %60 = arith.cmpi eq, %59, %c1_i64 : i64
      %61 = arith.select %60, %3, %2 : memref<16xf32, #hivm.address_space<ub>>
      %62 = arith.index_cast %arg9 : i32 to index
      %63 = arith.index_cast %35 : i32 to index
      %64 = affine.apply #map()[%62, %63]
      %65 = arith.index_cast %64 : index to i64
      %66 = arith.cmpi eq, %65, %c1_i64 : i64
      %67 = arith.select %66, %5, %4 : memref<16xf32, #hivm.address_space<ub>>
      %68 = arith.index_cast %arg9 : i32 to index
      %69 = arith.index_cast %35 : i32 to index
      %70 = affine.apply #map()[%68, %69]
      %71 = arith.index_cast %70 : index to i64
      %72 = arith.cmpi eq, %71, %c1_i64 : i64
      %73 = arith.select %72, %7, %6 : memref<16xf32, #hivm.address_space<ub>>
      %74 = arith.index_cast %arg9 : i32 to index
      %75 = arith.index_cast %35 : i32 to index
      %76 = affine.apply #map()[%74, %75]
      %77 = arith.index_cast %76 : index to i64
      %78 = arith.cmpi eq, %77, %c1_i64 : i64
      %79 = arith.select %78, %9, %8 : memref<16xf32, #hivm.address_space<ub>>
      %80 = arith.index_cast %arg9 : i32 to index
      %81 = arith.index_cast %35 : i32 to index
      %82 = affine.apply #map()[%80, %81]
      %83 = arith.index_cast %82 : index to i64
      %84 = arith.cmpi eq, %83, %c1_i64 : i64
      %85 = arith.select %84, %11, %10 : memref<16xf32, #hivm.address_space<ub>>
      %86 = arith.index_cast %arg9 : i32 to index
      %87 = arith.index_cast %35 : i32 to index
      %88 = affine.apply #map()[%86, %87]
      %89 = arith.index_cast %88 : index to i64
      %90 = arith.cmpi eq, %89, %c1_i64 : i64
      %91 = arith.select %90, %13, %12 : memref<16xf32, #hivm.address_space<ub>>
      %92 = arith.index_cast %arg9 : i32 to index
      %93 = arith.index_cast %35 : i32 to index
      %94 = affine.apply #map()[%92, %93]
      %95 = arith.index_cast %94 : index to i64
      %96 = arith.cmpi eq, %95, %c1_i64 : i64
      %97 = arith.select %96, %15, %14 : memref<16xf32, #hivm.address_space<ub>>
      %98 = arith.index_cast %arg9 : i32 to index
      %99 = arith.index_cast %35 : i32 to index
      %100 = affine.apply #map()[%98, %99]
      %101 = arith.index_cast %100 : index to i64
      %102 = arith.cmpi eq, %101, %c1_i64 : i64
      %103 = arith.select %102, %17, %16 : memref<16xf32, #hivm.address_space<ub>>
      %104 = arith.index_cast %arg9 : i32 to index
      %105 = arith.index_cast %35 : i32 to index
      %106 = affine.apply #map()[%104, %105]
      %107 = arith.index_cast %106 : index to i64
      %108 = arith.cmpi eq, %107, %c1_i64 : i64
      %109 = arith.select %108, %19, %18 : memref<16xf32, #hivm.address_space<ub>>
      %110 = arith.index_cast %arg9 : i32 to index
      %111 = arith.index_cast %35 : i32 to index
      %112 = affine.apply #map()[%110, %111]
      %113 = arith.index_cast %112 : index to i64
      %114 = arith.cmpi eq, %113, %c1_i64 : i64
      %115 = arith.select %114, %21, %20 : memref<16xf32, #hivm.address_space<ub>>
      %116 = arith.index_cast %arg9 : i32 to index
      %117 = arith.index_cast %35 : i32 to index
      %118 = affine.apply #map()[%116, %117]
      %119 = arith.index_cast %118 : index to i64
      %120 = arith.cmpi eq, %119, %c1_i64 : i64
      %121 = arith.select %120, %23, %22 : memref<16xf32, #hivm.address_space<ub>>
      %122 = arith.index_cast %arg9 : i32 to index
      %123 = arith.index_cast %35 : i32 to index
      %124 = affine.apply #map()[%122, %123]
      %125 = arith.index_cast %124 : index to i64
      %126 = arith.cmpi eq, %125, %c1_i64 : i64
      %127 = arith.select %126, %25, %24 : memref<16xf32, #hivm.address_space<ub>>
      %128 = arith.index_cast %arg9 : i32 to index
      %129 = arith.index_cast %35 : i32 to index
      %130 = affine.apply #map()[%128, %129]
      %131 = arith.index_cast %130 : index to i64
      %132 = arith.cmpi eq, %131, %c1_i64 : i64
      %133 = arith.select %132, %27, %26 : memref<16xf32, #hivm.address_space<ub>>
      %134 = arith.index_cast %arg9 : i32 to index
      %135 = arith.index_cast %35 : i32 to index
      %136 = affine.apply #map()[%134, %135]
      %137 = arith.index_cast %136 : index to i64
      %138 = arith.cmpi eq, %137, %c1_i64 : i64
      %139 = arith.select %138, %29, %28 : memref<16xf32, #hivm.address_space<ub>>
      %140 = arith.index_cast %arg9 : i32 to index
      %141 = arith.index_cast %35 : i32 to index
      %142 = affine.apply #map()[%140, %141]
      %143 = arith.index_cast %142 : index to i64
      %144 = arith.cmpi eq, %143, %c1_i64 : i64
      %145 = arith.select %144, %31, %30 : memref<16x16xf32, #hivm.address_space<ub>>
      %146 = arith.index_cast %arg9 : i32 to index
      %147 = arith.index_cast %35 : i32 to index
      %148 = affine.apply #map()[%146, %147]
      %149 = arith.index_cast %148 : index to i1
      %150 = arith.select %149, %c5_i64, %c6_i64 : i64
      %151 = arith.index_cast %148 : index to i1
      %152 = arith.select %151, %c3_i64, %c4_i64 : i64
      %153 = arith.index_cast %148 : index to i1
      %154 = arith.select %153, %c1_i64, %c2_i64 : i64
      %155 = arith.index_cast %148 : index to i1
      %156 = arith.xori %155, %true : i1
      %157 = arith.extui %156 : i1 to i64
      hivm.hir.set_mask_norm
      %158 = arith.divsi %arg9, %arg8 : i32
      %159 = arith.remsi %158, %arg7 : i32
      %160 = arith.divsi %arg9, %36 : i32
      %161 = arith.remsi %160, %arg6 : i32
      %162 = arith.divsi %159, %c8_i32 : i32
      %163 = arith.remsi %159, %c8_i32 : i32
      %164 = arith.muli %162, %arg5 : i32
      %165 = arith.muli %164, %c8_i32 : i32
      %166 = arith.addi %165, %163 : i32
      %167 = arith.muli %166, %c16_i32 : i32
      %168 = arith.index_cast %167 : i32 to index
      %169 = arith.muli %161, %c16_i32 : i32
      %170 = arith.remsi %169, %c16_i32 : i32
      %171 = arith.maxsi %169, %c0_i32 : i32
      %172 = arith.index_cast %171 : i32 to index
      %173 = arith.maxsi %170, %c0_i32 : i32
      %174 = arith.index_cast %173 : i32 to index
      %175 = affine.apply #map1()[%168, %172]
      %176 = affine.apply #map2()[%174, %168, %172]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%176], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %177 = affine.max #map3()[%47, %174, %172]
      %178 = affine.min #map4()[%177]
      %179 = affine.max #map5()[%174]
      %180 = affine.min #map4()[%179]
      %181 = arith.subi %c0_i32, %169 : i32
      %182 = arith.maxsi %181, %c0_i32 : i32
      %183 = arith.index_cast %182 : i32 to index
      %184 = affine.min #map6()[%183, %177]
      %185 = affine.apply #map7()[%178, %184]
      %186 = arith.subi %c0_i32, %170 : i32
      %187 = arith.maxsi %186, %c0_i32 : i32
      %188 = arith.index_cast %187 : i32 to index
      %189 = affine.min #map6()[%188, %179]
      %190 = affine.apply #map7()[%180, %189]
      %191 = arith.cmpi slt, %185, %c16 : index
      %192 = arith.cmpi slt, %190, %c16 : index
      %193 = arith.ori %191, %192 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%185, %190] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_29 = memref.subview %145[%184, %189] [%185, %190] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %193 {
        %collapse_shape_613 = memref.collapse_shape %145 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        %cast_614 = memref.cast %collapse_shape_613 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_614) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %154]
      %cast_30 = memref.cast %subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_31 = memref.cast %subview_29 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_30, %cast_31, %c2_i32, %cst_0, %189) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %194 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_32 = memref.collapse_shape %145 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_33 = memref.collapse_shape %194 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %195 = hivm.hir.pointer_cast(%c7328_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_34 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %collapse_shape_32 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %collapse_shape_33 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %195 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_34, %cast_35, %cst_0, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %196 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_38 = memref.collapse_shape %196 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %197 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_39 = memref.cast %collapse_shape_33 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %collapse_shape_38 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %197 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_39, %cst, %cast_40, %cast_41) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %198 = hivm.hir.pointer_cast(%c6304_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_42 = memref.collapse_shape %198 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %199 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_43 = memref.cast %collapse_shape_38 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %199 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_43, %cst_0, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %200 = hivm.hir.pointer_cast(%c7392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_46 = memref.collapse_shape %200 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %201 = hivm.hir.pointer_cast(%c8416_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_47 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %collapse_shape_46 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_50 = memref.cast %201 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_47, %cast_48, %cst_0, %cast_49, %cast_50) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %202 = arith.addi %169, %c2_i32 : i32
      %203 = arith.muli %202, %c128_i32 : i32
      %204 = arith.index_cast %203 : i32 to index
      %205 = arith.index_cast %170 : i32 to index
      %206 = affine.apply #map8()[%205, %168, %204]
      %reinterpret_cast_51 = memref.reinterpret_cast %arg3 to offset: [%206], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_52 = memref.cast %reinterpret_cast_51 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_53 = memref.cast %139 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_52, %cast_53, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %207 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_54 = memref.cast %139 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %139 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %207 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_54, %cst, %cast_55, %cast_56) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %208 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_57 = memref.cast %139 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_58 = memref.cast %139 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_59 = memref.cast %208 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_57, %cst_0, %cast_58, %cast_59) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_60 = memref.expand_shape %139 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %209 = hivm.hir.pointer_cast(%c7392_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %210 = hivm.hir.pointer_cast(%c8544_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_61 = memref.cast %expand_shape_60 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_62 = memref.cast %200 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_63 = memref.cast %209 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_64 = memref.cast %210 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_61, %cast_62, %cast_63, %cast_64) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %211 = hivm.hir.pointer_cast(%c9056_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %212 = hivm.hir.pointer_cast(%c9120_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_65 = memref.cast %209 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_66 = memref.cast %211 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_67 = memref.cast %212 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_65, %cast_66, %cast_67, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_68 = memref.collapse_shape %211 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %213 = hivm.hir.pointer_cast(%c9056_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %214 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_69 = memref.cast %139 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_70 = memref.cast %collapse_shape_68 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_71 = memref.cast %213 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_72 = memref.cast %214 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_69, %cast_70, %cast_71, %cast_72) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_73 = memref.expand_shape %213 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %215 = hivm.hir.pointer_cast(%c10144_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_74 = memref.cast %expand_shape_73 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_75 = memref.cast %215 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_74, %cast_75) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_76 = memref.subview %198[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_77 = memref.subview %215[0, 0] [2, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_78 = memref.collapse_shape %subview_76 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_79 = memref.collapse_shape %subview_77 [[0, 1]] : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_80 = memref.cast %collapse_shape_78 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_81 = memref.cast %collapse_shape_79 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_80, %cast_81, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_82 = memref.subview %215[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_83 = memref.subview %198[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_84 = memref.collapse_shape %subview_82 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_85 = memref.collapse_shape %subview_83 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_86 = memref.cast %collapse_shape_84 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_87 = memref.cast %collapse_shape_85 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_86, %cast_87, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %216 = hivm.hir.pointer_cast(%c11168_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_88 = memref.collapse_shape %216 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %217 = hivm.hir.pointer_cast(%c12192_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_89 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_90 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_91 = memref.cast %collapse_shape_88 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_92 = memref.cast %217 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_89, %cast_90, %cst_0, %cast_91, %cast_92) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %218 = arith.addi %169, %c3_i32 : i32
      %219 = arith.muli %218, %c128_i32 : i32
      %220 = arith.index_cast %219 : i32 to index
      %221 = affine.apply #map8()[%205, %168, %220]
      %reinterpret_cast_93 = memref.reinterpret_cast %arg3 to offset: [%221], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_94 = memref.cast %reinterpret_cast_93 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_95 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_94, %cast_95, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %222 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_96 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_97 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_98 = memref.cast %222 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_96, %cst, %cast_97, %cast_98) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %223 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_99 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_100 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_101 = memref.cast %223 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_99, %cst_0, %cast_100, %cast_101) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_102 = memref.expand_shape %133 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %224 = hivm.hir.pointer_cast(%c11168_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %225 = hivm.hir.pointer_cast(%c12320_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_103 = memref.cast %expand_shape_102 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_104 = memref.cast %216 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_105 = memref.cast %224 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_106 = memref.cast %225 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_103, %cast_104, %cast_105, %cast_106) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %226 = hivm.hir.pointer_cast(%c12832_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %227 = hivm.hir.pointer_cast(%c12896_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_107 = memref.cast %224 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_108 = memref.cast %226 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_109 = memref.cast %227 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_107, %cast_108, %cast_109, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_110 = memref.collapse_shape %226 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %228 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_111 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_112 = memref.cast %collapse_shape_110 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_113 = memref.cast %133 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_114 = memref.cast %228 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_111, %cast_112, %cast_113, %cast_114) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_115 = memref.expand_shape %133 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %229 = hivm.hir.pointer_cast(%c13920_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_116 = memref.cast %expand_shape_115 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_117 = memref.cast %229 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_116, %cast_117) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_118 = memref.subview %229[0, 0] [3, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_119 = memref.collapse_shape %subview_118 [[0, 1]] : memref<3x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<48xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_120 = memref.cast %collapse_shape_84 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_121 = memref.cast %collapse_shape_119 : memref<48xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_120, %cast_121, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_122 = memref.subview %229[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_123 = memref.subview %198[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_124 = memref.collapse_shape %subview_122 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_125 = memref.collapse_shape %subview_123 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_126 = memref.cast %collapse_shape_124 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_127 = memref.cast %collapse_shape_125 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_126, %cast_127, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %230 = hivm.hir.pointer_cast(%c14944_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_128 = memref.collapse_shape %230 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %231 = hivm.hir.pointer_cast(%c15968_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_129 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_130 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_131 = memref.cast %collapse_shape_128 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_132 = memref.cast %231 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_129, %cast_130, %cst_0, %cast_131, %cast_132) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %232 = arith.addi %169, %c4_i32 : i32
      %233 = arith.muli %232, %c128_i32 : i32
      %234 = arith.index_cast %233 : i32 to index
      %235 = affine.apply #map8()[%205, %168, %234]
      %reinterpret_cast_133 = memref.reinterpret_cast %arg3 to offset: [%235], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_134 = memref.cast %reinterpret_cast_133 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_135 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_134, %cast_135, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %236 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_136 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_137 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_138 = memref.cast %236 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_136, %cst, %cast_137, %cast_138) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %237 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_139 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_140 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_141 = memref.cast %237 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_139, %cst_0, %cast_140, %cast_141) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_142 = memref.expand_shape %127 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %238 = hivm.hir.pointer_cast(%c14944_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %239 = hivm.hir.pointer_cast(%c16096_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_143 = memref.cast %expand_shape_142 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_144 = memref.cast %230 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_145 = memref.cast %238 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_146 = memref.cast %239 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_143, %cast_144, %cast_145, %cast_146) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %240 = hivm.hir.pointer_cast(%c16608_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %241 = hivm.hir.pointer_cast(%c16672_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_147 = memref.cast %238 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_148 = memref.cast %240 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_149 = memref.cast %241 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_147, %cast_148, %cast_149, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_150 = memref.collapse_shape %240 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %242 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_151 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_152 = memref.cast %collapse_shape_150 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_153 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_154 = memref.cast %242 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_151, %cast_152, %cast_153, %cast_154) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_155 = memref.expand_shape %127 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %243 = hivm.hir.pointer_cast(%c17696_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_156 = memref.cast %expand_shape_155 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_157 = memref.cast %243 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_156, %cast_157) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_158 = memref.subview %243[0, 0] [4, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_159 = memref.collapse_shape %subview_158 [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_160 = memref.cast %collapse_shape_124 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_161 = memref.cast %collapse_shape_159 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_160, %cast_161, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_162 = memref.subview %243[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_163 = memref.subview %198[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_164 = memref.collapse_shape %subview_162 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_165 = memref.collapse_shape %subview_163 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_166 = memref.cast %collapse_shape_164 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_167 = memref.cast %collapse_shape_165 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_166, %cast_167, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %244 = hivm.hir.pointer_cast(%c18720_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_168 = memref.collapse_shape %244 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %245 = hivm.hir.pointer_cast(%c19744_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_169 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_170 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_171 = memref.cast %collapse_shape_168 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_172 = memref.cast %245 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_169, %cast_170, %cst_0, %cast_171, %cast_172) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %246 = arith.addi %169, %c5_i32 : i32
      %247 = arith.muli %246, %c128_i32 : i32
      %248 = arith.index_cast %247 : i32 to index
      %249 = affine.apply #map8()[%205, %168, %248]
      %reinterpret_cast_173 = memref.reinterpret_cast %arg3 to offset: [%249], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_174 = memref.cast %reinterpret_cast_173 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_175 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_174, %cast_175, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %250 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_176 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_177 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_178 = memref.cast %250 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_176, %cst, %cast_177, %cast_178) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %251 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_179 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_180 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_181 = memref.cast %251 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_179, %cst_0, %cast_180, %cast_181) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_182 = memref.expand_shape %121 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %252 = hivm.hir.pointer_cast(%c18720_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %253 = hivm.hir.pointer_cast(%c19872_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_183 = memref.cast %expand_shape_182 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_184 = memref.cast %244 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_185 = memref.cast %252 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_186 = memref.cast %253 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_183, %cast_184, %cast_185, %cast_186) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %254 = hivm.hir.pointer_cast(%c20384_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %255 = hivm.hir.pointer_cast(%c20448_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_187 = memref.cast %252 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_188 = memref.cast %254 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_189 = memref.cast %255 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_187, %cast_188, %cast_189, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_190 = memref.collapse_shape %254 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %256 = hivm.hir.pointer_cast(%c20384_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %257 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_191 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_192 = memref.cast %collapse_shape_190 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_193 = memref.cast %256 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_194 = memref.cast %257 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_191, %cast_192, %cast_193, %cast_194) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_195 = memref.expand_shape %256 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %258 = hivm.hir.pointer_cast(%c21472_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_196 = memref.cast %expand_shape_195 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_197 = memref.cast %258 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_196, %cast_197) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_198 = memref.subview %258[0, 0] [5, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_199 = memref.collapse_shape %subview_198 [[0, 1]] : memref<5x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<80xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_200 = memref.cast %collapse_shape_164 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_201 = memref.cast %collapse_shape_199 : memref<80xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_200, %cast_201, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_202 = memref.subview %258[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_203 = memref.subview %198[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_204 = memref.collapse_shape %subview_202 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_205 = memref.collapse_shape %subview_203 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_206 = memref.cast %collapse_shape_204 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_207 = memref.cast %collapse_shape_205 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_206, %cast_207, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %259 = hivm.hir.pointer_cast(%c22496_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_208 = memref.collapse_shape %259 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %260 = hivm.hir.pointer_cast(%c23520_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_209 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_210 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_211 = memref.cast %collapse_shape_208 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_212 = memref.cast %260 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_209, %cast_210, %cst_0, %cast_211, %cast_212) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %261 = arith.addi %169, %c6_i32 : i32
      %262 = arith.muli %261, %c128_i32 : i32
      %263 = arith.index_cast %262 : i32 to index
      %264 = affine.apply #map8()[%205, %168, %263]
      %reinterpret_cast_213 = memref.reinterpret_cast %arg3 to offset: [%264], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_214 = memref.cast %reinterpret_cast_213 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_215 = memref.cast %115 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_214, %cast_215, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %265 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_216 = memref.cast %115 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_217 = memref.cast %115 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_218 = memref.cast %265 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_216, %cst, %cast_217, %cast_218) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %266 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_219 = memref.cast %115 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_220 = memref.cast %115 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_221 = memref.cast %266 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_219, %cst_0, %cast_220, %cast_221) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_222 = memref.expand_shape %115 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %267 = hivm.hir.pointer_cast(%c22496_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %268 = hivm.hir.pointer_cast(%c23648_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_223 = memref.cast %expand_shape_222 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_224 = memref.cast %259 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_225 = memref.cast %267 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_226 = memref.cast %268 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_223, %cast_224, %cast_225, %cast_226) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %269 = hivm.hir.pointer_cast(%c24160_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %270 = hivm.hir.pointer_cast(%c24224_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_227 = memref.cast %267 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_228 = memref.cast %269 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_229 = memref.cast %270 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_227, %cast_228, %cast_229, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_230 = memref.collapse_shape %269 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %271 = hivm.hir.pointer_cast(%c24160_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %272 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_231 = memref.cast %115 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_232 = memref.cast %collapse_shape_230 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_233 = memref.cast %271 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_234 = memref.cast %272 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_231, %cast_232, %cast_233, %cast_234) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_235 = memref.expand_shape %271 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %273 = hivm.hir.pointer_cast(%c25248_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_236 = memref.cast %expand_shape_235 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_237 = memref.cast %273 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_236, %cast_237) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_238 = memref.subview %273[0, 0] [6, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_239 = memref.collapse_shape %subview_238 [[0, 1]] : memref<6x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<96xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_240 = memref.cast %collapse_shape_204 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_241 = memref.cast %collapse_shape_239 : memref<96xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_240, %cast_241, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_242 = memref.subview %273[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_243 = memref.subview %198[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_244 = memref.collapse_shape %subview_242 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_245 = memref.collapse_shape %subview_243 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_246 = memref.cast %collapse_shape_244 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_247 = memref.cast %collapse_shape_245 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_246, %cast_247, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %274 = hivm.hir.pointer_cast(%c26272_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_248 = memref.collapse_shape %274 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %275 = hivm.hir.pointer_cast(%c27296_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_249 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_250 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_251 = memref.cast %collapse_shape_248 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_252 = memref.cast %275 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_249, %cast_250, %cst_0, %cast_251, %cast_252) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %276 = arith.addi %169, %c7_i32 : i32
      %277 = arith.muli %276, %c128_i32 : i32
      %278 = arith.index_cast %277 : i32 to index
      %279 = affine.apply #map8()[%205, %168, %278]
      %reinterpret_cast_253 = memref.reinterpret_cast %arg3 to offset: [%279], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_254 = memref.cast %reinterpret_cast_253 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_255 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_254, %cast_255, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %280 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_256 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_257 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_258 = memref.cast %280 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_256, %cst, %cast_257, %cast_258) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %281 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_259 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_260 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_261 = memref.cast %281 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_259, %cst_0, %cast_260, %cast_261) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_262 = memref.expand_shape %109 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %282 = hivm.hir.pointer_cast(%c26272_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %283 = hivm.hir.pointer_cast(%c27424_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_263 = memref.cast %expand_shape_262 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_264 = memref.cast %274 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_265 = memref.cast %282 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_266 = memref.cast %283 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_263, %cast_264, %cast_265, %cast_266) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %284 = hivm.hir.pointer_cast(%c27936_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %285 = hivm.hir.pointer_cast(%c28000_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_267 = memref.cast %282 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_268 = memref.cast %284 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_269 = memref.cast %285 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_267, %cast_268, %cast_269, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_270 = memref.collapse_shape %284 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %286 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_271 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_272 = memref.cast %collapse_shape_270 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_273 = memref.cast %109 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_274 = memref.cast %286 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_271, %cast_272, %cast_273, %cast_274) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_275 = memref.expand_shape %109 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %287 = hivm.hir.pointer_cast(%c29024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_276 = memref.cast %expand_shape_275 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_277 = memref.cast %287 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_276, %cast_277) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_278 = memref.subview %287[0, 0] [7, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_279 = memref.collapse_shape %subview_278 [[0, 1]] : memref<7x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<112xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_280 = memref.cast %collapse_shape_244 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_281 = memref.cast %collapse_shape_279 : memref<112xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_280, %cast_281, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_282 = memref.subview %287[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_283 = memref.subview %198[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_284 = memref.collapse_shape %subview_282 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_285 = memref.collapse_shape %subview_283 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_286 = memref.cast %collapse_shape_284 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_287 = memref.cast %collapse_shape_285 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_286, %cast_287, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %288 = hivm.hir.pointer_cast(%c30048_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_288 = memref.collapse_shape %288 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %289 = hivm.hir.pointer_cast(%c31072_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_289 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_290 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_291 = memref.cast %collapse_shape_288 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_292 = memref.cast %289 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_289, %cast_290, %cst_0, %cast_291, %cast_292) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %290 = arith.addi %169, %c8_i32 : i32
      %291 = arith.muli %290, %c128_i32 : i32
      %292 = arith.index_cast %291 : i32 to index
      %293 = affine.apply #map8()[%205, %168, %292]
      %reinterpret_cast_293 = memref.reinterpret_cast %arg3 to offset: [%293], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_294 = memref.cast %reinterpret_cast_293 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_295 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_294, %cast_295, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %294 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_296 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_297 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_298 = memref.cast %294 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_296, %cst, %cast_297, %cast_298) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %295 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_299 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_300 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_301 = memref.cast %295 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_299, %cst_0, %cast_300, %cast_301) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_302 = memref.expand_shape %103 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %296 = hivm.hir.pointer_cast(%c30048_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %297 = hivm.hir.pointer_cast(%c31200_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_303 = memref.cast %expand_shape_302 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_304 = memref.cast %288 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_305 = memref.cast %296 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_306 = memref.cast %297 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_303, %cast_304, %cast_305, %cast_306) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %298 = hivm.hir.pointer_cast(%c31712_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %299 = hivm.hir.pointer_cast(%c31776_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_307 = memref.cast %296 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_308 = memref.cast %298 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_309 = memref.cast %299 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_307, %cast_308, %cast_309, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_310 = memref.collapse_shape %298 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %300 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_311 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_312 = memref.cast %collapse_shape_310 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_313 = memref.cast %103 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_314 = memref.cast %300 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_311, %cast_312, %cast_313, %cast_314) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_315 = memref.expand_shape %103 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %301 = hivm.hir.pointer_cast(%c32800_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_316 = memref.cast %expand_shape_315 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_317 = memref.cast %301 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_316, %cast_317) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_318 = memref.subview %301[0, 0] [8, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_319 = memref.collapse_shape %subview_318 [[0, 1]] : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<128xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_320 = memref.cast %collapse_shape_284 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_321 = memref.cast %collapse_shape_319 : memref<128xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_320, %cast_321, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_322 = memref.subview %301[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_323 = memref.subview %198[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_324 = memref.collapse_shape %subview_322 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_325 = memref.collapse_shape %subview_323 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_326 = memref.cast %collapse_shape_324 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_327 = memref.cast %collapse_shape_325 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_326, %cast_327, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %302 = hivm.hir.pointer_cast(%c33824_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_328 = memref.collapse_shape %302 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %303 = hivm.hir.pointer_cast(%c34848_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_329 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_330 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_331 = memref.cast %collapse_shape_328 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_332 = memref.cast %303 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_329, %cast_330, %cst_0, %cast_331, %cast_332) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %304 = arith.addi %169, %c9_i32 : i32
      %305 = arith.muli %304, %c128_i32 : i32
      %306 = arith.index_cast %305 : i32 to index
      %307 = affine.apply #map8()[%205, %168, %306]
      %reinterpret_cast_333 = memref.reinterpret_cast %arg3 to offset: [%307], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_334 = memref.cast %reinterpret_cast_333 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_335 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_334, %cast_335, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %308 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_336 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_337 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_338 = memref.cast %308 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_336, %cst, %cast_337, %cast_338) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %309 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_339 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_340 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_341 = memref.cast %309 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_339, %cst_0, %cast_340, %cast_341) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_342 = memref.expand_shape %97 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %310 = hivm.hir.pointer_cast(%c33824_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %311 = hivm.hir.pointer_cast(%c34976_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_343 = memref.cast %expand_shape_342 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_344 = memref.cast %302 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_345 = memref.cast %310 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_346 = memref.cast %311 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_343, %cast_344, %cast_345, %cast_346) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %312 = hivm.hir.pointer_cast(%c35488_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %313 = hivm.hir.pointer_cast(%c35552_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_347 = memref.cast %310 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_348 = memref.cast %312 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_349 = memref.cast %313 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_347, %cast_348, %cast_349, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_350 = memref.collapse_shape %312 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %314 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_351 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_352 = memref.cast %collapse_shape_350 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_353 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_354 = memref.cast %314 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_351, %cast_352, %cast_353, %cast_354) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_355 = memref.expand_shape %97 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %315 = hivm.hir.pointer_cast(%c36576_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_356 = memref.cast %expand_shape_355 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_357 = memref.cast %315 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_356, %cast_357) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_358 = memref.subview %315[0, 0] [9, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_359 = memref.collapse_shape %subview_358 [[0, 1]] : memref<9x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<144xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_360 = memref.cast %collapse_shape_324 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_361 = memref.cast %collapse_shape_359 : memref<144xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_360, %cast_361, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_362 = memref.subview %315[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_363 = memref.subview %198[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_364 = memref.collapse_shape %subview_362 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_365 = memref.collapse_shape %subview_363 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_366 = memref.cast %collapse_shape_364 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_367 = memref.cast %collapse_shape_365 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_366, %cast_367, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %316 = hivm.hir.pointer_cast(%c37600_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_368 = memref.collapse_shape %316 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %317 = hivm.hir.pointer_cast(%c38624_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_369 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_370 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_371 = memref.cast %collapse_shape_368 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_372 = memref.cast %317 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_369, %cast_370, %cst_0, %cast_371, %cast_372) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %318 = arith.addi %169, %c10_i32 : i32
      %319 = arith.muli %318, %c128_i32 : i32
      %320 = arith.index_cast %319 : i32 to index
      %321 = affine.apply #map8()[%205, %168, %320]
      %reinterpret_cast_373 = memref.reinterpret_cast %arg3 to offset: [%321], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_374 = memref.cast %reinterpret_cast_373 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_375 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_374, %cast_375, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %322 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_376 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_377 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_378 = memref.cast %322 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_376, %cst, %cast_377, %cast_378) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %323 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_379 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_380 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_381 = memref.cast %323 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_379, %cst_0, %cast_380, %cast_381) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_382 = memref.expand_shape %91 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %324 = hivm.hir.pointer_cast(%c37600_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %325 = hivm.hir.pointer_cast(%c38752_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_383 = memref.cast %expand_shape_382 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_384 = memref.cast %316 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_385 = memref.cast %324 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_386 = memref.cast %325 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_383, %cast_384, %cast_385, %cast_386) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %326 = hivm.hir.pointer_cast(%c39264_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %327 = hivm.hir.pointer_cast(%c39328_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_387 = memref.cast %324 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_388 = memref.cast %326 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_389 = memref.cast %327 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_387, %cast_388, %cast_389, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_390 = memref.collapse_shape %326 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %328 = hivm.hir.pointer_cast(%c39264_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %329 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_391 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_392 = memref.cast %collapse_shape_390 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_393 = memref.cast %328 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_394 = memref.cast %329 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_391, %cast_392, %cast_393, %cast_394) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_395 = memref.expand_shape %328 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %330 = hivm.hir.pointer_cast(%c40352_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_396 = memref.cast %expand_shape_395 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_397 = memref.cast %330 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_396, %cast_397) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_398 = memref.subview %330[0, 0] [10, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_399 = memref.collapse_shape %subview_398 [[0, 1]] : memref<10x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<160xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_400 = memref.cast %collapse_shape_364 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_401 = memref.cast %collapse_shape_399 : memref<160xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_400, %cast_401, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_402 = memref.subview %330[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_403 = memref.subview %198[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_404 = memref.collapse_shape %subview_402 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_405 = memref.collapse_shape %subview_403 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_406 = memref.cast %collapse_shape_404 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_407 = memref.cast %collapse_shape_405 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_406, %cast_407, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %331 = hivm.hir.pointer_cast(%c41376_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_408 = memref.collapse_shape %331 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %332 = hivm.hir.pointer_cast(%c42400_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_409 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_410 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_411 = memref.cast %collapse_shape_408 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_412 = memref.cast %332 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_409, %cast_410, %cst_0, %cast_411, %cast_412) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %333 = arith.addi %169, %c11_i32 : i32
      %334 = arith.muli %333, %c128_i32 : i32
      %335 = arith.index_cast %334 : i32 to index
      %336 = affine.apply #map8()[%205, %168, %335]
      %reinterpret_cast_413 = memref.reinterpret_cast %arg3 to offset: [%336], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_414 = memref.cast %reinterpret_cast_413 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_415 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_414, %cast_415, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %337 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_416 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_417 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_418 = memref.cast %337 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_416, %cst, %cast_417, %cast_418) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %338 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_419 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_420 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_421 = memref.cast %338 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_419, %cst_0, %cast_420, %cast_421) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_422 = memref.expand_shape %85 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %339 = hivm.hir.pointer_cast(%c41376_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %340 = hivm.hir.pointer_cast(%c42528_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_423 = memref.cast %expand_shape_422 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_424 = memref.cast %331 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_425 = memref.cast %339 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_426 = memref.cast %340 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_423, %cast_424, %cast_425, %cast_426) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %341 = hivm.hir.pointer_cast(%c43040_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %342 = hivm.hir.pointer_cast(%c43104_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_427 = memref.cast %339 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_428 = memref.cast %341 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_429 = memref.cast %342 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_427, %cast_428, %cast_429, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_430 = memref.collapse_shape %341 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %343 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_431 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_432 = memref.cast %collapse_shape_430 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_433 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_434 = memref.cast %343 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_431, %cast_432, %cast_433, %cast_434) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_435 = memref.expand_shape %85 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %344 = hivm.hir.pointer_cast(%c44128_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_436 = memref.cast %expand_shape_435 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_437 = memref.cast %344 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_436, %cast_437) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_438 = memref.subview %344[0, 0] [11, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_439 = memref.collapse_shape %subview_438 [[0, 1]] : memref<11x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<176xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_440 = memref.cast %collapse_shape_404 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_441 = memref.cast %collapse_shape_439 : memref<176xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_440, %cast_441, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_442 = memref.subview %344[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_443 = memref.subview %198[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_444 = memref.collapse_shape %subview_442 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_445 = memref.collapse_shape %subview_443 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_446 = memref.cast %collapse_shape_444 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_447 = memref.cast %collapse_shape_445 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_446, %cast_447, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %345 = hivm.hir.pointer_cast(%c45152_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_448 = memref.collapse_shape %345 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %346 = hivm.hir.pointer_cast(%c46176_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_449 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_450 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_451 = memref.cast %collapse_shape_448 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_452 = memref.cast %346 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_449, %cast_450, %cst_0, %cast_451, %cast_452) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %347 = arith.addi %169, %c12_i32 : i32
      %348 = arith.muli %347, %c128_i32 : i32
      %349 = arith.index_cast %348 : i32 to index
      %350 = affine.apply #map8()[%205, %168, %349]
      %reinterpret_cast_453 = memref.reinterpret_cast %arg3 to offset: [%350], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_454 = memref.cast %reinterpret_cast_453 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_455 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_454, %cast_455, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %351 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_456 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_457 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_458 = memref.cast %351 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_456, %cst, %cast_457, %cast_458) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %352 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_459 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_460 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_461 = memref.cast %352 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_459, %cst_0, %cast_460, %cast_461) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_462 = memref.expand_shape %79 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %353 = hivm.hir.pointer_cast(%c45152_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %354 = hivm.hir.pointer_cast(%c46304_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_463 = memref.cast %expand_shape_462 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_464 = memref.cast %345 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_465 = memref.cast %353 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_466 = memref.cast %354 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_463, %cast_464, %cast_465, %cast_466) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %355 = hivm.hir.pointer_cast(%c46816_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %356 = hivm.hir.pointer_cast(%c46880_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_467 = memref.cast %353 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_468 = memref.cast %355 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_469 = memref.cast %356 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_467, %cast_468, %cast_469, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_470 = memref.collapse_shape %355 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %357 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_471 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_472 = memref.cast %collapse_shape_470 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_473 = memref.cast %79 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_474 = memref.cast %357 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_471, %cast_472, %cast_473, %cast_474) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_475 = memref.expand_shape %79 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %358 = hivm.hir.pointer_cast(%c47904_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_476 = memref.cast %expand_shape_475 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_477 = memref.cast %358 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_476, %cast_477) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_478 = memref.subview %358[0, 0] [12, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_479 = memref.collapse_shape %subview_478 [[0, 1]] : memref<12x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<192xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_480 = memref.cast %collapse_shape_444 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_481 = memref.cast %collapse_shape_479 : memref<192xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_480, %cast_481, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_482 = memref.subview %358[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_483 = memref.subview %198[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_484 = memref.collapse_shape %subview_482 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_485 = memref.collapse_shape %subview_483 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_486 = memref.cast %collapse_shape_484 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_487 = memref.cast %collapse_shape_485 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_486, %cast_487, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %359 = hivm.hir.pointer_cast(%c48928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_488 = memref.collapse_shape %359 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %360 = hivm.hir.pointer_cast(%c49952_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_489 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_490 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_491 = memref.cast %collapse_shape_488 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_492 = memref.cast %360 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_489, %cast_490, %cst_0, %cast_491, %cast_492) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %361 = arith.addi %169, %c13_i32 : i32
      %362 = arith.muli %361, %c128_i32 : i32
      %363 = arith.index_cast %362 : i32 to index
      %364 = affine.apply #map8()[%205, %168, %363]
      %reinterpret_cast_493 = memref.reinterpret_cast %arg3 to offset: [%364], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %cast_494 = memref.cast %reinterpret_cast_493 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_495 = memref.cast %73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_494, %cast_495, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %365 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_496 = memref.cast %73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_497 = memref.cast %73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_498 = memref.cast %365 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_496, %cst, %cast_497, %cast_498) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %366 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_499 = memref.cast %73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_500 = memref.cast %73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_501 = memref.cast %366 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_499, %cst_0, %cast_500, %cast_501) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_502 = memref.expand_shape %73 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %367 = hivm.hir.pointer_cast(%c48928_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %368 = hivm.hir.pointer_cast(%c50080_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_503 = memref.cast %expand_shape_502 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_504 = memref.cast %359 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_505 = memref.cast %367 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_506 = memref.cast %368 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_503, %cast_504, %cast_505, %cast_506) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %369 = hivm.hir.pointer_cast(%c50592_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %370 = hivm.hir.pointer_cast(%c50656_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_507 = memref.cast %367 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_508 = memref.cast %369 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_509 = memref.cast %370 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_507, %cast_508, %cast_509, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_510 = memref.collapse_shape %369 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %371 = hivm.hir.pointer_cast(%c50592_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %372 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_511 = memref.cast %73 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_512 = memref.cast %collapse_shape_510 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_513 = memref.cast %371 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_514 = memref.cast %372 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_511, %cast_512, %cast_513, %cast_514) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %154]
      %expand_shape_515 = memref.expand_shape %371 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %373 = hivm.hir.pointer_cast(%c51680_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_516 = memref.cast %expand_shape_515 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_517 = memref.cast %373 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_516, %cast_517) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_518 = memref.subview %373[0, 0] [13, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_519 = memref.collapse_shape %subview_518 [[0, 1]] : memref<13x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<208xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_520 = memref.cast %collapse_shape_484 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_521 = memref.cast %collapse_shape_519 : memref<208xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_520, %cast_521, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_522 = memref.subview %373[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_523 = memref.subview %198[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_524 = memref.collapse_shape %subview_522 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_525 = memref.collapse_shape %subview_523 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_526 = memref.cast %collapse_shape_524 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_527 = memref.cast %collapse_shape_525 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_526, %cast_527, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %374 = hivm.hir.pointer_cast(%c52704_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_528 = memref.collapse_shape %374 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %375 = hivm.hir.pointer_cast(%c53728_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_529 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_530 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_531 = memref.cast %collapse_shape_528 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_532 = memref.cast %375 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_529, %cast_530, %cst_0, %cast_531, %cast_532) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %376 = arith.addi %169, %c14_i32 : i32
      %377 = arith.muli %376, %c128_i32 : i32
      %378 = arith.index_cast %377 : i32 to index
      %379 = affine.apply #map8()[%205, %168, %378]
      %reinterpret_cast_533 = memref.reinterpret_cast %arg3 to offset: [%379], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %152]
      %cast_534 = memref.cast %reinterpret_cast_533 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_535 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_534, %cast_535, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %380 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_536 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_537 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_538 = memref.cast %380 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_536, %cst, %cast_537, %cast_538) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %381 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_539 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_540 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_541 = memref.cast %381 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_539, %cst_0, %cast_540, %cast_541) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_542 = memref.expand_shape %67 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %382 = hivm.hir.pointer_cast(%c52704_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %383 = hivm.hir.pointer_cast(%c53856_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_543 = memref.cast %expand_shape_542 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_544 = memref.cast %374 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_545 = memref.cast %382 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_546 = memref.cast %383 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_543, %cast_544, %cast_545, %cast_546) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %384 = hivm.hir.pointer_cast(%c54368_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %385 = hivm.hir.pointer_cast(%c54432_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_547 = memref.cast %382 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_548 = memref.cast %384 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_549 = memref.cast %385 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_547, %cast_548, %cast_549, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_550 = memref.collapse_shape %384 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %386 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_551 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_552 = memref.cast %collapse_shape_550 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_553 = memref.cast %67 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_554 = memref.cast %386 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_551, %cast_552, %cast_553, %cast_554) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_555 = memref.expand_shape %67 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      %387 = hivm.hir.pointer_cast(%c55456_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_556 = memref.cast %expand_shape_555 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_557 = memref.cast %387 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_556, %cast_557) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %152]
      %subview_558 = memref.subview %387[0, 0] [14, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_559 = memref.collapse_shape %subview_558 [[0, 1]] : memref<14x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<224xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_560 = memref.cast %collapse_shape_524 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_561 = memref.cast %collapse_shape_559 : memref<224xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_560, %cast_561, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_562 = memref.subview %387[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %subview_563 = memref.subview %198[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_564 = memref.collapse_shape %subview_562 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_565 = memref.collapse_shape %subview_563 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_566 = memref.cast %collapse_shape_564 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_567 = memref.cast %collapse_shape_565 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_566, %cast_567, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %388 = hivm.hir.pointer_cast(%c56480_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %collapse_shape_568 = memref.collapse_shape %388 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %389 = hivm.hir.pointer_cast(%c57504_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_569 = memref.cast %collapse_shape_16 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_570 = memref.cast %collapse_shape_42 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_571 = memref.cast %collapse_shape_568 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_572 = memref.cast %389 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_569, %cast_570, %cst_0, %cast_571, %cast_572) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %390 = arith.addi %169, %c15_i32 : i32
      %391 = arith.muli %390, %c128_i32 : i32
      %392 = arith.index_cast %391 : i32 to index
      %393 = affine.apply #map8()[%205, %168, %392]
      %reinterpret_cast_573 = memref.reinterpret_cast %arg3 to offset: [%393], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %150]
      %cast_574 = memref.cast %reinterpret_cast_573 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_575 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_574, %cast_575, %c0_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %394 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_576 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_577 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_578 = memref.cast %394 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_576, %cst, %cast_577, %cast_578) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %395 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_579 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_580 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_581 = memref.cast %395 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_579, %cst_0, %cast_580, %cast_581) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_582 = memref.expand_shape %61 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
      %396 = hivm.hir.pointer_cast(%c56480_i64) : memref<16x16xf32, #hivm.address_space<ub>>
      %397 = hivm.hir.pointer_cast(%c57632_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_583 = memref.cast %expand_shape_582 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_584 = memref.cast %388 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_585 = memref.cast %396 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_586 = memref.cast %397 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_583, %cast_584, %cast_585, %cast_586) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %398 = hivm.hir.pointer_cast(%c58144_i64) : memref<1x16xf32, #hivm.address_space<ub>>
      %399 = hivm.hir.pointer_cast(%c58208_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_587 = memref.cast %396 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_588 = memref.cast %398 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_589 = memref.cast %399 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_587, %cast_588, %cast_589, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_590 = memref.collapse_shape %398 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %400 = hivm.hir.pointer_cast(%c58144_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %401 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_591 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_592 = memref.cast %collapse_shape_590 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_593 = memref.cast %400 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_594 = memref.cast %401 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_591, %cast_592, %cast_593, %cast_594) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %150]
      %expand_shape_595 = memref.expand_shape %400 [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %157]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_596 = memref.cast %expand_shape_595 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_597 = memref.cast %55 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_float(%cast_596, %cast_597) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_598 = memref.subview %55[0, 0] [15, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %collapse_shape_599 = memref.collapse_shape %subview_598 [[0, 1]] : memref<15x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<240xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_600 = memref.cast %collapse_shape_564 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_601 = memref.cast %collapse_shape_599 : memref<240xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_600, %cast_601, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %collapse_shape_602 = memref.collapse_shape %55 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      %collapse_shape_603 = memref.collapse_shape %55 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %402 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_604 = memref.cast %collapse_shape_602 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_605 = memref.cast %collapse_shape_25 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_606 = memref.cast %collapse_shape_603 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_607 = memref.cast %402 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_604, %cast_605, %cast_606, %cast_607) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_608 = memref.reinterpret_cast %arg4 to offset: [%175], sizes: [16, 16], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %403 = affine.max #map9()[%47, %172]
      %404 = affine.min #map4()[%403]
      %405 = affine.min #map6()[%183, %403]
      %406 = affine.apply #map7()[%404, %405]
      %subview_609 = memref.subview %55[%405, 0] [%406, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_610 = memref.subview %reinterpret_cast_608[0, 0] [%406, 16] [1, 1] : memref<16x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_611 = memref.cast %subview_609 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_612 = memref.cast %subview_610 : memref<?x16xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_611, %cast_612, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %157]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
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
