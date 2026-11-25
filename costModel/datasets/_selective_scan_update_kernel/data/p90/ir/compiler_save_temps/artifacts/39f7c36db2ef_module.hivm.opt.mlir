#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<()[s0] -> (16, s0)>
#map2 = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
#map4 = affine_map<()[s0, s1] -> (s1, s0)>
#map5 = affine_map<()[s0, s1] -> (s1 + 32, s0)>
#map6 = affine_map<()[s0, s1] -> (s0 - s1)>
#map7 = affine_map<()[s0, s1] -> (32, s0 - s1)>
#map8 = affine_map<()[s0] -> (32, s0)>
#map9 = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 + s2 * s1 + ((s4 - s5) floordiv 48) * (s1 * s3)) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_selective_scan_update_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int32_t_to_int64_t_1d_with_mode(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_int64_t(memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_int64_t(memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_le_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_selective_scan_update_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32, %arg39: i32, %arg40: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<41xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c2176_i64 = arith.constant 2176 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c2368_i64 = arith.constant 2368 : i64
    %c2496_i64 = arith.constant 2496 : i64
    %c2624_i64 = arith.constant 2624 : i64
    %c2880_i64 = arith.constant 2880 : i64
    %c6976_i64 = arith.constant 6976 : i64
    %c7104_i64 = arith.constant 7104 : i64
    %c11200_i64 = arith.constant 11200 : i64
    %c11328_i64 = arith.constant 11328 : i64
    %c13376_i64 = arith.constant 13376 : i64
    %c14400_i64 = arith.constant 14400 : i64
    %c18496_i64 = arith.constant 18496 : i64
    %c18752_i64 = arith.constant 18752 : i64
    %c22848_i64 = arith.constant 22848 : i64
    %c22976_i64 = arith.constant 22976 : i64
    %c27072_i64 = arith.constant 27072 : i64
    %c27200_i64 = arith.constant 27200 : i64
    %c29248_i64 = arith.constant 29248 : i64
    %c30272_i64 = arith.constant 30272 : i64
    %c34368_i64 = arith.constant 34368 : i64
    %c36416_i64 = arith.constant 36416 : i64
    %c38848_i64 = arith.constant 38848 : i64
    %c38880_i64 = arith.constant 38880 : i64
    %c39008_i64 = arith.constant 39008 : i64
    %c39136_i64 = arith.constant 39136 : i64
    %c39168_i64 = arith.constant 39168 : i64
    %c41216_i64 = arith.constant 41216 : i64
    %c43264_i64 = arith.constant 43264 : i64
    %c44672_i64 = arith.constant 44672 : i64
    %c46720_i64 = arith.constant 46720 : i64
    %c47744_i64 = arith.constant 47744 : i64
    %c49792_i64 = arith.constant 49792 : i64
    %c50816_i64 = arith.constant 50816 : i64
    %c50848_i64 = arith.constant 50848 : i64
    %c51040_i64 = arith.constant 51040 : i64
    %c512 = arith.constant 512 : index
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c51936_i64 = arith.constant 51936 : i64
    %c50880_i64 = arith.constant 50880 : i64
    %c51168_i64 = arith.constant 51168 : i64
    %c38464_i64 = arith.constant 38464 : i64
    %c51296_i64 = arith.constant 51296 : i64
    %c38592_i64 = arith.constant 38592 : i64
    %c51424_i64 = arith.constant 51424 : i64
    %c38720_i64 = arith.constant 38720 : i64
    %c51552_i64 = arith.constant 51552 : i64
    %c44288_i64 = arith.constant 44288 : i64
    %c51616_i64 = arith.constant 51616 : i64
    %c44352_i64 = arith.constant 44352 : i64
    %c51680_i64 = arith.constant 51680 : i64
    %c44416_i64 = arith.constant 44416 : i64
    %c51808_i64 = arith.constant 51808 : i64
    %c44544_i64 = arith.constant 44544 : i64
    %c50912_i64 = arith.constant 50912 : i64
    %c51968_i64 = arith.constant 51968 : i64
    %0 = hivm.hir.pointer_cast(%c50912_i64) : memref<32xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c51968_i64) : memref<32xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c44544_i64) : memref<32xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c51808_i64) : memref<32xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c44416_i64) : memref<32xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c51680_i64) : memref<32xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c44352_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c51616_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c44288_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c51552_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c38720_i64) : memref<32xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c51424_i64) : memref<32xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c38592_i64) : memref<32xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c51296_i64) : memref<32xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c38464_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c51168_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c50880_i64) : memref<1xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c51936_i64) : memref<1xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c50912_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c51968_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c38464_i64) : memref<32xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c51168_i64) : memref<32xf32, #hivm.address_space<ub>>
    %22 = arith.muli %arg38, %arg39 : i32
    %23 = arith.muli %22, %arg40 : i32
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.trunci %24 : i64 to i32
    %26 = arith.muli %arg40, %arg39 : i32
    %27 = hivm.hir.pointer_cast(%c0_i64) : memref<32x16xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %27 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %cast = memref.cast %collapse_shape : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_2, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %28 = hivm.hir.pointer_cast(%c2048_i64) : memref<32xf32, #hivm.address_space<ub>>
    %cast_3 = memref.cast %28 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast_3) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %29 = hivm.hir.pointer_cast(%c2176_i64) : memref<32xi32, #hivm.address_space<ub>>
    %cast_4 = memref.cast %29 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_4, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %30 = hivm.hir.pointer_cast(%c2304_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast_5 = memref.cast %30 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_5, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %31 = arith.cmpi eq, %arg21, %c1_i32 : i32
    %32 = arith.cmpi eq, %arg28, %c1_i32 : i32
    %33 = arith.index_cast %arg16 : i32 to index
    %34 = arith.index_cast %arg17 : i32 to index
    %35 = affine.max #map()[%34]
    %36 = affine.min #map1()[%35]
    %37 = affine.max #map()[%36]
    %38 = affine.min #map1()[%37]
    %39 = arith.cmpi slt, %36, %c16 : index
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg41 = %25 to %23 step %c48_i32  : i32 {
      %40 = arith.index_cast %arg41 : i32 to index
      %41 = arith.index_cast %25 : i32 to index
      %42 = affine.apply #map2()[%40, %41]
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.cmpi eq, %43, %c1_i64 : i64
      %45 = arith.select %44, %1, %0 : memref<32xf32, #hivm.address_space<ub>>
      %46 = arith.index_cast %arg41 : i32 to index
      %47 = arith.index_cast %25 : i32 to index
      %48 = affine.apply #map2()[%46, %47]
      %49 = arith.index_cast %48 : index to i64
      %50 = arith.cmpi eq, %49, %c1_i64 : i64
      %51 = arith.select %50, %3, %2 : memref<32xf32, #hivm.address_space<ub>>
      %52 = arith.index_cast %arg41 : i32 to index
      %53 = arith.index_cast %25 : i32 to index
      %54 = affine.apply #map2()[%52, %53]
      %55 = arith.index_cast %54 : index to i64
      %56 = arith.cmpi eq, %55, %c1_i64 : i64
      %57 = arith.select %56, %5, %4 : memref<32xf32, #hivm.address_space<ub>>
      %58 = arith.index_cast %arg41 : i32 to index
      %59 = arith.index_cast %25 : i32 to index
      %60 = affine.apply #map2()[%58, %59]
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.cmpi eq, %61, %c1_i64 : i64
      %63 = arith.select %62, %7, %6 : memref<1x16xf32, #hivm.address_space<ub>>
      %64 = arith.index_cast %arg41 : i32 to index
      %65 = arith.index_cast %25 : i32 to index
      %66 = affine.apply #map2()[%64, %65]
      %67 = arith.index_cast %66 : index to i64
      %68 = arith.cmpi eq, %67, %c1_i64 : i64
      %69 = arith.select %68, %9, %8 : memref<1x16xf32, #hivm.address_space<ub>>
      %70 = arith.index_cast %arg41 : i32 to index
      %71 = arith.index_cast %25 : i32 to index
      %72 = affine.apply #map2()[%70, %71]
      %73 = arith.index_cast %72 : index to i64
      %74 = arith.cmpi eq, %73, %c1_i64 : i64
      %75 = arith.select %74, %11, %10 : memref<32xf32, #hivm.address_space<ub>>
      %76 = arith.index_cast %arg41 : i32 to index
      %77 = arith.index_cast %25 : i32 to index
      %78 = affine.apply #map2()[%76, %77]
      %79 = arith.index_cast %78 : index to i64
      %80 = arith.cmpi eq, %79, %c1_i64 : i64
      %81 = arith.select %80, %13, %12 : memref<32xf32, #hivm.address_space<ub>>
      %82 = arith.index_cast %arg41 : i32 to index
      %83 = arith.index_cast %25 : i32 to index
      %84 = affine.apply #map2()[%82, %83]
      %85 = arith.index_cast %84 : index to i64
      %86 = arith.cmpi eq, %85, %c1_i64 : i64
      %87 = arith.select %86, %15, %14 : memref<32x1xf32, #hivm.address_space<ub>>
      %88 = arith.index_cast %arg41 : i32 to index
      %89 = arith.index_cast %25 : i32 to index
      %90 = affine.apply #map2()[%88, %89]
      %91 = arith.index_cast %90 : index to i64
      %92 = arith.cmpi eq, %91, %c1_i64 : i64
      %93 = arith.select %92, %19, %18 : memref<32x1xf32, #hivm.address_space<ub>>
      %94 = arith.index_cast %arg41 : i32 to index
      %95 = arith.index_cast %25 : i32 to index
      %96 = affine.apply #map2()[%94, %95]
      %97 = arith.index_cast %96 : index to i64
      %98 = arith.cmpi eq, %97, %c1_i64 : i64
      %99 = arith.select %98, %21, %20 : memref<32xf32, #hivm.address_space<ub>>
      %100 = arith.index_cast %arg41 : i32 to index
      %101 = arith.index_cast %25 : i32 to index
      %102 = affine.apply #map2()[%100, %101]
      %103 = arith.index_cast %102 : index to i1
      %104 = arith.select %103, %c3_i64, %c4_i64 : i64
      %105 = arith.index_cast %102 : index to i1
      %106 = arith.select %105, %c1_i64, %c2_i64 : i64
      %107 = arith.index_cast %102 : index to i1
      %108 = arith.xori %107, %true : i1
      %109 = arith.extui %108 : i1 to i64
      hivm.hir.set_mask_norm
      %110 = arith.remsi %arg41, %arg40 : i32
      %111 = arith.divsi %arg41, %arg40 : i32
      %112 = arith.remsi %111, %arg39 : i32
      %113 = arith.divsi %arg41, %26 : i32
      %114 = arith.remsi %113, %arg38 : i32
      %115 = arith.muli %112, %arg19 : i32
      %116 = arith.muli %110, %arg20 : i32
      %117 = arith.addi %115, %116 : i32
      %118 = arith.extsi %117 : i32 to i64
      %119 = arith.muli %112, %arg22 : i32
      %120 = arith.muli %110, %arg23 : i32
      %121 = arith.addi %119, %120 : i32
      %122 = arith.muli %112, %arg24 : i32
      %123 = arith.muli %110, %arg25 : i32
      %124 = arith.addi %122, %123 : i32
      %125 = arith.muli %110, %arg26 : i32
      %126 = arith.muli %110, %arg27 : i32
      %127 = arith.extsi %126 : i32 to i64
      %128 = arith.muli %112, %arg29 : i32
      %129 = arith.divsi %110, %arg18 : i32
      %130 = arith.muli %129, %arg30 : i32
      %131 = arith.addi %128, %130 : i32
      %132 = arith.muli %112, %arg31 : i32
      %133 = arith.muli %129, %arg32 : i32
      %134 = arith.addi %132, %133 : i32
      %135 = arith.muli %112, %arg34 : i32
      %136 = arith.muli %110, %arg35 : i32
      %137 = arith.addi %135, %136 : i32
      %138 = arith.muli %112, %arg36 : i32
      %139 = arith.muli %110, %arg37 : i32
      %140 = arith.addi %138, %139 : i32
      %141 = arith.muli %114, %c32_i32 : i32
      %142 = hivm.hir.pointer_cast(%c2368_i64) : memref<32xi32, #hivm.address_space<ub>>
      %143 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_6 = memref.cast %29 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %142 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %143 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_6, %141, %cast_7, %cast_8) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %144 = scf.if %31 -> (memref<32x16xi64, #hivm.address_space<ub>>) {
        %210 = hivm.hir.pointer_cast(%c2496_i64) : memref<32x1xi32, #hivm.address_space<ub>>
        %collapse_shape_154 = memref.collapse_shape %210 [[0, 1]] : memref<32x1xi32, #hivm.address_space<ub>> into memref<32xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %211 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %142 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_156 = memref.cast %collapse_shape_154 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_157 = memref.cast %211 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_int32_t(%cast_155, %arg21, %cast_156, %cast_157) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %212 = hivm.hir.pointer_cast(%c2624_i64) : memref<32x1xi64, #hivm.address_space<ub>>
        %collapse_shape_158 = memref.collapse_shape %212 [[0, 1]] : memref<32x1xi64, #hivm.address_space<ub>> into memref<32xi64, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_159 = memref.cast %collapse_shape_154 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_160 = memref.cast %collapse_shape_158 : memref<32xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_159, %cast_160, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %213 = hivm.hir.pointer_cast(%c2624_i64) : memref<32x1xi64, #hivm.address_space<ub>>
        %collapse_shape_161 = memref.collapse_shape %213 [[0, 1]] : memref<32x1xi64, #hivm.address_space<ub>> into memref<32xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg42 = %c0 to %c32 step %c1 {
          %219 = memref.load %collapse_shape_158[%arg42] : memref<32xi64, #hivm.address_space<ub>>
          %220 = arith.addi %219, %118 : i64
          memref.store %220, %collapse_shape_161[%arg42] : memref<32xi64, #hivm.address_space<ub>>
        }
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %214 = hivm.hir.pointer_cast(%c2880_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %215 = hivm.hir.pointer_cast(%c6976_i64) : memref<0xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_162 = memref.cast %213 : memref<32x1xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_163 = memref.cast %214 : memref<32x16xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_164 = memref.cast %215 : memref<0xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_int64_t(%cast_162, %cast_163, %cast_164) : (memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %216 = hivm.hir.pointer_cast(%c6976_i64) : memref<16xi64, #hivm.address_space<ub>>
        %cast_165 = memref.cast %30 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_166 = memref.cast %216 : memref<16xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_165, %cast_166, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %expand_shape_167 = memref.expand_shape %216 [[0, 1]] output_shape [1, 16] : memref<16xi64, #hivm.address_space<ub>> into memref<1x16xi64, #hivm.address_space<ub>>
        %217 = hivm.hir.pointer_cast(%c7104_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_168 = memref.cast %expand_shape_167 : memref<1x16xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_169 = memref.cast %217 : memref<32x16xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_int64_t(%cast_168, %cast_169) : (memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %218 = hivm.hir.pointer_cast(%c7104_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %collapse_shape_170 = memref.collapse_shape %214 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        %collapse_shape_171 = memref.collapse_shape %217 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        %collapse_shape_172 = memref.collapse_shape %218 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg42 = %c0 to %c512 step %c1 {
          %219 = memref.load %collapse_shape_170[%arg42] : memref<512xi64, #hivm.address_space<ub>>
          %220 = memref.load %collapse_shape_171[%arg42] : memref<512xi64, #hivm.address_space<ub>>
          %221 = arith.addi %219, %220 : i64
          memref.store %221, %collapse_shape_172[%arg42] : memref<512xi64, #hivm.address_space<ub>>
        }
        scf.yield %218 : memref<32x16xi64, #hivm.address_space<ub>>
      } else {
        %210 = hivm.hir.pointer_cast(%c11200_i64) : memref<32x1xi32, #hivm.address_space<ub>>
        %collapse_shape_154 = memref.collapse_shape %210 [[0, 1]] : memref<32x1xi32, #hivm.address_space<ub>> into memref<32xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %211 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %142 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_156 = memref.cast %collapse_shape_154 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_157 = memref.cast %211 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_int32_t(%cast_155, %arg21, %cast_156, %cast_157) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_158 = memref.expand_shape %30 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
        %212 = hivm.hir.pointer_cast(%c11328_i64) : memref<32x16xi32, #hivm.address_space<ub>>
        %213 = hivm.hir.pointer_cast(%c13376_i64) : memref<256xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_159 = memref.cast %210 : memref<32x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_160 = memref.cast %expand_shape_158 : memref<1x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_161 = memref.cast %212 : memref<32x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_162 = memref.cast %213 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_2d_int32_t(%cast_159, %cast_160, %cast_161, %cast_162) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %214 = hivm.hir.pointer_cast(%c14400_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %collapse_shape_163 = memref.collapse_shape %212 [[0, 1]] : memref<32x16xi32, #hivm.address_space<ub>> into memref<512xi32, #hivm.address_space<ub>>
        %collapse_shape_164 = memref.collapse_shape %214 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_165 = memref.cast %collapse_shape_163 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_166 = memref.cast %collapse_shape_164 : memref<512xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_165, %cast_166, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %215 = hivm.hir.pointer_cast(%c14400_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %collapse_shape_167 = memref.collapse_shape %215 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg42 = %c0 to %c512 step %c1 {
          %216 = memref.load %collapse_shape_164[%arg42] : memref<512xi64, #hivm.address_space<ub>>
          %217 = arith.addi %216, %118 : i64
          memref.store %217, %collapse_shape_167[%arg42] : memref<512xi64, #hivm.address_space<ub>>
        }
        scf.yield %215 : memref<32x16xi64, #hivm.address_space<ub>>
      }
      %145 = arith.index_cast %121 : i32 to index
      %146 = arith.index_cast %141 : i32 to index
      %147 = affine.apply #map3()[%145, %146]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%147], sizes: [32, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %148 = arith.index_cast %124 : i32 to index
      %149 = affine.apply #map3()[%148, %146]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%149], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %150 = arith.index_cast %125 : i32 to index
      %151 = affine.apply #map3()[%150, %146]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%151], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %152 = arith.muli %110, %arg33 : i32
      %153 = scf.if %32 -> (memref<32x16xi64, #hivm.address_space<ub>>) {
        %210 = hivm.hir.pointer_cast(%c18496_i64) : memref<32x1xi64, #hivm.address_space<ub>>
        %collapse_shape_154 = memref.collapse_shape %210 [[0, 1]] : memref<32x1xi64, #hivm.address_space<ub>> into memref<32xi64, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_155 = memref.cast %142 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_156 = memref.cast %collapse_shape_154 : memref<32xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_155, %cast_156, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %211 = hivm.hir.pointer_cast(%c18496_i64) : memref<32x1xi64, #hivm.address_space<ub>>
        %collapse_shape_157 = memref.collapse_shape %211 [[0, 1]] : memref<32x1xi64, #hivm.address_space<ub>> into memref<32xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg42 = %c0 to %c32 step %c1 {
          %217 = memref.load %collapse_shape_154[%arg42] : memref<32xi64, #hivm.address_space<ub>>
          %218 = arith.addi %217, %127 : i64
          memref.store %218, %collapse_shape_157[%arg42] : memref<32xi64, #hivm.address_space<ub>>
        }
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %212 = hivm.hir.pointer_cast(%c18752_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %213 = hivm.hir.pointer_cast(%c22848_i64) : memref<0xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_158 = memref.cast %211 : memref<32x1xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_159 = memref.cast %212 : memref<32x16xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_160 = memref.cast %213 : memref<0xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_int64_t(%cast_158, %cast_159, %cast_160) : (memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %214 = hivm.hir.pointer_cast(%c22848_i64) : memref<16xi64, #hivm.address_space<ub>>
        %cast_161 = memref.cast %30 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_162 = memref.cast %214 : memref<16xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_161, %cast_162, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %expand_shape_163 = memref.expand_shape %214 [[0, 1]] output_shape [1, 16] : memref<16xi64, #hivm.address_space<ub>> into memref<1x16xi64, #hivm.address_space<ub>>
        %215 = hivm.hir.pointer_cast(%c22976_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_164 = memref.cast %expand_shape_163 : memref<1x16xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_165 = memref.cast %215 : memref<32x16xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_int64_t(%cast_164, %cast_165) : (memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %216 = hivm.hir.pointer_cast(%c22976_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %collapse_shape_166 = memref.collapse_shape %212 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        %collapse_shape_167 = memref.collapse_shape %215 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        %collapse_shape_168 = memref.collapse_shape %216 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg42 = %c0 to %c512 step %c1 {
          %217 = memref.load %collapse_shape_166[%arg42] : memref<512xi64, #hivm.address_space<ub>>
          %218 = memref.load %collapse_shape_167[%arg42] : memref<512xi64, #hivm.address_space<ub>>
          %219 = arith.addi %217, %218 : i64
          memref.store %219, %collapse_shape_168[%arg42] : memref<512xi64, #hivm.address_space<ub>>
        }
        scf.yield %216 : memref<32x16xi64, #hivm.address_space<ub>>
      } else {
        %210 = hivm.hir.pointer_cast(%c27072_i64) : memref<32x1xi32, #hivm.address_space<ub>>
        %collapse_shape_154 = memref.collapse_shape %210 [[0, 1]] : memref<32x1xi32, #hivm.address_space<ub>> into memref<32xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %211 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %142 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_156 = memref.cast %collapse_shape_154 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_157 = memref.cast %211 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_int32_t(%cast_155, %arg28, %cast_156, %cast_157) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_158 = memref.expand_shape %30 [[0, 1]] output_shape [1, 16] : memref<16xi32, #hivm.address_space<ub>> into memref<1x16xi32, #hivm.address_space<ub>>
        %212 = hivm.hir.pointer_cast(%c27200_i64) : memref<32x16xi32, #hivm.address_space<ub>>
        %213 = hivm.hir.pointer_cast(%c29248_i64) : memref<256xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_159 = memref.cast %210 : memref<32x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_160 = memref.cast %expand_shape_158 : memref<1x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_161 = memref.cast %212 : memref<32x16xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_162 = memref.cast %213 : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_2d_int32_t(%cast_159, %cast_160, %cast_161, %cast_162) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %214 = hivm.hir.pointer_cast(%c30272_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %collapse_shape_163 = memref.collapse_shape %212 [[0, 1]] : memref<32x16xi32, #hivm.address_space<ub>> into memref<512xi32, #hivm.address_space<ub>>
        %collapse_shape_164 = memref.collapse_shape %214 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_165 = memref.cast %collapse_shape_163 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_166 = memref.cast %collapse_shape_164 : memref<512xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_int32_t_to_int64_t_1d_with_mode(%cast_165, %cast_166, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %215 = hivm.hir.pointer_cast(%c30272_i64) : memref<32x16xi64, #hivm.address_space<ub>>
        %collapse_shape_167 = memref.collapse_shape %215 [[0, 1]] : memref<32x16xi64, #hivm.address_space<ub>> into memref<512xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg42 = %c0 to %c512 step %c1 {
          %216 = memref.load %collapse_shape_164[%arg42] : memref<512xi64, #hivm.address_space<ub>>
          %217 = arith.addi %216, %127 : i64
          memref.store %217, %collapse_shape_167[%arg42] : memref<512xi64, #hivm.address_space<ub>>
        }
        scf.yield %215 : memref<32x16xi64, #hivm.address_space<ub>>
      }
      %154 = arith.index_cast %131 : i32 to index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg8 to offset: [%154], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %155 = arith.index_cast %134 : i32 to index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg9 to offset: [%155], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %156 = arith.index_cast %152 : i32 to index
      %157 = affine.apply #map3()[%156, %146]
      %reinterpret_cast_13 = memref.reinterpret_cast %arg10 to offset: [%157], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %158 = arith.index_cast %137 : i32 to index
      %159 = affine.apply #map3()[%158, %146]
      %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [%159], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %160 = arith.index_cast %140 : i32 to index
      %161 = affine.apply #map3()[%160, %146]
      %reinterpret_cast_15 = memref.reinterpret_cast %arg12 to offset: [%161], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %162 = affine.max #map4()[%146, %33]
      %163 = affine.min #map5()[%162, %146]
      %164 = affine.apply #map6()[%163, %146]
      %165 = affine.min #map7()[%163, %146]
      %166 = affine.max #map()[%165]
      %167 = affine.min #map8()[%166]
      %168 = hivm.hir.pointer_cast(%c34368_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg42 = %c0 to %167 step %c1 {
        scf.for %arg43 = %c0 to %38 step %c1 {
          %210 = memref.load %144[%arg42, %arg43] {DiscreteMemAccess} : memref<32x16xi64, #hivm.address_space<ub>>
          %211 = arith.index_cast %210 : i64 to index
          %reinterpret_cast_154 = memref.reinterpret_cast %arg3 to offset: [%211], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %212 = memref.load %reinterpret_cast_154[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %212, %168[%arg42, %arg43] : memref<32x16xf32, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %subview = memref.subview %168[0, 0] [%166, %37] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %169 = hivm.hir.pointer_cast(%c36416_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %collapse_shape_16 = memref.collapse_shape %169 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %cast_17 = memref.cast %collapse_shape : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %collapse_shape_16 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_17, %cast_18, %c0_i32, %cst_2) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_19 = memref.subview %169[0, 0] [%166, %37] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_20 = memref.cast %subview : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %subview_19 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_2d_float(%cast_20, %cast_21, %c0_i32, %cst_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %170 = arith.cmpi slt, %164, %c32 : index
      %subview_22 = memref.subview %reinterpret_cast[0, 0] [%164, 1] [1, 1] : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_23 = memref.subview %87[0, 0] [%164, 1] [1, 1] : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %170 {
        %collapse_shape_154 = memref.collapse_shape %87 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %collapse_shape_154 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_155) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_24 = memref.collapse_shape %subview_22 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_25 = memref.collapse_shape %subview_23 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %106]
      %cast_26 = memref.cast %collapse_shape_24 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_27 = memref.cast %collapse_shape_25 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_26, %cast_27, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %collapse_shape_28 = memref.collapse_shape %87 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_29 = memref.subview %reinterpret_cast_9[0] [%164] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_30 = memref.subview %81[0] [%164] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %170 {
        %cast_154 = memref.cast %81 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_31 = memref.cast %subview_29 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_32 = memref.cast %subview_30 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_31, %cast_32, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %subview_33 = memref.subview %reinterpret_cast_10[0] [%164] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_34 = memref.subview %75[0] [%164] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %170 {
        %cast_154 = memref.cast %75 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_35 = memref.cast %subview_33 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_36 = memref.cast %subview_34 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_35, %cast_36, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %171 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_37 = memref.cast %81 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %75 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_39 = memref.cast %75 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %171 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_37, %cast_38, %cast_39, %cast_40) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %172 = hivm.hir.pointer_cast(%c38848_i64) : memref<32xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_41 = memref.cast %75 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %28 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %172 : memref<32xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_le_1d_float(%cast_41, %cast_42, %cast_43) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %173 = hivm.hir.pointer_cast(%c38880_i64) : memref<32xf32, #hivm.address_space<ub>>
      %174 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_44 = memref.cast %75 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %173 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %174 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_44, %cast_45, %cast_46) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %175 = hivm.hir.pointer_cast(%c38880_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %176 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_47 = memref.cast %173 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %175 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %176 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_47, %cst_1, %cast_48, %cast_49) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %177 = hivm.hir.pointer_cast(%c38880_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %178 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_50 = memref.cast %175 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_51 = memref.cast %177 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_52 = memref.cast %178 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vln_1d_float(%cast_50, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %179 = hivm.hir.pointer_cast(%c39008_i64) : memref<32xf32, #hivm.address_space<ub>>
      %180 = hivm.hir.pointer_cast(%c39136_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_53 = memref.cast %172 : memref<32xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_54 = memref.cast %177 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %75 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %179 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_57 = memref.cast %180 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_53, %cast_54, %cast_55, %cast_56, %cast_57) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %181 = hivm.hir.pointer_cast(%c39168_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      scf.for %arg42 = %c0 to %167 step %c1 {
        scf.for %arg43 = %c0 to %38 step %c1 {
          %210 = memref.load %153[%arg42, %arg43] {DiscreteMemAccess} : memref<32x16xi64, #hivm.address_space<ub>>
          %211 = arith.index_cast %210 : i64 to index
          %reinterpret_cast_154 = memref.reinterpret_cast %arg7 to offset: [%211], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %212 = memref.load %reinterpret_cast_154[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %212, %181[%arg42, %arg43] : memref<32x16xf32, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %subview_58 = memref.subview %181[0, 0] [%166, %37] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %182 = hivm.hir.pointer_cast(%c41216_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %collapse_shape_59 = memref.collapse_shape %182 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %cast_60 = memref.cast %collapse_shape : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %collapse_shape_59 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_60, %cast_61, %c0_i32, %cst_2) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %subview_62 = memref.subview %182[0, 0] [%166, %37] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_63 = memref.cast %subview_58 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_64 = memref.cast %subview_62 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_2d_float(%cast_63, %cast_64, %c0_i32, %cst_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %expand_shape = memref.expand_shape %179 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      %183 = hivm.hir.pointer_cast(%c41216_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %184 = hivm.hir.pointer_cast(%c43264_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_65 = memref.cast %182 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_66 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_67 = memref.cast %183 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_68 = memref.cast %184 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_65, %cast_66, %cast_67, %cast_68) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %185 = hivm.hir.pointer_cast(%c41216_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %collapse_shape_69 = memref.collapse_shape %183 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_70 = memref.collapse_shape %185 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %186 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_71 = memref.cast %collapse_shape_69 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_72 = memref.cast %collapse_shape_70 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_73 = memref.cast %186 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_71, %cast_72, %cast_73) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %subview_74 = memref.subview %reinterpret_cast_11[0, 0] [1, %36] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_75 = memref.subview %69[0, 0] [1, %36] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %39 {
        %collapse_shape_154 = memref.collapse_shape %69 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %collapse_shape_154 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_155) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_76 = memref.collapse_shape %subview_74 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_77 = memref.collapse_shape %subview_75 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_78 = memref.cast %collapse_shape_76 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_79 = memref.cast %collapse_shape_77 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_78, %cast_79, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %collapse_shape_80 = memref.collapse_shape %69 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %subview_81 = memref.subview %reinterpret_cast_12[0, 0] [1, %36] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_82 = memref.subview %63[0, 0] [1, %36] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %39 {
        %collapse_shape_154 = memref.collapse_shape %63 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %collapse_shape_154 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_155) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_83 = memref.collapse_shape %subview_81 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_84 = memref.collapse_shape %subview_82 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %collapse_shape_83 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_86 = memref.cast %collapse_shape_84 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_85, %cast_86, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
      %subview_87 = memref.subview %reinterpret_cast_13[0] [%164] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_88 = memref.subview %57[0] [%164] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %170 {
        %cast_154 = memref.cast %57 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_89 = memref.cast %subview_87 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_90 = memref.cast %subview_88 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_89, %cast_90, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %subview_91 = memref.subview %reinterpret_cast_14[0] [%164] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_92 = memref.subview %51[0] [%164] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %170 {
        %cast_154 = memref.cast %51 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %104]
      %cast_93 = memref.cast %subview_91 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_94 = memref.cast %subview_92 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_93, %cast_94, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %187 = hivm.hir.pointer_cast(%c44672_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %188 = hivm.hir.pointer_cast(%c46720_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
      %cast_95 = memref.cast %69 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_96 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_97 = memref.cast %187 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_98 = memref.cast %188 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_95, %cast_96, %cast_97, %cast_98) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %189 = hivm.hir.pointer_cast(%c47744_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %collapse_shape_99 = memref.collapse_shape %189 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %190 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_100 = memref.cast %collapse_shape_16 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_101 = memref.cast %collapse_shape_70 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_102 = memref.cast %collapse_shape_99 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_103 = memref.cast %190 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_100, %cast_101, %cast_102, %cast_103) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %191 = hivm.hir.pointer_cast(%c44672_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %192 = hivm.hir.pointer_cast(%c49792_i64) : memref<256xf32, #hivm.address_space<ub>>
      %cast_104 = memref.cast %187 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_105 = memref.cast %87 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_106 = memref.cast %191 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_107 = memref.cast %192 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_104, %cast_105, %cast_106, %cast_107) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %193 = hivm.hir.pointer_cast(%c47744_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      %collapse_shape_108 = memref.collapse_shape %191 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      %collapse_shape_109 = memref.collapse_shape %193 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_110 = memref.cast %collapse_shape_99 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_111 = memref.cast %collapse_shape_108 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_112 = memref.cast %collapse_shape_109 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_113 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_110, %cast_111, %cast_112, %cast_113) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg42 = %c0 to %167 step %c1 {
        %210 = memref.load %179[%arg42] {DiscreteMemAccess} : memref<32xf32, #hivm.address_space<ub>>
        %211 = hivm.hir.pointer_cast(%c50816_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %210, %211[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %212 = memref.load %collapse_shape_28[%arg42] {DiscreteMemAccess} : memref<32xf32, #hivm.address_space<ub>>
        %213 = hivm.hir.pointer_cast(%c50848_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %212, %213[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %214 = arith.index_cast %arg42 : index to i32
        %215 = arith.addi %141, %214 : i32
        %216 = arith.cmpi slt, %215, %arg16 : i32
        scf.for %arg43 = %c0 to %38 step %c1 {
          %217 = arith.index_cast %arg41 : i32 to index
          %218 = arith.index_cast %25 : i32 to index
          %219 = affine.apply #map9()[%arg43, %38, %arg42, %167, %217, %218]
          %220 = arith.index_cast %219 : index to i64
          %221 = arith.cmpi eq, %220, %c1_i64 : i64
          %222 = arith.select %221, %17, %16 : memref<1xf32, #hivm.address_space<ub>>
          %223 = arith.index_cast %arg41 : i32 to index
          %224 = arith.index_cast %25 : i32 to index
          %225 = affine.apply #map9()[%arg43, %38, %arg42, %167, %223, %224]
          %226 = arith.index_cast %225 : index to i1
          %227 = arith.xori %226, %true : i1
          %228 = arith.extui %227 : i1 to i64
          %229 = memref.load %144[%arg42, %arg43] {DiscreteMemAccess} : memref<32x16xi64, #hivm.address_space<ub>>
          %230 = memref.load %169[%arg42, %arg43] {DiscreteMemAccess} : memref<32x16xf32, #hivm.address_space<ub>>
          %231 = memref.load %185[%arg42, %arg43] {DiscreteMemAccess} : memref<32x16xf32, #hivm.address_space<ub>>
          %232 = arith.mulf %230, %231 : f32
          %233 = memref.load %collapse_shape_80[%arg43] {DiscreteMemAccess} : memref<16xf32, #hivm.address_space<ub>>
          %234 = memref.load %211[%c0] : memref<1xf32, #hivm.address_space<ub>>
          %235 = arith.mulf %233, %234 : f32
          %236 = memref.load %213[%c0] : memref<1xf32, #hivm.address_space<ub>>
          %237 = arith.mulf %235, %236 : f32
          %238 = arith.addf %232, %237 : f32
          %239 = arith.index_cast %arg43 : index to i32
          %240 = arith.cmpi slt, %239, %arg17 : i32
          %241 = arith.andi %216, %240 : i1
          scf.if %241 {
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %228]
            memref.store %238, %222[%c0] : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %242 = arith.index_cast %229 : i64 to index
            %reinterpret_cast_154 = memref.reinterpret_cast %arg3 to offset: [%242], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            %cast_155 = memref.cast %222 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_156 = memref.cast %reinterpret_cast_154 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_155, %cast_156, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %228]
          } {pipeline.veconly}
        } {ExtractedLoadOrStore, pipeline.veconly}
      } {ExtractedLoadOrStore, pipeline.veconly}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      %195 = hivm.hir.pointer_cast(%c47744_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %196 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_114 = memref.cast %193 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_115 = memref.cast %63 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_116 = memref.cast %195 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_117 = memref.cast %196 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_114, %cast_115, %cast_116, %cast_117) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %109]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %197 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_118 = memref.cast %195 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_119 = memref.cast %93 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_120 = memref.cast %197 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_sum_ar_float(%cast_118, %cast_119, %cast_120, %cst_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_121 = memref.collapse_shape %93 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %198 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_122 = memref.cast %collapse_shape_28 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_123 = memref.cast %57 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_124 = memref.cast %99 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_125 = memref.cast %198 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_122, %cast_123, %cast_124, %cast_125) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %199 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_126 = memref.cast %collapse_shape_121 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_127 = memref.cast %99 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_128 = memref.cast %45 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_129 = memref.cast %199 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_126, %cast_127, %cast_128, %cast_129) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %106]
      %200 = hivm.hir.pointer_cast(%c51040_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %201 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_130 = memref.cast %51 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_131 = memref.cast %200 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_132 = memref.cast %201 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_130, %cst, %cast_131, %cast_132) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %202 = hivm.hir.pointer_cast(%c51040_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %203 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_133 = memref.cast %200 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_134 = memref.cast %202 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_135 = memref.cast %203 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_133, %cst_2, %cast_134, %cast_135) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %204 = hivm.hir.pointer_cast(%c51040_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %205 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_136 = memref.cast %202 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_137 = memref.cast %204 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_138 = memref.cast %205 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_136, %cast_137, %cast_138) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %206 = hivm.hir.pointer_cast(%c51040_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %207 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_139 = memref.cast %204 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_140 = memref.cast %206 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_141 = memref.cast %207 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_139, %cst_1, %cast_140, %cast_141) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %208 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_142 = memref.cast %51 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_143 = memref.cast %206 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_144 = memref.cast %51 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_145 = memref.cast %208 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_1d_float(%cast_142, %cast_143, %cast_144, %cast_145) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %209 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_146 = memref.cast %45 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_147 = memref.cast %51 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_148 = memref.cast %45 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_149 = memref.cast %209 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_146, %cast_147, %cast_148, %cast_149) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %104]
      %subview_150 = memref.subview %45[0] [%164] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_151 = memref.subview %reinterpret_cast_15[0] [%164] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_152 = memref.cast %subview_150 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_153 = memref.cast %subview_151 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_152, %cast_153, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %109]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
