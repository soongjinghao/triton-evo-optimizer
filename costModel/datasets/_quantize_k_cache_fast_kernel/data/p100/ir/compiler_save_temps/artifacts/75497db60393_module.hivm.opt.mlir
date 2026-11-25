#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0] -> (512, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
#map5 = affine_map<()[s0] -> (64, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_quantize_k_cache_fast_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_bfloat16_t_to_1d(bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bfloat16_t_to_float_1d_with_mode(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vabs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmins_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_sv_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_quantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c128 = arith.constant 128 : index
    %c512_i32 = arith.constant 512 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant 4.480000e+02 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant -4.480000e+02 : f32
    %c0_i64 = arith.constant 0 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c1792_i64 = arith.constant 1792 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c2336_i64 = arith.constant 2336 : i64
    %c2592_i64 = arith.constant 2592 : i64
    %c2624_i64 = arith.constant 2624 : i64
    %c2656_i64 = arith.constant 2656 : i64
    %c3168_i64 = arith.constant 3168 : i64
    %c3680_i64 = arith.constant 3680 : i64
    %c3744_i64 = arith.constant 3744 : i64
    %c4256_i64 = arith.constant 4256 : i64
    %c4288_i64 = arith.constant 4288 : i64
    %c4320_i64 = arith.constant 4320 : i64
    %c4352_i64 = arith.constant 4352 : i64
    %c4864_i64 = arith.constant 4864 : i64
    %c5376_i64 = arith.constant 5376 : i64
    %c5920_i64 = arith.constant 5920 : i64
    %c5984_i64 = arith.constant 5984 : i64
    %c6304_i64 = arith.constant 6304 : i64
    %c6816_i64 = arith.constant 6816 : i64
    %c7328_i64 = arith.constant 7328 : i64
    %c7360_i64 = arith.constant 7360 : i64
    %c7616_i64 = arith.constant 7616 : i64
    %c7648_i64 = arith.constant 7648 : i64
    %c7680_i64 = arith.constant 7680 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c8704_i64 = arith.constant 8704 : i64
    %c8768_i64 = arith.constant 8768 : i64
    %c9280_i64 = arith.constant 9280 : i64
    %c9312_i64 = arith.constant 9312 : i64
    %c9344_i64 = arith.constant 9344 : i64
    %c9376_i64 = arith.constant 9376 : i64
    %c9888_i64 = arith.constant 9888 : i64
    %c10400_i64 = arith.constant 10400 : i64
    %c10944_i64 = arith.constant 10944 : i64
    %c11008_i64 = arith.constant 11008 : i64
    %c11328_i64 = arith.constant 11328 : i64
    %c11840_i64 = arith.constant 11840 : i64
    %c12352_i64 = arith.constant 12352 : i64
    %c12384_i64 = arith.constant 12384 : i64
    %c12640_i64 = arith.constant 12640 : i64
    %c12672_i64 = arith.constant 12672 : i64
    %c12704_i64 = arith.constant 12704 : i64
    %c13216_i64 = arith.constant 13216 : i64
    %c13728_i64 = arith.constant 13728 : i64
    %c13792_i64 = arith.constant 13792 : i64
    %c14304_i64 = arith.constant 14304 : i64
    %c14336_i64 = arith.constant 14336 : i64
    %c14368_i64 = arith.constant 14368 : i64
    %c14400_i64 = arith.constant 14400 : i64
    %c14912_i64 = arith.constant 14912 : i64
    %c15424_i64 = arith.constant 15424 : i64
    %c15968_i64 = arith.constant 15968 : i64
    %c16032_i64 = arith.constant 16032 : i64
    %c16352_i64 = arith.constant 16352 : i64
    %c16864_i64 = arith.constant 16864 : i64
    %c17376_i64 = arith.constant 17376 : i64
    %c17408_i64 = arith.constant 17408 : i64
    %c17664_i64 = arith.constant 17664 : i64
    %c17696_i64 = arith.constant 17696 : i64
    %c17728_i64 = arith.constant 17728 : i64
    %c18240_i64 = arith.constant 18240 : i64
    %c18752_i64 = arith.constant 18752 : i64
    %c18816_i64 = arith.constant 18816 : i64
    %c19328_i64 = arith.constant 19328 : i64
    %c19360_i64 = arith.constant 19360 : i64
    %c19392_i64 = arith.constant 19392 : i64
    %c19424_i64 = arith.constant 19424 : i64
    %c19936_i64 = arith.constant 19936 : i64
    %c20448_i64 = arith.constant 20448 : i64
    %c20992_i64 = arith.constant 20992 : i64
    %c21056_i64 = arith.constant 21056 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c21376_i64 = arith.constant 21376 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c21632_i64 = arith.constant 21632 : i64
    %c5408_i64 = arith.constant 5408 : i64
    %c22144_i64 = arith.constant 22144 : i64
    %c6016_i64 = arith.constant 6016 : i64
    %c22176_i64 = arith.constant 22176 : i64
    %c6048_i64 = arith.constant 6048 : i64
    %c22432_i64 = arith.constant 22432 : i64
    %c10432_i64 = arith.constant 10432 : i64
    %c22944_i64 = arith.constant 22944 : i64
    %c11040_i64 = arith.constant 11040 : i64
    %c22976_i64 = arith.constant 22976 : i64
    %c11072_i64 = arith.constant 11072 : i64
    %c23232_i64 = arith.constant 23232 : i64
    %c15456_i64 = arith.constant 15456 : i64
    %c23744_i64 = arith.constant 23744 : i64
    %c16064_i64 = arith.constant 16064 : i64
    %c23776_i64 = arith.constant 23776 : i64
    %c16096_i64 = arith.constant 16096 : i64
    %c24032_i64 = arith.constant 24032 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c24544_i64 = arith.constant 24544 : i64
    %c21088_i64 = arith.constant 21088 : i64
    %c21120_i64 = arith.constant 21120 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %0 = hivm.hir.pointer_cast(%c21120_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c24576_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c21088_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c24544_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c20480_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c24032_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c16096_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c23776_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c16064_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c23744_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c15456_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c23232_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c11072_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c22976_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c11040_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c22944_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c10432_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c22432_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c6048_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c22176_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c6016_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c22144_i64) : memref<1xbf16, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c5408_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c21632_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c21376_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c5408_i64) : memref<128xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c21632_i64) : memref<128xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c10432_i64) : memref<128xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c22432_i64) : memref<128xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c15456_i64) : memref<128xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c23232_i64) : memref<128xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c20480_i64) : memref<128xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c24032_i64) : memref<128xf32, #hivm.address_space<ub>>
    %34 = arith.muli %arg13, %arg14 : i32
    %35 = arith.muli %34, %arg15 : i32
    %36 = hivm.hir.get_block_idx -> i64
    %37 = arith.trunci %36 : i64 to i32
    %38 = arith.muli %arg15, %arg14 : i32
    %39 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
    %cast = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_3, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %40 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf32, #hivm.address_space<ub>>
    %cast_4 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_1, %cast_4) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    scf.for %arg16 = %37 to %35 step %c48_i32  : i32 {
      %41 = arith.index_cast %arg16 : i32 to index
      %42 = arith.index_cast %37 : i32 to index
      %43 = affine.apply #map()[%41, %42]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.cmpi eq, %44, %c1_i64 : i64
      %46 = arith.select %45, %1, %0 : memref<128xbf16, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg16 : i32 to index
      %48 = arith.index_cast %37 : i32 to index
      %49 = affine.apply #map()[%47, %48]
      %50 = arith.index_cast %49 : index to i64
      %51 = arith.cmpi eq, %50, %c1_i64 : i64
      %52 = arith.select %51, %3, %2 : memref<1xbf16, #hivm.address_space<ub>>
      %53 = arith.index_cast %arg16 : i32 to index
      %54 = arith.index_cast %37 : i32 to index
      %55 = affine.apply #map()[%53, %54]
      %56 = arith.index_cast %55 : index to i64
      %57 = arith.cmpi eq, %56, %c1_i64 : i64
      %58 = arith.select %57, %5, %4 : memref<128xbf16, #hivm.address_space<ub>>
      %59 = arith.index_cast %arg16 : i32 to index
      %60 = arith.index_cast %37 : i32 to index
      %61 = affine.apply #map()[%59, %60]
      %62 = arith.index_cast %61 : index to i64
      %63 = arith.cmpi eq, %62, %c1_i64 : i64
      %64 = arith.select %63, %7, %6 : memref<128xbf16, #hivm.address_space<ub>>
      %65 = arith.index_cast %arg16 : i32 to index
      %66 = arith.index_cast %37 : i32 to index
      %67 = affine.apply #map()[%65, %66]
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.cmpi eq, %68, %c1_i64 : i64
      %70 = arith.select %69, %9, %8 : memref<1xbf16, #hivm.address_space<ub>>
      %71 = arith.index_cast %arg16 : i32 to index
      %72 = arith.index_cast %37 : i32 to index
      %73 = affine.apply #map()[%71, %72]
      %74 = arith.index_cast %73 : index to i64
      %75 = arith.cmpi eq, %74, %c1_i64 : i64
      %76 = arith.select %75, %11, %10 : memref<128xbf16, #hivm.address_space<ub>>
      %77 = arith.index_cast %arg16 : i32 to index
      %78 = arith.index_cast %37 : i32 to index
      %79 = affine.apply #map()[%77, %78]
      %80 = arith.index_cast %79 : index to i64
      %81 = arith.cmpi eq, %80, %c1_i64 : i64
      %82 = arith.select %81, %13, %12 : memref<128xbf16, #hivm.address_space<ub>>
      %83 = arith.index_cast %arg16 : i32 to index
      %84 = arith.index_cast %37 : i32 to index
      %85 = affine.apply #map()[%83, %84]
      %86 = arith.index_cast %85 : index to i64
      %87 = arith.cmpi eq, %86, %c1_i64 : i64
      %88 = arith.select %87, %15, %14 : memref<1xbf16, #hivm.address_space<ub>>
      %89 = arith.index_cast %arg16 : i32 to index
      %90 = arith.index_cast %37 : i32 to index
      %91 = affine.apply #map()[%89, %90]
      %92 = arith.index_cast %91 : index to i64
      %93 = arith.cmpi eq, %92, %c1_i64 : i64
      %94 = arith.select %93, %17, %16 : memref<128xbf16, #hivm.address_space<ub>>
      %95 = arith.index_cast %arg16 : i32 to index
      %96 = arith.index_cast %37 : i32 to index
      %97 = affine.apply #map()[%95, %96]
      %98 = arith.index_cast %97 : index to i64
      %99 = arith.cmpi eq, %98, %c1_i64 : i64
      %100 = arith.select %99, %19, %18 : memref<128xbf16, #hivm.address_space<ub>>
      %101 = arith.index_cast %arg16 : i32 to index
      %102 = arith.index_cast %37 : i32 to index
      %103 = affine.apply #map()[%101, %102]
      %104 = arith.index_cast %103 : index to i64
      %105 = arith.cmpi eq, %104, %c1_i64 : i64
      %106 = arith.select %105, %21, %20 : memref<1xbf16, #hivm.address_space<ub>>
      %107 = arith.index_cast %arg16 : i32 to index
      %108 = arith.index_cast %37 : i32 to index
      %109 = affine.apply #map()[%107, %108]
      %110 = arith.index_cast %109 : index to i64
      %111 = arith.cmpi eq, %110, %c1_i64 : i64
      %112 = arith.select %111, %23, %22 : memref<128xbf16, #hivm.address_space<ub>>
      %113 = arith.index_cast %arg16 : i32 to index
      %114 = arith.index_cast %37 : i32 to index
      %115 = affine.apply #map()[%113, %114]
      %116 = arith.index_cast %115 : index to i64
      %117 = arith.cmpi eq, %116, %c1_i64 : i64
      %118 = arith.select %117, %25, %24 : memref<128xbf16, #hivm.address_space<ub>>
      %119 = arith.index_cast %arg16 : i32 to index
      %120 = arith.index_cast %37 : i32 to index
      %121 = affine.apply #map()[%119, %120]
      %122 = arith.index_cast %121 : index to i64
      %123 = arith.cmpi eq, %122, %c1_i64 : i64
      %124 = arith.select %123, %27, %26 : memref<128xf32, #hivm.address_space<ub>>
      %125 = arith.index_cast %arg16 : i32 to index
      %126 = arith.index_cast %37 : i32 to index
      %127 = affine.apply #map()[%125, %126]
      %128 = arith.index_cast %127 : index to i64
      %129 = arith.cmpi eq, %128, %c1_i64 : i64
      %130 = arith.select %129, %29, %28 : memref<128xf32, #hivm.address_space<ub>>
      %131 = arith.index_cast %arg16 : i32 to index
      %132 = arith.index_cast %37 : i32 to index
      %133 = affine.apply #map()[%131, %132]
      %134 = arith.index_cast %133 : index to i64
      %135 = arith.cmpi eq, %134, %c1_i64 : i64
      %136 = arith.select %135, %31, %30 : memref<128xf32, #hivm.address_space<ub>>
      %137 = arith.index_cast %arg16 : i32 to index
      %138 = arith.index_cast %37 : i32 to index
      %139 = affine.apply #map()[%137, %138]
      %140 = arith.index_cast %139 : index to i64
      %141 = arith.cmpi eq, %140, %c1_i64 : i64
      %142 = arith.select %141, %33, %32 : memref<128xf32, #hivm.address_space<ub>>
      %143 = arith.index_cast %arg16 : i32 to index
      %144 = arith.index_cast %37 : i32 to index
      %145 = affine.apply #map()[%143, %144]
      %146 = arith.index_cast %145 : index to i1
      %147 = arith.xori %146, %true : i1
      %148 = arith.extui %147 : i1 to i64
      %149 = arith.index_cast %145 : index to i1
      %150 = arith.select %149, %c2_i64, %c3_i64 : i64
      %151 = arith.index_cast %145 : index to i1
      %152 = arith.select %151, %c4_i64, %c5_i64 : i64
      %153 = arith.index_cast %145 : index to i1
      %154 = arith.select %153, %c6_i64, %c7_i64 : i64
      hivm.hir.set_mask_norm
      %155 = arith.divsi %arg16, %arg15 : i32
      %156 = arith.remsi %155, %arg14 : i32
      %157 = arith.divsi %arg16, %38 : i32
      %158 = arith.remsi %157, %arg13 : i32
      %159 = arith.cmpi slt, %156, %c1_i32 : i32
      scf.if %159 {
        %160 = arith.muli %156, %c4_i32 : i32
        %161 = arith.cmpi slt, %160, %c4_i32 : i32
        scf.if %161 {
          %168 = arith.muli %156, %c512_i32 : i32
          %169 = arith.muli %158, %arg11 : i32
          %170 = arith.index_cast %169 : i32 to index
          %171 = arith.index_cast %168 : i32 to index
          %172 = affine.apply #map1()[%170, %171]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%172], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %173 = affine.max #map2()[%171]
          %174 = affine.min #map3()[%173, %171]
          %175 = affine.apply #map4()[%174, %171]
          %176 = arith.cmpi slt, %175, %c128 : index
          %subview = memref.subview %reinterpret_cast[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_5 = memref.subview %118[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %176 {
            %cast_79 = memref.cast %118 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_79) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_6 = memref.cast %subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_7 = memref.cast %subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_6, %cast_7, %c2_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %177 = hivm.hir.pointer_cast(%c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_8 = memref.cast %118 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_9 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_8, %cast_9, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %178 = hivm.hir.pointer_cast(%c1792_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %179 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_10 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_11 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_12 = memref.cast %179 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vabs_1d_float(%cast_10, %cast_11, %cast_12) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %180 = hivm.hir.pointer_cast(%c2304_i64) : memref<1xf32, #hivm.address_space<ub>>
          %181 = hivm.hir.pointer_cast(%c2336_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_13 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_14 = memref.cast %180 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_15 = memref.cast %181 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_max_r_float(%cast_13, %cast_14, %cast_15, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %182 = memref.load %180[%c0] : memref<1xf32, #hivm.address_space<ub>>
          %183 = arith.divf %182, %cst_1 : f32
          %184 = arith.divf %cst_2, %183 : f32
          %185 = hivm.hir.pointer_cast(%c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
          %186 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_16 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_17 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_18 = memref.cast %186 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_16, %184, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %187 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_19 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_20 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_19, %cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %188 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %189 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_22 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_23 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_24 = memref.cast %189 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_22, %cast_23, %cast_24) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %190 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_25 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_26 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_25, %cast_26, %cast_27) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %191 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %192 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_28 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %192 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %193 = hivm.hir.pointer_cast(%c2656_i64) : memref<128xf32, #hivm.address_space<ub>>
          %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_31 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmins_vs_1d_float(%cast_31, %cst_1, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %195 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xf32, #hivm.address_space<ub>>
          %196 = hivm.hir.pointer_cast(%c3680_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_34 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_36 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_37 = memref.cast %196 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_34, %cst_1, %cast_35, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %197 = hivm.hir.pointer_cast(%c3744_i64) : memref<128xf32, #hivm.address_space<ub>>
          %198 = hivm.hir.pointer_cast(%c4256_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_38 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_39 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_40 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_41 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_42 = memref.cast %198 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_38, %cast_39, %cast_40, %cast_41, %cast_42) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %199 = hivm.hir.pointer_cast(%c4288_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_43 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_44 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_45 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_43, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %200 = hivm.hir.pointer_cast(%c4288_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %201 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_46 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_47 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_48 = memref.cast %201 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_46, %cast_47, %cast_48) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %202 = hivm.hir.pointer_cast(%c4320_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_49 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_50 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_51 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_49, %cast_50, %cast_51) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %203 = hivm.hir.pointer_cast(%c4320_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %204 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_52 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_53 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_54 = memref.cast %204 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_52, %cast_53, %cast_54) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %205 = hivm.hir.pointer_cast(%c4352_i64) : memref<128xf32, #hivm.address_space<ub>>
          %206 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_55 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_56 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_57 = memref.cast %206 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmaxs_vs_1d_float(%cast_55, %cst_3, %cast_56, %cast_57) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %207 = hivm.hir.pointer_cast(%c4864_i64) : memref<128xf32, #hivm.address_space<ub>>
          %208 = hivm.hir.pointer_cast(%c5376_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_58 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_59 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_60 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_61 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_62 = memref.cast %208 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_58, %cast_59, %cast_60, %cast_61, %cast_62) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %209 = hivm.hir.pointer_cast(%c5920_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_63 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_64 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_65 = memref.cast %124 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_66 = memref.cast %209 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_63, %cst_3, %cast_64, %cast_65, %cast_66) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_67 = memref.cast %124 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_68 = memref.cast %112 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_67, %cast_68, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %210 = arith.muli %158, %arg8 : i32
          %211 = arith.index_cast %210 : i32 to index
          %212 = affine.apply #map1()[%211, %171]
          %reinterpret_cast_69 = memref.reinterpret_cast %arg3 to offset: [%212], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %213 = arith.muli %158, %arg9 : i32
          %subview_70 = memref.subview %112[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          %subview_71 = memref.subview %reinterpret_cast_69[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_72 = memref.cast %subview_70 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_73 = memref.cast %subview_71 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_72, %cast_73, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
          %214 = hivm.hir.pointer_cast(%c5984_i64) : memref<1xf32, #hivm.address_space<ub>>
          memref.store %183, %214[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %215 = hivm.hir.pointer_cast(%c5984_i64) : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %cast_74 = memref.cast %214 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_75 = memref.cast %215 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_74, %cast_75, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %216 = memref.load %215[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %154]
          memref.store %216, %106[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %217 = arith.index_cast %213 : i32 to index
          %218 = arith.index_cast %160 : i32 to index
          %219 = affine.apply #map1()[%217, %218]
          %reinterpret_cast_76 = memref.reinterpret_cast %arg4 to offset: [%219], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_77 = memref.cast %106 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_78 = memref.cast %reinterpret_cast_76 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_77, %cast_78, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %154]
        }
        %162 = arith.addi %160, %c1_i32 : i32
        %163 = arith.cmpi slt, %162, %c4_i32 : i32
        scf.if %163 {
          %168 = arith.muli %162, %c128_i32 : i32
          %169 = arith.muli %158, %arg11 : i32
          %170 = arith.index_cast %169 : i32 to index
          %171 = arith.index_cast %168 : i32 to index
          %172 = affine.apply #map1()[%170, %171]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%172], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %173 = affine.max #map2()[%171]
          %174 = affine.min #map3()[%173, %171]
          %175 = affine.apply #map4()[%174, %171]
          %176 = arith.cmpi slt, %175, %c128 : index
          %subview = memref.subview %reinterpret_cast[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_5 = memref.subview %100[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %176 {
            %cast_79 = memref.cast %100 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_79) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_6 = memref.cast %subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_7 = memref.cast %subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_6, %cast_7, %c2_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %177 = hivm.hir.pointer_cast(%c6304_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_8 = memref.cast %100 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_9 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_8, %cast_9, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %178 = hivm.hir.pointer_cast(%c6816_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %179 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_10 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_11 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_12 = memref.cast %179 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vabs_1d_float(%cast_10, %cast_11, %cast_12) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %180 = hivm.hir.pointer_cast(%c7328_i64) : memref<1xf32, #hivm.address_space<ub>>
          %181 = hivm.hir.pointer_cast(%c7360_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_13 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_14 = memref.cast %180 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_15 = memref.cast %181 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_max_r_float(%cast_13, %cast_14, %cast_15, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %182 = memref.load %180[%c0] : memref<1xf32, #hivm.address_space<ub>>
          %183 = arith.divf %182, %cst_1 : f32
          %184 = arith.divf %cst_2, %183 : f32
          %185 = hivm.hir.pointer_cast(%c6304_i64) : memref<128xf32, #hivm.address_space<ub>>
          %186 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_16 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_17 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_18 = memref.cast %186 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_16, %184, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %187 = hivm.hir.pointer_cast(%c7616_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_19 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_20 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_19, %cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %188 = hivm.hir.pointer_cast(%c7616_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %189 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_22 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_23 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_24 = memref.cast %189 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_22, %cast_23, %cast_24) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %190 = hivm.hir.pointer_cast(%c7648_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_25 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_26 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_25, %cast_26, %cast_27) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %191 = hivm.hir.pointer_cast(%c7648_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %192 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_28 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %192 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %193 = hivm.hir.pointer_cast(%c7680_i64) : memref<128xf32, #hivm.address_space<ub>>
          %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_31 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmins_vs_1d_float(%cast_31, %cst_1, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %195 = hivm.hir.pointer_cast(%c8192_i64) : memref<128xf32, #hivm.address_space<ub>>
          %196 = hivm.hir.pointer_cast(%c8704_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_34 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_36 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_37 = memref.cast %196 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_34, %cst_1, %cast_35, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %197 = hivm.hir.pointer_cast(%c8768_i64) : memref<128xf32, #hivm.address_space<ub>>
          %198 = hivm.hir.pointer_cast(%c9280_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_38 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_39 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_40 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_41 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_42 = memref.cast %198 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_38, %cast_39, %cast_40, %cast_41, %cast_42) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %199 = hivm.hir.pointer_cast(%c9312_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_43 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_44 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_45 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_43, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %200 = hivm.hir.pointer_cast(%c9312_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %201 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_46 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_47 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_48 = memref.cast %201 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_46, %cast_47, %cast_48) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %202 = hivm.hir.pointer_cast(%c9344_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_49 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_50 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_51 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_49, %cast_50, %cast_51) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %203 = hivm.hir.pointer_cast(%c9344_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %204 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_52 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_53 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_54 = memref.cast %204 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_52, %cast_53, %cast_54) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %205 = hivm.hir.pointer_cast(%c9376_i64) : memref<128xf32, #hivm.address_space<ub>>
          %206 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_55 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_56 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_57 = memref.cast %206 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmaxs_vs_1d_float(%cast_55, %cst_3, %cast_56, %cast_57) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %207 = hivm.hir.pointer_cast(%c9888_i64) : memref<128xf32, #hivm.address_space<ub>>
          %208 = hivm.hir.pointer_cast(%c10400_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_58 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_59 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_60 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_61 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_62 = memref.cast %208 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_58, %cast_59, %cast_60, %cast_61, %cast_62) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %209 = hivm.hir.pointer_cast(%c10944_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_63 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_64 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_65 = memref.cast %130 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_66 = memref.cast %209 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_63, %cst_3, %cast_64, %cast_65, %cast_66) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_67 = memref.cast %130 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_68 = memref.cast %94 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_67, %cast_68, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %210 = arith.muli %158, %arg8 : i32
          %211 = arith.index_cast %210 : i32 to index
          %212 = affine.apply #map1()[%211, %171]
          %reinterpret_cast_69 = memref.reinterpret_cast %arg3 to offset: [%212], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %213 = arith.muli %158, %arg9 : i32
          %subview_70 = memref.subview %94[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          %subview_71 = memref.subview %reinterpret_cast_69[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_72 = memref.cast %subview_70 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_73 = memref.cast %subview_71 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_72, %cast_73, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
          %214 = hivm.hir.pointer_cast(%c11008_i64) : memref<1xf32, #hivm.address_space<ub>>
          memref.store %183, %214[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %215 = hivm.hir.pointer_cast(%c11008_i64) : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %cast_74 = memref.cast %214 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_75 = memref.cast %215 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_74, %cast_75, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %216 = memref.load %215[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %152]
          memref.store %216, %88[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %217 = arith.index_cast %213 : i32 to index
          %218 = arith.index_cast %162 : i32 to index
          %219 = affine.apply #map1()[%217, %218]
          %reinterpret_cast_76 = memref.reinterpret_cast %arg4 to offset: [%219], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_77 = memref.cast %88 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_78 = memref.cast %reinterpret_cast_76 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_77, %cast_78, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %152]
        }
        %164 = arith.addi %160, %c2_i32 : i32
        %165 = arith.cmpi slt, %164, %c4_i32 : i32
        scf.if %165 {
          %168 = arith.muli %164, %c128_i32 : i32
          %169 = arith.muli %158, %arg11 : i32
          %170 = arith.index_cast %169 : i32 to index
          %171 = arith.index_cast %168 : i32 to index
          %172 = affine.apply #map1()[%170, %171]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%172], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %173 = affine.max #map2()[%171]
          %174 = affine.min #map3()[%173, %171]
          %175 = affine.apply #map4()[%174, %171]
          %176 = arith.cmpi slt, %175, %c128 : index
          %subview = memref.subview %reinterpret_cast[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_5 = memref.subview %82[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %176 {
            %cast_79 = memref.cast %82 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_79) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_6 = memref.cast %subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_7 = memref.cast %subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_6, %cast_7, %c2_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %177 = hivm.hir.pointer_cast(%c11328_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_8 = memref.cast %82 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_9 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_8, %cast_9, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %178 = hivm.hir.pointer_cast(%c11840_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %179 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_10 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_11 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_12 = memref.cast %179 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vabs_1d_float(%cast_10, %cast_11, %cast_12) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %180 = hivm.hir.pointer_cast(%c12352_i64) : memref<1xf32, #hivm.address_space<ub>>
          %181 = hivm.hir.pointer_cast(%c12384_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_13 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_14 = memref.cast %180 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_15 = memref.cast %181 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_max_r_float(%cast_13, %cast_14, %cast_15, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %182 = memref.load %180[%c0] : memref<1xf32, #hivm.address_space<ub>>
          %183 = arith.divf %182, %cst_1 : f32
          %184 = arith.divf %cst_2, %183 : f32
          %185 = hivm.hir.pointer_cast(%c11328_i64) : memref<128xf32, #hivm.address_space<ub>>
          %186 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_16 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_17 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_18 = memref.cast %186 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_16, %184, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %187 = hivm.hir.pointer_cast(%c12640_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_19 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_20 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_19, %cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %188 = hivm.hir.pointer_cast(%c12640_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %189 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_22 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_23 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_24 = memref.cast %189 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_22, %cast_23, %cast_24) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %190 = hivm.hir.pointer_cast(%c12672_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_25 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_26 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_25, %cast_26, %cast_27) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %191 = hivm.hir.pointer_cast(%c12672_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %192 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_28 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %192 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %193 = hivm.hir.pointer_cast(%c12704_i64) : memref<128xf32, #hivm.address_space<ub>>
          %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_31 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmins_vs_1d_float(%cast_31, %cst_1, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %195 = hivm.hir.pointer_cast(%c13216_i64) : memref<128xf32, #hivm.address_space<ub>>
          %196 = hivm.hir.pointer_cast(%c13728_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_34 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_36 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_37 = memref.cast %196 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_34, %cst_1, %cast_35, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %197 = hivm.hir.pointer_cast(%c13792_i64) : memref<128xf32, #hivm.address_space<ub>>
          %198 = hivm.hir.pointer_cast(%c14304_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_38 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_39 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_40 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_41 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_42 = memref.cast %198 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_38, %cast_39, %cast_40, %cast_41, %cast_42) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %199 = hivm.hir.pointer_cast(%c14336_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_43 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_44 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_45 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_43, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %200 = hivm.hir.pointer_cast(%c14336_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %201 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_46 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_47 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_48 = memref.cast %201 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_46, %cast_47, %cast_48) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %202 = hivm.hir.pointer_cast(%c14368_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_49 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_50 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_51 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_49, %cast_50, %cast_51) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %203 = hivm.hir.pointer_cast(%c14368_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %204 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_52 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_53 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_54 = memref.cast %204 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_52, %cast_53, %cast_54) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %205 = hivm.hir.pointer_cast(%c14400_i64) : memref<128xf32, #hivm.address_space<ub>>
          %206 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_55 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_56 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_57 = memref.cast %206 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmaxs_vs_1d_float(%cast_55, %cst_3, %cast_56, %cast_57) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %207 = hivm.hir.pointer_cast(%c14912_i64) : memref<128xf32, #hivm.address_space<ub>>
          %208 = hivm.hir.pointer_cast(%c15424_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_58 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_59 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_60 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_61 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_62 = memref.cast %208 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_58, %cast_59, %cast_60, %cast_61, %cast_62) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %209 = hivm.hir.pointer_cast(%c15968_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_63 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_64 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_65 = memref.cast %136 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_66 = memref.cast %209 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_63, %cst_3, %cast_64, %cast_65, %cast_66) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_67 = memref.cast %136 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_68 = memref.cast %76 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_67, %cast_68, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %210 = arith.muli %158, %arg8 : i32
          %211 = arith.index_cast %210 : i32 to index
          %212 = affine.apply #map1()[%211, %171]
          %reinterpret_cast_69 = memref.reinterpret_cast %arg3 to offset: [%212], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %213 = arith.muli %158, %arg9 : i32
          %subview_70 = memref.subview %76[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          %subview_71 = memref.subview %reinterpret_cast_69[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_72 = memref.cast %subview_70 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_73 = memref.cast %subview_71 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_72, %cast_73, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
          %214 = hivm.hir.pointer_cast(%c16032_i64) : memref<1xf32, #hivm.address_space<ub>>
          memref.store %183, %214[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %215 = hivm.hir.pointer_cast(%c16032_i64) : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %cast_74 = memref.cast %214 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_75 = memref.cast %215 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_74, %cast_75, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %216 = memref.load %215[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %150]
          memref.store %216, %70[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %217 = arith.index_cast %213 : i32 to index
          %218 = arith.index_cast %164 : i32 to index
          %219 = affine.apply #map1()[%217, %218]
          %reinterpret_cast_76 = memref.reinterpret_cast %arg4 to offset: [%219], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_77 = memref.cast %70 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_78 = memref.cast %reinterpret_cast_76 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_77, %cast_78, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %150]
        }
        %166 = arith.addi %160, %c3_i32 : i32
        %167 = arith.cmpi slt, %166, %c4_i32 : i32
        scf.if %167 {
          %168 = arith.muli %166, %c128_i32 : i32
          %169 = arith.muli %158, %arg11 : i32
          %170 = arith.index_cast %169 : i32 to index
          %171 = arith.index_cast %168 : i32 to index
          %172 = affine.apply #map1()[%170, %171]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%172], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %173 = affine.max #map2()[%171]
          %174 = affine.min #map3()[%173, %171]
          %175 = affine.apply #map4()[%174, %171]
          %176 = arith.cmpi slt, %175, %c128 : index
          %subview = memref.subview %reinterpret_cast[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_5 = memref.subview %64[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %176 {
            %cast_79 = memref.cast %64 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_79) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_6 = memref.cast %subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_7 = memref.cast %subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_6, %cast_7, %c2_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %177 = hivm.hir.pointer_cast(%c16352_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_8 = memref.cast %64 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_9 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_8, %cast_9, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %178 = hivm.hir.pointer_cast(%c16864_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %179 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_10 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_11 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_12 = memref.cast %179 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vabs_1d_float(%cast_10, %cast_11, %cast_12) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %180 = hivm.hir.pointer_cast(%c17376_i64) : memref<1xf32, #hivm.address_space<ub>>
          %181 = hivm.hir.pointer_cast(%c17408_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_13 = memref.cast %178 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_14 = memref.cast %180 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_15 = memref.cast %181 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_max_r_float(%cast_13, %cast_14, %cast_15, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %182 = memref.load %180[%c0] : memref<1xf32, #hivm.address_space<ub>>
          %183 = arith.divf %182, %cst_1 : f32
          %184 = arith.divf %cst_2, %183 : f32
          %185 = hivm.hir.pointer_cast(%c16352_i64) : memref<128xf32, #hivm.address_space<ub>>
          %186 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_16 = memref.cast %177 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_17 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_18 = memref.cast %186 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_16, %184, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %187 = hivm.hir.pointer_cast(%c17664_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_19 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_20 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_19, %cast_20, %cast_21) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %188 = hivm.hir.pointer_cast(%c17664_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %189 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_22 = memref.cast %187 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_23 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_24 = memref.cast %189 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_22, %cast_23, %cast_24) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %190 = hivm.hir.pointer_cast(%c17696_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_25 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_26 = memref.cast %40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_25, %cast_26, %cast_27) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %191 = hivm.hir.pointer_cast(%c17696_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %192 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_28 = memref.cast %190 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %192 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %193 = hivm.hir.pointer_cast(%c17728_i64) : memref<128xf32, #hivm.address_space<ub>>
          %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_31 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmins_vs_1d_float(%cast_31, %cst_1, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %195 = hivm.hir.pointer_cast(%c18240_i64) : memref<128xf32, #hivm.address_space<ub>>
          %196 = hivm.hir.pointer_cast(%c18752_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_34 = memref.cast %188 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %193 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_36 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_37 = memref.cast %196 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_34, %cst_1, %cast_35, %cast_36, %cast_37) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %197 = hivm.hir.pointer_cast(%c18816_i64) : memref<128xf32, #hivm.address_space<ub>>
          %198 = hivm.hir.pointer_cast(%c19328_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_38 = memref.cast %191 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_39 = memref.cast %185 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_40 = memref.cast %195 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_41 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_42 = memref.cast %198 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_38, %cast_39, %cast_40, %cast_41, %cast_42) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %199 = hivm.hir.pointer_cast(%c19360_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_43 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_44 = memref.cast %39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_45 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_43, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %200 = hivm.hir.pointer_cast(%c19360_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %201 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_46 = memref.cast %199 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_47 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_48 = memref.cast %201 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_46, %cast_47, %cast_48) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %202 = hivm.hir.pointer_cast(%c19392_i64) : memref<128xi1, #hivm.address_space<ub>>
          %cast_49 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_50 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_51 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_49, %cast_50, %cast_51) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %203 = hivm.hir.pointer_cast(%c19392_i64) : memref<128xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %204 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
          %cast_52 = memref.cast %202 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_53 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_54 = memref.cast %204 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vnot_1d_bool(%cast_52, %cast_53, %cast_54) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %205 = hivm.hir.pointer_cast(%c19424_i64) : memref<128xf32, #hivm.address_space<ub>>
          %206 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_55 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_56 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_57 = memref.cast %206 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmaxs_vs_1d_float(%cast_55, %cst_3, %cast_56, %cast_57) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %207 = hivm.hir.pointer_cast(%c19936_i64) : memref<128xf32, #hivm.address_space<ub>>
          %208 = hivm.hir.pointer_cast(%c20448_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_58 = memref.cast %200 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_59 = memref.cast %197 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_60 = memref.cast %205 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_61 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_62 = memref.cast %208 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_58, %cast_59, %cast_60, %cast_61, %cast_62) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %209 = hivm.hir.pointer_cast(%c20992_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_63 = memref.cast %203 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_64 = memref.cast %207 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_65 = memref.cast %142 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_66 = memref.cast %209 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_63, %cst_3, %cast_64, %cast_65, %cast_66) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_67 = memref.cast %142 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_68 = memref.cast %58 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_67, %cast_68, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %210 = arith.muli %158, %arg8 : i32
          %211 = arith.index_cast %210 : i32 to index
          %212 = affine.apply #map1()[%211, %171]
          %reinterpret_cast_69 = memref.reinterpret_cast %arg3 to offset: [%212], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %213 = arith.muli %158, %arg9 : i32
          %subview_70 = memref.subview %58[0] [%175] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          %subview_71 = memref.subview %reinterpret_cast_69[0] [%175] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_72 = memref.cast %subview_70 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_73 = memref.cast %subview_71 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_72, %cast_73, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          %214 = hivm.hir.pointer_cast(%c21056_i64) : memref<1xf32, #hivm.address_space<ub>>
          memref.store %183, %214[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %215 = hivm.hir.pointer_cast(%c21056_i64) : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %cast_74 = memref.cast %214 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_75 = memref.cast %215 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_74, %cast_75, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %216 = memref.load %215[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %148]
          memref.store %216, %52[%c0] : memref<1xbf16, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %217 = arith.index_cast %213 : i32 to index
          %218 = arith.index_cast %166 : i32 to index
          %219 = affine.apply #map1()[%217, %218]
          %reinterpret_cast_76 = memref.reinterpret_cast %arg4 to offset: [%219], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_77 = memref.cast %52 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_78 = memref.cast %reinterpret_cast_76 : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_77, %cast_78, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %148]
        }
      } else {
        %160 = arith.subi %156, %c1_i32 : i32
        %161 = arith.cmpi slt, %160, %c1_i32 : i32
        scf.if %161 {
          %162 = arith.muli %160, %c128_i32 : i32
          %163 = arith.muli %158, %arg12 : i32
          %164 = arith.index_cast %163 : i32 to index
          %165 = arith.index_cast %162 : i32 to index
          %166 = affine.apply #map1()[%164, %165]
          %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%166], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %167 = arith.muli %158, %arg10 : i32
          %168 = arith.index_cast %167 : i32 to index
          %169 = affine.apply #map1()[%168, %165]
          %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%169], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %170 = affine.max #map5()[%165]
          %171 = affine.min #map3()[%170, %165]
          %172 = affine.apply #map4()[%171, %165]
          %173 = arith.cmpi slt, %172, %c128 : index
          %subview = memref.subview %reinterpret_cast[0] [%172] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_6 = memref.subview %46[0] [%172] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %173 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_12 = memref.cast %46 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_0, %cast_12) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %148]
          %cast_7 = memref.cast %subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_8 = memref.cast %subview_6 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_7, %cast_8, %c2_i32, %cst_0, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %subview_9 = memref.subview %reinterpret_cast_5[0] [%172] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_10 = memref.cast %subview_6 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_11 = memref.cast %subview_9 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_10, %cast_11, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %148]
        }
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
