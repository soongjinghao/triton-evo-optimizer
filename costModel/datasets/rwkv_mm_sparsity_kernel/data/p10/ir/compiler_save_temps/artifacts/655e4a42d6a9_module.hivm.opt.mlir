#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (512, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0] -> (s0 mod 2)>
#map6 = affine_map<()[s0] -> (1024, s0)>
#map7 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map8 = affine_map<()[s0, s1] -> (s0 - s1)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 512)>
#map10 = affine_map<()[s0, s1] -> (128, s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @rwkv_mm_sparsity_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmps_eq_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vand_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @rwkv_mm_sparsity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c0_i64 = arith.constant 0 : i64
    %c512_i64 = arith.constant 512 : i64
    %c768_i64 = arith.constant 768 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %c1664_i64 = arith.constant 1664 : i64
    %c1760_i64 = arith.constant 1760 : i64
    %c18656_i64 = arith.constant 18656 : i64
    %c51456_i64 = arith.constant 51456 : i64
    %c51712_i64 = arith.constant 51712 : i64
    %c51808_i64 = arith.constant 51808 : i64
    %c68192_i64 = arith.constant 68192 : i64
    %c72288_i64 = arith.constant 72288 : i64
    %c76384_i64 = arith.constant 76384 : i64
    %c109152_i64 = arith.constant 109152 : i64
    %c109216_i64 = arith.constant 109216 : i64
    %c109472_i64 = arith.constant 109472 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c142496_i64 = arith.constant 142496 : i64
    %c18144_i64 = arith.constant 18144 : i64
    %c143008_i64 = arith.constant 143008 : i64
    %c18688_i64 = arith.constant 18688 : i64
    %c142240_i64 = arith.constant 142240 : i64
    %c175776_i64 = arith.constant 175776 : i64
    %0 = hivm.hir.pointer_cast(%c142240_i64) : memref<64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c175776_i64) : memref<64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c18688_i64) : memref<128x64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c143008_i64) : memref<128x64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c18144_i64) : memref<128x1xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c142496_i64) : memref<128x1xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg6, %arg7 : i32
    %7 = arith.muli %6, %arg8 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg8, %arg7 : i32
    %11 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
    %cast = memref.cast %11 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %12 = hivm.hir.pointer_cast(%c512_i64) : memref<64xf32, #hivm.address_space<ub>>
    %cast_1 = memref.cast %12 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast_1) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %13 = hivm.hir.pointer_cast(%c768_i64) : memref<64xi32, #hivm.address_space<ub>>
    %cast_2 = memref.cast %13 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_2, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    scf.for %arg9 = %9 to %7 step %c48_i32  : i32 {
      %14 = arith.index_cast %arg9 : i32 to index
      %15 = arith.index_cast %9 : i32 to index
      %16 = affine.apply #map()[%14, %15]
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.cmpi eq, %17, %c1_i64 : i64
      %19 = arith.select %18, %1, %0 : memref<64xf32, #hivm.address_space<ub>>
      %20 = arith.index_cast %arg9 : i32 to index
      %21 = arith.index_cast %9 : i32 to index
      %22 = affine.apply #map()[%20, %21]
      %23 = arith.index_cast %22 : index to i1
      %24 = arith.xori %23, %true : i1
      %25 = arith.extui %24 : i1 to i64
      hivm.hir.set_mask_norm
      %26 = arith.divsi %arg9, %10 : i32
      %27 = arith.remsi %26, %arg6 : i32
      %28 = arith.muli %27, %c64_i32 : i32
      %29 = hivm.hir.pointer_cast(%c1024_i64) : memref<64xi32, #hivm.address_space<ub>>
      %30 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_3 = memref.cast %13 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_4 = memref.cast %29 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %30 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_3, %28, %cast_4, %cast_5) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %31 = hivm.hir.pointer_cast(%c1280_i64) : memref<64xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %32 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_6 = memref.cast %29 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %31 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %32 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_6, %c512_i32, %cast_7, %cast_8) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %33 = hivm.hir.pointer_cast(%c1024_i64) : memref<64xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_9 = memref.cast %31 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %29 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %33 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_9, %cast_10, %cast_11) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %34 = hivm.hir.pointer_cast(%c1024_i64) : memref<64xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %35 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_12 = memref.cast %33 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %34 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %35 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_12, %cast_13, %cast_14) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %36 = hivm.hir.pointer_cast(%c1536_i64) : memref<64xf16, #hivm.address_space<ub>>
      %37 = hivm.hir.pointer_cast(%c1664_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_15 = memref.cast %34 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %36 : memref<64xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %37 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_half_1d_with_temp(%cast_15, %cast_16, %cast_17, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %expand_shape = memref.expand_shape %36 [[0, 1]] output_shape [1, 64] : memref<64xf16, #hivm.address_space<ub>> into memref<1x64xf16, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c1760_i64) : memref<128x64xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_18 = memref.cast %expand_shape : memref<1x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %38 : memref<128x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_half(%cast_18, %cast_19) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %39 = hivm.hir.pointer_cast(%c1760_i64) : memref<128x64xi1, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %38 [[0, 1]] : memref<128x64xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
      %collapse_shape_20 = memref.collapse_shape %39 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_21 = memref.cast %collapse_shape : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %collapse_shape_20 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmps_eq_1d_half(%cast_21, %cst, %cast_22) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %40 = hivm.hir.pointer_cast(%c1760_i64) : memref<128x64xi1, #hivm.address_space<ub>>
      %collapse_shape_23 = memref.collapse_shape %40 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %41 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_24 = memref.cast %collapse_shape_20 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_25 = memref.cast %collapse_shape_23 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %41 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_24, %cast_25, %cast_26) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %42 = arith.index_cast %28 : i32 to index
      %43 = affine.max #map1()[%42]
      %44 = affine.min #map2()[%43, %42]
      %45 = affine.min #map3()[%44, %42]
      %46 = affine.max #map4()[%45]
      %47 = arith.cmpi slt, %46, %c64 : index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %25]
      %cast_27 = memref.cast %12 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_28 = memref.cast %19 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_27, %cast_28, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %48 = scf.for %arg10 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg11 = %19) -> (memref<64xf32, #hivm.address_space<ub>>)  : i32 {
        %50 = arith.index_cast %arg10 : i32 to index
        %51 = affine.apply #map5()[%50]
        %52 = arith.index_cast %51 : index to i64
        %53 = arith.cmpi eq, %52, %c1_i64 : i64
        %54 = arith.select %53, %3, %2 : memref<128x64xf32, #hivm.address_space<ub>>
        %55 = arith.index_cast %arg10 : i32 to index
        %56 = affine.apply #map5()[%55]
        %57 = arith.index_cast %56 : index to i64
        %58 = arith.cmpi eq, %57, %c1_i64 : i64
        %59 = arith.select %58, %5, %4 : memref<128x1xf32, #hivm.address_space<ub>>
        %60 = arith.index_cast %arg10 : i32 to index
        %61 = affine.apply #map5()[%60]
        %62 = arith.index_cast %61 : index to i1
        %63 = arith.select %62, %c3_i64, %c4_i64 : i64
        %64 = arith.index_cast %61 : index to i1
        %65 = arith.select %64, %c1_i64, %c2_i64 : i64
        %66 = arith.muli %arg10, %c128_i32 : i32
        %67 = arith.index_cast %66 : i32 to index
        %reinterpret_cast_32 = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [128, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
        %68 = affine.max #map6()[%67]
        %69 = affine.min #map7()[%68, %67]
        %70 = affine.apply #map8()[%69, %67]
        %71 = arith.cmpi slt, %70, %c128 : index
        %subview_33 = memref.subview %reinterpret_cast_32[0, 0] [%70, 1] [1, 1] : memref<128x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_34 = memref.subview %59[0, 0] [%70, 1] [1, 1] : memref<128x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.if %71 {
          %collapse_shape_89 = memref.collapse_shape %59 [[0, 1]] : memref<128x1xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
          %cast_90 = memref.cast %collapse_shape_89 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_90) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %collapse_shape_35 = memref.collapse_shape %subview_33 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_36 = memref.collapse_shape %subview_34 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %65]
        %cast_37 = memref.cast %collapse_shape_35 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_38 = memref.cast %collapse_shape_36 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_37, %cast_38, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_39 = memref.collapse_shape %59 [[0, 1]] : memref<128x1xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        %72 = hivm.hir.pointer_cast(%c18656_i64) : memref<128xi1, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_40 = memref.cast %collapse_shape_39 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %11 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_42 = memref.cast %72 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_40, %cast_41, %cast_42) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %73 = hivm.hir.pointer_cast(%c18656_i64) : memref<128xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %74 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_43 = memref.cast %72 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %73 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_45 = memref.cast %74 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_43, %cast_44, %cast_45) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %75 = affine.apply #map9()[%42, %67]
        %reinterpret_cast_46 = memref.reinterpret_cast %arg4 to offset: [%75], sizes: [128, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %76 = affine.min #map10()[%69, %67]
        %77 = affine.max #map4()[%76]
        %78 = arith.cmpi slt, %77, %c128 : index
        %79 = arith.ori %78, %47 : i1
        %subview_47 = memref.subview %reinterpret_cast_46[0, 0] [%77, %46] [1, 1] : memref<128x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_48 = memref.subview %54[0, 0] [%77, %46] [1, 1] : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
        scf.if %79 {
          %collapse_shape_89 = memref.collapse_shape %54 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          %cast_90 = memref.cast %collapse_shape_89 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_90) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %cast_49 = memref.cast %subview_47 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_50 = memref.cast %subview_48 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_49, %cast_50, %c2_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %80 = hivm.hir.pointer_cast(%c51456_i64) : memref<128xf16, #hivm.address_space<ub>>
        %81 = hivm.hir.pointer_cast(%c51712_i64) : memref<48xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_51 = memref.cast %73 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_52 = memref.cast %80 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_53 = memref.cast %81 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bool_to_half_1d_with_temp(%cast_51, %cast_52, %cast_53, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %expand_shape_54 = memref.expand_shape %80 [[0, 1]] output_shape [128, 1] : memref<128xf16, #hivm.address_space<ub>> into memref<128x1xf16, #hivm.address_space<ub>>
        %82 = hivm.hir.pointer_cast(%c51808_i64) : memref<128x64xf16, #hivm.address_space<ub>>
        %83 = hivm.hir.pointer_cast(%c68192_i64) : memref<2048xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_55 = memref.cast %expand_shape_54 : memref<128x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_56 = memref.cast %82 : memref<128x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_57 = memref.cast %83 : memref<2048xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_half(%cast_55, %cast_56, %cast_57) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %84 = hivm.hir.pointer_cast(%c51808_i64) : memref<128x64xi1, #hivm.address_space<ub>>
        %collapse_shape_58 = memref.collapse_shape %82 [[0, 1]] : memref<128x64xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
        %collapse_shape_59 = memref.collapse_shape %84 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_60 = memref.cast %collapse_shape_58 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_61 = memref.cast %collapse_shape_59 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmps_eq_1d_half(%cast_60, %cst, %cast_61) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %85 = hivm.hir.pointer_cast(%c51808_i64) : memref<128x64xi1, #hivm.address_space<ub>>
        %collapse_shape_62 = memref.collapse_shape %85 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %86 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_63 = memref.cast %collapse_shape_59 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_64 = memref.cast %collapse_shape_62 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_65 = memref.cast %86 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_63, %cast_64, %cast_65) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %87 = hivm.hir.pointer_cast(%c51808_i64) : memref<128x64xi1, #hivm.address_space<ub>>
        %collapse_shape_66 = memref.collapse_shape %87 [[0, 1]] : memref<128x64xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %88 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_67 = memref.cast %collapse_shape_62 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_68 = memref.cast %collapse_shape_23 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_69 = memref.cast %collapse_shape_66 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %88 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vand_1d_bool(%cast_67, %cast_68, %cast_69, %cast_70) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %89 = hivm.hir.pointer_cast(%c72288_i64) : memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_71 = memref.cast %59 : memref<128x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_72 = memref.cast %54 : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_73 = memref.cast %54 : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_74 = memref.cast %89 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_71, %cast_72, %cast_73, %cast_74) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %65]
        %90 = hivm.hir.pointer_cast(%c76384_i64) : memref<128x64xf32, #hivm.address_space<ub>>
        %collapse_shape_75 = memref.collapse_shape %54 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %collapse_shape_76 = memref.collapse_shape %90 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %91 = hivm.hir.pointer_cast(%c109152_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_77 = memref.cast %collapse_shape_66 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_78 = memref.cast %collapse_shape_75 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_79 = memref.cast %collapse_shape_76 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_80 = memref.cast %91 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vs_1d_float(%cast_77, %cast_78, %cst_0, %cast_79, %cast_80) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %92 = hivm.hir.pointer_cast(%c109216_i64) : memref<1x64xf32, #hivm.address_space<ub>>
        %93 = hivm.hir.pointer_cast(%c109472_i64) : memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_81 = memref.cast %90 : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_82 = memref.cast %92 : memref<1x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_83 = memref.cast %93 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_81, %cast_82, %cast_83, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_84 = memref.collapse_shape %92 [[0, 1]] : memref<1x64xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %94 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_85 = memref.cast %arg11 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_86 = memref.cast %collapse_shape_84 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_87 = memref.cast %19 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_88 = memref.cast %94 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_85, %cast_86, %cast_87, %cast_88) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %19 : memref<64xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%42], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %49 = affine.apply #map8()[%44, %42]
      %subview = memref.subview %48[0] [%49] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_29 = memref.subview %reinterpret_cast[0] [%49] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_30 = memref.cast %subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %subview_29 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_30, %cast_31, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %25]
    } {autoblockify.subloop}
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
