#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (512, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map3 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0] -> (s0 mod 2)>
#map6 = affine_map<()[s0] -> (1024, s0)>
#map7 = affine_map<()[s0, s1] -> (s1 + 512, s0)>
#map8 = affine_map<()[s0, s1] -> (s0 - s1)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 512)>
#map10 = affine_map<()[s0, s1] -> (512, s0 - s1)>
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
  func.func private @broadcast_last_axis_align_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vand_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @rwkv_mm_sparsity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %c16_i32 = arith.constant 16 : i32
    %c512_i32 = arith.constant 512 : i32
    %c69760_i64 = arith.constant 69760 : i64
    %c71808_i64 = arith.constant 71808 : i64
    %c71872_i64 = arith.constant 71872 : i64
    %c71936_i64 = arith.constant 71936 : i64
    %c73984_i64 = arith.constant 73984 : i64
    %c74048_i64 = arith.constant 74048 : i64
    %c74112_i64 = arith.constant 74112 : i64
    %c74016_i64 = arith.constant 74016 : i64
    %c90400_i64 = arith.constant 90400 : i64
    %c92448_i64 = arith.constant 92448 : i64
    %c90464_i64 = arith.constant 90464 : i64
    %c91488_i64 = arith.constant 91488 : i64
    %c107872_i64 = arith.constant 107872 : i64
    %c124256_i64 = arith.constant 124256 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c0_i64 = arith.constant 0 : i64
    %c36864_i64 = arith.constant 36864 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c69632_i64 = arith.constant 69632 : i64
    %c69696_i64 = arith.constant 69696 : i64
    %0 = hivm.hir.pointer_cast(%c69632_i64) : memref<16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c69696_i64) : memref<16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c4096_i64) : memref<512x16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c36864_i64) : memref<512x16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<512x1xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c2048_i64) : memref<512x1xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg6, %arg7 : i32
    %7 = arith.muli %6, %arg8 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg8, %arg7 : i32
    %11 = hivm.hir.pointer_cast(%c69760_i64) : memref<512xf32, #hivm.address_space<ub>>
    %cast = memref.cast %11 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %12 = hivm.hir.pointer_cast(%c71808_i64) : memref<16xf32, #hivm.address_space<ub>>
    %cast_1 = memref.cast %12 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast_1) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %13 = hivm.hir.pointer_cast(%c71872_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast_2 = memref.cast %13 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_2, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %14 = hivm.hir.pointer_cast(%c71936_i64) : memref<512xi32, #hivm.address_space<ub>>
    %cast_3 = memref.cast %14 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_3, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg9 = %9 to %7 step %c48_i32  : i32 {
      %15 = arith.index_cast %arg9 : i32 to index
      %16 = arith.index_cast %9 : i32 to index
      %17 = affine.apply #map()[%15, %16]
      %18 = arith.index_cast %17 : index to i64
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      %20 = arith.select %19, %1, %0 : memref<16xf32, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg9 : i32 to index
      %22 = arith.index_cast %9 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i1
      %25 = arith.xori %24, %true : i1
      %26 = arith.extui %25 : i1 to i64
      hivm.hir.set_mask_norm
      %27 = arith.divsi %arg9, %10 : i32
      %28 = arith.remsi %27, %arg6 : i32
      %29 = arith.muli %28, %c16_i32 : i32
      %30 = hivm.hir.pointer_cast(%c73984_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %31 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_4 = memref.cast %13 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %30 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %31 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_4, %29, %cast_5, %cast_6) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %32 = hivm.hir.pointer_cast(%c74048_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %33 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_7 = memref.cast %30 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %32 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_9 = memref.cast %33 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_7, %c512_i32, %cast_8, %cast_9) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %34 = hivm.hir.pointer_cast(%c74048_i64) : memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_10 = memref.cast %32 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %30 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %34 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_10, %cast_11, %cast_12) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %35 = hivm.hir.pointer_cast(%c74048_i64) : memref<16xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %36 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_13 = memref.cast %34 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %35 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %36 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_13, %cast_14, %cast_15) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %37 = hivm.hir.pointer_cast(%c73984_i64) : memref<16xf16, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c74112_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_16 = memref.cast %35 : memref<16xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %37 : memref<16xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %38 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_half_1d_with_temp(%cast_16, %cast_17, %cast_18, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %expand_shape = memref.expand_shape %37 [[0, 1]] output_shape [1, 16] : memref<16xf16, #hivm.address_space<ub>> into memref<1x16xf16, #hivm.address_space<ub>>
      %39 = hivm.hir.pointer_cast(%c74016_i64) : memref<512x16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_19 = memref.cast %expand_shape : memref<1x16xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %39 : memref<512x16xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_half(%cast_19, %cast_20) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %40 = hivm.hir.pointer_cast(%c74016_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %39 [[0, 1]] : memref<512x16xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
      %collapse_shape_21 = memref.collapse_shape %40 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_22 = memref.cast %collapse_shape : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %collapse_shape_21 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmps_eq_1d_half(%cast_22, %cst, %cast_23) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %41 = hivm.hir.pointer_cast(%c74016_i64) : memref<512x16xi1, #hivm.address_space<ub>>
      %collapse_shape_24 = memref.collapse_shape %41 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %42 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_25 = memref.cast %collapse_shape_21 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %collapse_shape_24 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %42 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_25, %cast_26, %cast_27) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %43 = arith.index_cast %29 : i32 to index
      %44 = affine.max #map1()[%43]
      %45 = affine.min #map2()[%44, %43]
      %46 = affine.min #map3()[%45, %43]
      %47 = affine.max #map4()[%46]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %26]
      %cast_28 = memref.cast %12 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_29 = memref.cast %20 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_28, %cast_29, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %48 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %20) -> (memref<16xf32, #hivm.address_space<ub>>)  : i32 {
        %50 = arith.index_cast %arg10 : i32 to index
        %51 = affine.apply #map5()[%50]
        %52 = arith.index_cast %51 : index to i64
        %53 = arith.cmpi eq, %52, %c1_i64 : i64
        %54 = arith.select %53, %3, %2 : memref<512x16xf32, #hivm.address_space<ub>>
        %55 = arith.index_cast %arg10 : i32 to index
        %56 = affine.apply #map5()[%55]
        %57 = arith.index_cast %56 : index to i64
        %58 = arith.cmpi eq, %57, %c1_i64 : i64
        %59 = arith.select %58, %5, %4 : memref<512x1xf32, #hivm.address_space<ub>>
        %60 = arith.index_cast %arg10 : i32 to index
        %61 = affine.apply #map5()[%60]
        %62 = arith.index_cast %61 : index to i1
        %63 = arith.select %62, %c1_i64, %c2_i64 : i64
        %64 = arith.muli %arg10, %c512_i32 : i32
        %65 = hivm.hir.pointer_cast(%c90400_i64) : memref<512xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %66 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_33 = memref.cast %14 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_34 = memref.cast %65 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_35 = memref.cast %66 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_int32_t(%cast_33, %64, %cast_34, %cast_35) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %67 = hivm.hir.pointer_cast(%c92448_i64) : memref<512xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %68 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_36 = memref.cast %65 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_37 = memref.cast %67 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_38 = memref.cast %68 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmaxs_vs_1d_int32_t(%cast_36, %c1024_i32, %cast_37, %cast_38) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %69 = hivm.hir.pointer_cast(%c92448_i64) : memref<512xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_39 = memref.cast %67 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %65 : memref<512xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %69 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_int32_t(%cast_39, %cast_40, %cast_41) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %70 = hivm.hir.pointer_cast(%c92448_i64) : memref<512xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %71 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_42 = memref.cast %69 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_43 = memref.cast %70 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %71 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_42, %cast_43, %cast_44) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %72 = arith.index_cast %64 : i32 to index
        %reinterpret_cast_45 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [512, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<512x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
        %73 = affine.max #map6()[%72]
        %74 = affine.min #map7()[%73, %72]
        %75 = affine.apply #map8()[%74, %72]
        %76 = arith.cmpi slt, %75, %c512 : index
        %subview_46 = memref.subview %reinterpret_cast_45[0, 0] [%75, 1] [1, 1] : memref<512x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_47 = memref.subview %59[0, 0] [%75, 1] [1, 1] : memref<512x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.if %76 {
          %collapse_shape_122 = memref.collapse_shape %59 [[0, 1]] : memref<512x1xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
          %cast_123 = memref.cast %collapse_shape_122 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_123) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %collapse_shape_48 = memref.collapse_shape %subview_46 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_49 = memref.collapse_shape %subview_47 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %cast_50 = memref.cast %collapse_shape_48 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_51 = memref.cast %collapse_shape_49 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_50, %cast_51, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_52 = memref.collapse_shape %59 [[0, 1]] : memref<512x1xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        %77 = hivm.hir.pointer_cast(%c90400_i64) : memref<512xi1, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_53 = memref.cast %collapse_shape_52 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_54 = memref.cast %11 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_55 = memref.cast %77 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_53, %cast_54, %cast_55) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %78 = hivm.hir.pointer_cast(%c90400_i64) : memref<512xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %79 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_56 = memref.cast %77 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_57 = memref.cast %78 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_58 = memref.cast %79 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_56, %cast_57, %cast_58) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %80 = affine.apply #map9()[%43, %72]
        %reinterpret_cast_59 = memref.reinterpret_cast %arg4 to offset: [%80], sizes: [512, 16], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<512x16xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %81 = hivm.hir.pointer_cast(%c90464_i64) : memref<512xf16, #hivm.address_space<ub>>
        %82 = hivm.hir.pointer_cast(%c91488_i64) : memref<48xf16, #hivm.address_space<ub>>
        %cast_60 = memref.cast %70 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_61 = memref.cast %81 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_62 = memref.cast %82 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bool_to_half_1d_with_temp(%cast_60, %cast_61, %cast_62, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %expand_shape_63 = memref.expand_shape %81 [[0, 1]] output_shape [512, 1] : memref<512xf16, #hivm.address_space<ub>> into memref<512x1xf16, #hivm.address_space<ub>>
        %83 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xf16, #hivm.address_space<ub>>
        %84 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_64 = memref.cast %expand_shape_63 : memref<512x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_65 = memref.cast %83 : memref<512x16xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %84 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_half(%cast_64, %cast_65, %cast_66) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %85 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xi1, #hivm.address_space<ub>>
        %collapse_shape_67 = memref.collapse_shape %83 [[0, 1]] : memref<512x16xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
        %collapse_shape_68 = memref.collapse_shape %85 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_69 = memref.cast %collapse_shape_67 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %collapse_shape_68 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmps_eq_1d_half(%cast_69, %cst, %cast_70) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %86 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xi1, #hivm.address_space<ub>>
        %collapse_shape_71 = memref.collapse_shape %86 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %87 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_72 = memref.cast %collapse_shape_68 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_73 = memref.cast %collapse_shape_71 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_74 = memref.cast %87 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_72, %cast_73, %cast_74) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %88 = hivm.hir.pointer_cast(%c90464_i64) : memref<512xf16, #hivm.address_space<ub>>
        %89 = hivm.hir.pointer_cast(%c107872_i64) : memref<48xf16, #hivm.address_space<ub>>
        %cast_75 = memref.cast %78 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_76 = memref.cast %88 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_77 = memref.cast %89 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_bool_to_half_1d_with_temp(%cast_75, %cast_76, %cast_77, %c4_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %expand_shape_78 = memref.expand_shape %88 [[0, 1]] output_shape [512, 1] : memref<512xf16, #hivm.address_space<ub>> into memref<512x1xf16, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xf16, #hivm.address_space<ub>>
        %91 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_79 = memref.cast %expand_shape_78 : memref<512x1xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_80 = memref.cast %90 : memref<512x16xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_81 = memref.cast %91 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_last_axis_align_2d_half(%cast_79, %cast_80, %cast_81) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %92 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xi1, #hivm.address_space<ub>>
        %collapse_shape_82 = memref.collapse_shape %90 [[0, 1]] : memref<512x16xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
        %collapse_shape_83 = memref.collapse_shape %92 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_84 = memref.cast %collapse_shape_82 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_85 = memref.cast %collapse_shape_83 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmps_eq_1d_half(%cast_84, %cst, %cast_85) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %93 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xi1, #hivm.address_space<ub>>
        %collapse_shape_86 = memref.collapse_shape %93 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %94 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_87 = memref.cast %collapse_shape_83 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_88 = memref.cast %collapse_shape_86 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_89 = memref.cast %94 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_87, %cast_88, %cast_89) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %95 = hivm.hir.pointer_cast(%c91488_i64) : memref<512x16xi1, #hivm.address_space<ub>>
        %collapse_shape_90 = memref.collapse_shape %95 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        %96 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_91 = memref.cast %collapse_shape_71 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_92 = memref.cast %collapse_shape_24 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_93 = memref.cast %collapse_shape_90 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_94 = memref.cast %96 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vand_1d_bool(%cast_91, %cast_92, %cast_93, %cast_94) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %97 = affine.min #map10()[%74, %72]
        %98 = affine.max #map4()[%97]
        %subview_95 = memref.subview %reinterpret_cast_59[0, 0] [%98, %47] [1, 1] : memref<512x16xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_96 = memref.subview %54[0, 0] [%98, %47] [1, 1] : memref<512x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %cast_97 = memref.cast %subview_95 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_98 = memref.cast %subview_96 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_97, %cast_98, %c0_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %99 = hivm.hir.pointer_cast(%c107872_i64) : memref<512x16xi1, #hivm.address_space<ub>>
        %collapse_shape_99 = memref.collapse_shape %99 [[0, 1]] : memref<512x16xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %100 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_100 = memref.cast %collapse_shape_90 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_101 = memref.cast %collapse_shape_86 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_102 = memref.cast %collapse_shape_99 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_103 = memref.cast %100 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vand_1d_bool(%cast_100, %cast_101, %cast_102, %cast_103) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %101 = hivm.hir.pointer_cast(%c124256_i64) : memref<512x16xf32, #hivm.address_space<ub>>
        %collapse_shape_104 = memref.collapse_shape %54 [[0, 1]] : memref<512x16xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %collapse_shape_105 = memref.collapse_shape %101 [[0, 1]] : memref<512x16xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c90400_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_106 = memref.cast %collapse_shape_99 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_107 = memref.cast %collapse_shape_104 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_108 = memref.cast %collapse_shape_105 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_109 = memref.cast %102 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vs_1d_float(%cast_106, %cast_107, %cst_0, %cast_108, %cast_109) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %103 = hivm.hir.pointer_cast(%c124256_i64) : memref<512x16xf32, #hivm.address_space<ub>>
        %104 = hivm.hir.pointer_cast(%c90400_i64) : memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_110 = memref.cast %59 : memref<512x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_111 = memref.cast %101 : memref<512x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_112 = memref.cast %103 : memref<512x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_113 = memref.cast %104 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_110, %cast_111, %cast_112, %cast_113) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %105 = hivm.hir.pointer_cast(%c90400_i64) : memref<1x16xf32, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c90464_i64) : memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_114 = memref.cast %103 : memref<512x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_115 = memref.cast %105 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_116 = memref.cast %106 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_114, %cast_115, %cast_116, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_117 = memref.collapse_shape %105 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %107 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_118 = memref.cast %arg11 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_119 = memref.cast %collapse_shape_117 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_120 = memref.cast %20 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_121 = memref.cast %107 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_118, %cast_119, %cast_120, %cast_121) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %20 : memref<16xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%43], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %49 = affine.apply #map8()[%45, %43]
      %subview = memref.subview %48[0] [%49] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_30 = memref.subview %reinterpret_cast[0] [%49] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_31 = memref.cast %subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %subview_30 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_31, %cast_32, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %26]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
