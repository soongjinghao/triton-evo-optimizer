#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<()[s0] -> (16, s0)>
#map2 = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
#map5 = affine_map<()[s0, s1] -> (s1, s0)>
#map6 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map7 = affine_map<()[s0, s1] -> (s0 - s1)>
#map8 = affine_map<()[s0, s1] -> (64, s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_selective_scan_update_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_le_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_selective_scan_update_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32, %arg39: i32, %arg40: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<41xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %cst_0 = arith.constant 2.000000e+01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c5152_i64 = arith.constant 5152 : i64
    %c5408_i64 = arith.constant 5408 : i64
    %c5664_i64 = arith.constant 5664 : i64
    %c9792_i64 = arith.constant 9792 : i64
    %c16576_i64 = arith.constant 16576 : i64
    %c18624_i64 = arith.constant 18624 : i64
    %c20672_i64 = arith.constant 20672 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c21184_i64 = arith.constant 21184 : i64
    %c256_i64 = arith.constant 256 : i64
    %c25280_i64 = arith.constant 25280 : i64
    %c4352_i64 = arith.constant 4352 : i64
    %c25536_i64 = arith.constant 25536 : i64
    %c4608_i64 = arith.constant 4608 : i64
    %c25792_i64 = arith.constant 25792 : i64
    %c4864_i64 = arith.constant 4864 : i64
    %c26048_i64 = arith.constant 26048 : i64
    %c5696_i64 = arith.constant 5696 : i64
    %c30144_i64 = arith.constant 30144 : i64
    %c11840_i64 = arith.constant 11840 : i64
    %c30208_i64 = arith.constant 30208 : i64
    %c11904_i64 = arith.constant 11904 : i64
    %c30272_i64 = arith.constant 30272 : i64
    %c11968_i64 = arith.constant 11968 : i64
    %c30528_i64 = arith.constant 30528 : i64
    %c12224_i64 = arith.constant 12224 : i64
    %c30784_i64 = arith.constant 30784 : i64
    %c12480_i64 = arith.constant 12480 : i64
    %c20928_i64 = arith.constant 20928 : i64
    %c34880_i64 = arith.constant 34880 : i64
    %0 = hivm.hir.pointer_cast(%c20928_i64) : memref<64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c34880_i64) : memref<64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c12480_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c30784_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c12224_i64) : memref<64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c30528_i64) : memref<64xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c11968_i64) : memref<64xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c30272_i64) : memref<64xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c11904_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c30208_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c11840_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c30144_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c5696_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c26048_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c4864_i64) : memref<64xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c25792_i64) : memref<64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c4608_i64) : memref<64xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c25536_i64) : memref<64xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c4352_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c25280_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c256_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c21184_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %22 = arith.muli %arg38, %arg39 : i32
    %23 = arith.muli %22, %arg40 : i32
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.trunci %24 : i64 to i32
    %26 = arith.muli %arg40, %arg39 : i32
    %27 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
    %cast = memref.cast %27 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %28 = arith.index_cast %arg21 : i32 to index
    %29 = arith.index_cast %arg28 : i32 to index
    %30 = arith.index_cast %arg16 : i32 to index
    %31 = arith.index_cast %arg17 : i32 to index
    %32 = affine.max #map()[%31]
    %33 = affine.min #map1()[%32]
    %34 = affine.max #map()[%33]
    %35 = arith.cmpi slt, %34, %c16 : index
    %36 = arith.cmpi slt, %33, %c16 : index
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg41 = %25 to %23 step %c48_i32  : i32 {
      %37 = arith.index_cast %arg41 : i32 to index
      %38 = arith.index_cast %25 : i32 to index
      %39 = affine.apply #map2()[%37, %38]
      %40 = arith.index_cast %39 : index to i64
      %41 = arith.cmpi eq, %40, %c1_i64 : i64
      %42 = arith.select %41, %1, %0 : memref<64xf32, #hivm.address_space<ub>>
      %43 = arith.index_cast %arg41 : i32 to index
      %44 = arith.index_cast %25 : i32 to index
      %45 = affine.apply #map2()[%43, %44]
      %46 = arith.index_cast %45 : index to i64
      %47 = arith.cmpi eq, %46, %c1_i64 : i64
      %48 = arith.select %47, %3, %2 : memref<64x16xf32, #hivm.address_space<ub>>
      %49 = arith.index_cast %arg41 : i32 to index
      %50 = arith.index_cast %25 : i32 to index
      %51 = affine.apply #map2()[%49, %50]
      %52 = arith.index_cast %51 : index to i64
      %53 = arith.cmpi eq, %52, %c1_i64 : i64
      %54 = arith.select %53, %5, %4 : memref<64xf32, #hivm.address_space<ub>>
      %55 = arith.index_cast %arg41 : i32 to index
      %56 = arith.index_cast %25 : i32 to index
      %57 = affine.apply #map2()[%55, %56]
      %58 = arith.index_cast %57 : index to i64
      %59 = arith.cmpi eq, %58, %c1_i64 : i64
      %60 = arith.select %59, %7, %6 : memref<64xf32, #hivm.address_space<ub>>
      %61 = arith.index_cast %arg41 : i32 to index
      %62 = arith.index_cast %25 : i32 to index
      %63 = affine.apply #map2()[%61, %62]
      %64 = arith.index_cast %63 : index to i64
      %65 = arith.cmpi eq, %64, %c1_i64 : i64
      %66 = arith.select %65, %9, %8 : memref<1x16xf32, #hivm.address_space<ub>>
      %67 = arith.index_cast %arg41 : i32 to index
      %68 = arith.index_cast %25 : i32 to index
      %69 = affine.apply #map2()[%67, %68]
      %70 = arith.index_cast %69 : index to i64
      %71 = arith.cmpi eq, %70, %c1_i64 : i64
      %72 = arith.select %71, %11, %10 : memref<1x16xf32, #hivm.address_space<ub>>
      %73 = arith.index_cast %arg41 : i32 to index
      %74 = arith.index_cast %25 : i32 to index
      %75 = affine.apply #map2()[%73, %74]
      %76 = arith.index_cast %75 : index to i64
      %77 = arith.cmpi eq, %76, %c1_i64 : i64
      %78 = arith.select %77, %13, %12 : memref<64x16xf32, #hivm.address_space<ub>>
      %79 = arith.index_cast %arg41 : i32 to index
      %80 = arith.index_cast %25 : i32 to index
      %81 = affine.apply #map2()[%79, %80]
      %82 = arith.index_cast %81 : index to i64
      %83 = arith.cmpi eq, %82, %c1_i64 : i64
      %84 = arith.select %83, %15, %14 : memref<64xf32, #hivm.address_space<ub>>
      %85 = arith.index_cast %arg41 : i32 to index
      %86 = arith.index_cast %25 : i32 to index
      %87 = affine.apply #map2()[%85, %86]
      %88 = arith.index_cast %87 : index to i64
      %89 = arith.cmpi eq, %88, %c1_i64 : i64
      %90 = arith.select %89, %17, %16 : memref<64xf32, #hivm.address_space<ub>>
      %91 = arith.index_cast %arg41 : i32 to index
      %92 = arith.index_cast %25 : i32 to index
      %93 = affine.apply #map2()[%91, %92]
      %94 = arith.index_cast %93 : index to i64
      %95 = arith.cmpi eq, %94, %c1_i64 : i64
      %96 = arith.select %95, %19, %18 : memref<64x1xf32, #hivm.address_space<ub>>
      %97 = arith.index_cast %arg41 : i32 to index
      %98 = arith.index_cast %25 : i32 to index
      %99 = affine.apply #map2()[%97, %98]
      %100 = arith.index_cast %99 : index to i64
      %101 = arith.cmpi eq, %100, %c1_i64 : i64
      %102 = arith.select %101, %21, %20 : memref<64x16xf32, #hivm.address_space<ub>>
      %103 = arith.index_cast %arg41 : i32 to index
      %104 = arith.index_cast %25 : i32 to index
      %105 = affine.apply #map2()[%103, %104]
      %106 = arith.index_cast %105 : index to i1
      %107 = arith.select %106, %c3_i64, %c4_i64 : i64
      %108 = arith.index_cast %105 : index to i1
      %109 = arith.select %108, %c1_i64, %c2_i64 : i64
      hivm.hir.set_mask_norm
      %110 = arith.remsi %arg41, %arg40 : i32
      %111 = arith.divsi %arg41, %arg40 : i32
      %112 = arith.remsi %111, %arg39 : i32
      %113 = arith.divsi %arg41, %26 : i32
      %114 = arith.remsi %113, %arg38 : i32
      %115 = arith.muli %112, %arg19 : i32
      %116 = arith.muli %110, %arg20 : i32
      %117 = arith.addi %115, %116 : i32
      %118 = arith.muli %112, %arg22 : i32
      %119 = arith.muli %110, %arg23 : i32
      %120 = arith.addi %118, %119 : i32
      %121 = arith.muli %112, %arg24 : i32
      %122 = arith.muli %110, %arg25 : i32
      %123 = arith.addi %121, %122 : i32
      %124 = arith.muli %110, %arg26 : i32
      %125 = arith.muli %110, %arg27 : i32
      %126 = arith.muli %112, %arg29 : i32
      %127 = arith.divsi %110, %arg18 : i32
      %128 = arith.muli %127, %arg30 : i32
      %129 = arith.addi %126, %128 : i32
      %130 = arith.muli %112, %arg31 : i32
      %131 = arith.muli %127, %arg32 : i32
      %132 = arith.addi %130, %131 : i32
      %133 = arith.muli %112, %arg34 : i32
      %134 = arith.muli %110, %arg35 : i32
      %135 = arith.addi %133, %134 : i32
      %136 = arith.muli %112, %arg36 : i32
      %137 = arith.muli %110, %arg37 : i32
      %138 = arith.addi %136, %137 : i32
      %139 = arith.muli %114, %c64_i32 : i32
      %140 = arith.index_cast %117 : i32 to index
      %141 = arith.index_cast %139 : i32 to index
      %142 = affine.apply #map3()[%140, %141, %28]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%142], sizes: [64, 16], strides: [%28, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %143 = arith.index_cast %120 : i32 to index
      %144 = affine.apply #map4()[%143, %141]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%144], sizes: [64, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %145 = arith.index_cast %123 : i32 to index
      %146 = affine.apply #map4()[%145, %141]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%146], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %147 = arith.index_cast %124 : i32 to index
      %148 = affine.apply #map4()[%147, %141]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%148], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %149 = arith.muli %110, %arg33 : i32
      %150 = arith.index_cast %125 : i32 to index
      %151 = affine.apply #map3()[%150, %141, %29]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%151], sizes: [64, 16], strides: [%29, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %152 = arith.index_cast %129 : i32 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%152], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %153 = arith.index_cast %132 : i32 to index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg9 to offset: [%153], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %154 = arith.index_cast %149 : i32 to index
      %155 = affine.apply #map4()[%154, %141]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%155], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %156 = arith.index_cast %135 : i32 to index
      %157 = affine.apply #map4()[%156, %141]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg11 to offset: [%157], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %158 = arith.index_cast %138 : i32 to index
      %159 = affine.apply #map4()[%158, %141]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg12 to offset: [%159], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %160 = affine.max #map5()[%141, %30]
      %161 = affine.min #map6()[%160, %141]
      %162 = affine.apply #map7()[%161, %141]
      %163 = affine.min #map8()[%161, %141]
      %164 = affine.max #map()[%163]
      %165 = arith.cmpi slt, %164, %c64 : index
      %166 = arith.ori %165, %35 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%164, %34] [1, 1] : memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %102[0, 0] [%164, %34] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %166 {
        %collapse_shape_146 = memref.collapse_shape %102 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_13 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %subview_12 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_13, %cast_14, %c2_i32, %cst_2, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %167 = arith.cmpi slt, %162, %c64 : index
      %subview_15 = memref.subview %reinterpret_cast_3[0, 0] [%162, 1] [1, 1] : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_16 = memref.subview %96[0, 0] [%162, 1] [1, 1] : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %167 {
        %collapse_shape_146 = memref.collapse_shape %96 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_15 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %109]
      %cast_18 = memref.cast %collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_19 = memref.cast %collapse_shape_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_18, %cast_19, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %collapse_shape_20 = memref.collapse_shape %96 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      %subview_21 = memref.subview %reinterpret_cast_4[0] [%162] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_22 = memref.subview %90[0] [%162] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %167 {
        %cast_146 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_23 = memref.cast %subview_21 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_24 = memref.cast %subview_22 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_23, %cast_24, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %subview_25 = memref.subview %reinterpret_cast_5[0] [%162] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_26 = memref.subview %84[0] [%162] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %167 {
        %cast_146 = memref.cast %84 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_27 = memref.cast %subview_25 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_28 = memref.cast %subview_26 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_27, %cast_28, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %168 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_29 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %84 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %168 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_29, %cast_30, %cast_31, %cast_32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %169 = hivm.hir.pointer_cast(%c5120_i64) : memref<64xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_33 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %27 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %169 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_le_1d_float(%cast_33, %cast_34, %cast_35) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %170 = hivm.hir.pointer_cast(%c5152_i64) : memref<64xf32, #hivm.address_space<ub>>
      %171 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_36 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %170 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %171 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_36, %cast_37, %cast_38) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %172 = hivm.hir.pointer_cast(%c5152_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %173 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_39 = memref.cast %170 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %172 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %173 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_39, %cst_1, %cast_40, %cast_41) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %174 = hivm.hir.pointer_cast(%c5152_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %175 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_42 = memref.cast %172 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %174 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %175 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vln_1d_float(%cast_42, %cast_43, %cast_44) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %176 = hivm.hir.pointer_cast(%c5408_i64) : memref<64xf32, #hivm.address_space<ub>>
      %177 = hivm.hir.pointer_cast(%c5664_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_45 = memref.cast %169 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %174 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_47 = memref.cast %90 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %176 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %177 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_45, %cast_46, %cast_47, %cast_48, %cast_49) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_50 = memref.subview %reinterpret_cast_6[0, 0] [%164, %34] [1, 1] : memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_51 = memref.subview %78[0, 0] [%164, %34] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %166 {
        %collapse_shape_146 = memref.collapse_shape %78 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_52 = memref.cast %subview_50 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_53 = memref.cast %subview_51 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_52, %cast_53, %c2_i32, %cst_2, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %expand_shape = memref.expand_shape %176 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
      %178 = hivm.hir.pointer_cast(%c9792_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_54 = memref.cast %78 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %expand_shape : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %78 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_57 = memref.cast %178 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_54, %cast_55, %cast_56, %cast_57) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_58 = memref.collapse_shape %78 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_59 = memref.collapse_shape %78 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %179 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_60 = memref.cast %collapse_shape_58 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %collapse_shape_59 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_62 = memref.cast %179 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_60, %cast_61, %cast_62) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_63 = memref.subview %reinterpret_cast_7[0, 0] [1, %33] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_64 = memref.subview %72[0, 0] [1, %33] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %36 {
        %collapse_shape_146 = memref.collapse_shape %72 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_65 = memref.collapse_shape %subview_63 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_66 = memref.collapse_shape %subview_64 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_67 = memref.cast %collapse_shape_65 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_68 = memref.cast %collapse_shape_66 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_67, %cast_68, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
      %subview_69 = memref.subview %reinterpret_cast_8[0, 0] [1, %33] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_70 = memref.subview %66[0, 0] [1, %33] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %36 {
        %collapse_shape_146 = memref.collapse_shape %66 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_71 = memref.collapse_shape %subview_69 [[0, 1]] : memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_72 = memref.collapse_shape %subview_70 [[0, 1]] : memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_73 = memref.cast %collapse_shape_71 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_74 = memref.cast %collapse_shape_72 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_73, %cast_74, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
      %subview_75 = memref.subview %reinterpret_cast_9[0] [%162] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_76 = memref.subview %60[0] [%162] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %167 {
        %cast_146 = memref.cast %60 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %107]
      %cast_77 = memref.cast %subview_75 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_78 = memref.cast %subview_76 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_77, %cast_78, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %subview_79 = memref.subview %reinterpret_cast_10[0] [%162] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_80 = memref.subview %54[0] [%162] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %167 {
        %cast_146 = memref.cast %54 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_81 = memref.cast %subview_79 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_82 = memref.cast %subview_80 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_81, %cast_82, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %180 = hivm.hir.pointer_cast(%c16576_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
      %cast_83 = memref.cast %72 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_84 = memref.cast %expand_shape : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %48 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_86 = memref.cast %180 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_83, %cast_84, %cast_85, %cast_86) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_87 = memref.collapse_shape %102 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_88 = memref.collapse_shape %78 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %181 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_89 = memref.cast %collapse_shape_87 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_90 = memref.cast %collapse_shape_59 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_91 = memref.cast %collapse_shape_88 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_92 = memref.cast %181 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_89, %cast_90, %cast_91, %cast_92) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %182 = hivm.hir.pointer_cast(%c18624_i64) : memref<512xf32, #hivm.address_space<ub>>
      %cast_93 = memref.cast %48 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_94 = memref.cast %96 : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_95 = memref.cast %48 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_96 = memref.cast %182 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_93, %cast_94, %cast_95, %cast_96) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_97 = memref.collapse_shape %48 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_98 = memref.collapse_shape %48 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %183 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_99 = memref.cast %collapse_shape_88 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_100 = memref.cast %collapse_shape_97 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_101 = memref.cast %collapse_shape_98 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_102 = memref.cast %183 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_99, %cast_100, %cast_101, %cast_102) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_103 = memref.subview %48[0, 0] [%164, %34] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_104 = memref.cast %subview_103 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_105 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_104, %cast_105, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %184 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_106 = memref.cast %48 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_107 = memref.cast %66 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_108 = memref.cast %48 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_109 = memref.cast %184 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_106, %cast_107, %cast_108, %cast_109) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %185 = hivm.hir.pointer_cast(%c20672_i64) : memref<64x1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %186 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_110 = memref.cast %48 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_111 = memref.cast %185 : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_112 = memref.cast %186 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_sum_ar_float(%cast_110, %cast_111, %cast_112, %cst_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_113 = memref.collapse_shape %185 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %187 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_114 = memref.cast %collapse_shape_20 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_115 = memref.cast %60 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_116 = memref.cast %60 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_117 = memref.cast %187 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_114, %cast_115, %cast_116, %cast_117) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %109]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %188 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_118 = memref.cast %collapse_shape_113 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_119 = memref.cast %60 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_120 = memref.cast %60 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_121 = memref.cast %188 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_118, %cast_119, %cast_120, %cast_121) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %189 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_122 = memref.cast %54 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_123 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_124 = memref.cast %189 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_122, %cst, %cast_123, %cast_124) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %190 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_125 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_126 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_127 = memref.cast %190 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_125, %cst_2, %cast_126, %cast_127) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %191 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_128 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_129 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_130 = memref.cast %191 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_128, %cast_129, %cast_130) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %192 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_131 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_132 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_133 = memref.cast %192 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_131, %cst_1, %cast_132, %cast_133) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %193 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_134 = memref.cast %54 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_135 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_136 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_137 = memref.cast %193 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_1d_float(%cast_134, %cast_135, %cast_136, %cast_137) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_138 = memref.cast %60 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_139 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_140 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_141 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_138, %cast_139, %cast_140, %cast_141) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %107]
      %subview_142 = memref.subview %42[0] [%162] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_143 = memref.subview %reinterpret_cast_11[0] [%162] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_144 = memref.cast %subview_142 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_145 = memref.cast %subview_143 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_144, %cast_145, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
