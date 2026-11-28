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
  func.func @_selective_scan_update_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32, %arg37: i32, %arg38: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<39xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %cst_0 = arith.constant 2.000000e+01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c5152_i64 = arith.constant 5152 : i64
    %c5408_i64 = arith.constant 5408 : i64
    %c5664_i64 = arith.constant 5664 : i64
    %c9792_i64 = arith.constant 9792 : i64
    %c12480_i64 = arith.constant 12480 : i64
    %c16576_i64 = arith.constant 16576 : i64
    %c18624_i64 = arith.constant 18624 : i64
    %c20928_i64 = arith.constant 20928 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
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
    %c21184_i64 = arith.constant 21184 : i64
    %c256_i64 = arith.constant 256 : i64
    %c20672_i64 = arith.constant 20672 : i64
    %c30784_i64 = arith.constant 30784 : i64
    %0 = hivm.hir.pointer_cast(%c20672_i64) : memref<64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c30784_i64) : memref<64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c256_i64) : memref<64x16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c21184_i64) : memref<64x16xf32, #hivm.address_space<ub>>
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
    %22 = hivm.hir.pointer_cast(%c20672_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c30784_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c4352_i64) : memref<64xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c25280_i64) : memref<64xf32, #hivm.address_space<ub>>
    %26 = arith.muli %arg36, %arg37 : i32
    %27 = arith.muli %26, %arg38 : i32
    %28 = hivm.hir.get_block_idx -> i64
    %29 = arith.trunci %28 : i64 to i32
    %30 = arith.muli %arg38, %arg37 : i32
    %31 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
    %cast = memref.cast %31 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %32 = arith.index_cast %arg19 : i32 to index
    %33 = arith.index_cast %arg26 : i32 to index
    %34 = arith.index_cast %arg15 : i32 to index
    %35 = arith.index_cast %arg16 : i32 to index
    %36 = affine.max #map()[%35]
    %37 = affine.min #map1()[%36]
    %38 = affine.max #map()[%37]
    %39 = arith.cmpi slt, %38, %c16 : index
    %40 = arith.cmpi slt, %37, %c16 : index
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg39 = %29 to %27 step %c48_i32  : i32 {
      %41 = arith.index_cast %arg39 : i32 to index
      %42 = arith.index_cast %29 : i32 to index
      %43 = affine.apply #map2()[%41, %42]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.cmpi eq, %44, %c1_i64 : i64
      %46 = arith.select %45, %1, %0 : memref<64xf32, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg39 : i32 to index
      %48 = arith.index_cast %29 : i32 to index
      %49 = affine.apply #map2()[%47, %48]
      %50 = arith.index_cast %49 : index to i64
      %51 = arith.cmpi eq, %50, %c1_i64 : i64
      %52 = arith.select %51, %3, %2 : memref<64x16xf32, #hivm.address_space<ub>>
      %53 = arith.index_cast %arg39 : i32 to index
      %54 = arith.index_cast %29 : i32 to index
      %55 = affine.apply #map2()[%53, %54]
      %56 = arith.index_cast %55 : index to i64
      %57 = arith.cmpi eq, %56, %c1_i64 : i64
      %58 = arith.select %57, %5, %4 : memref<64xf32, #hivm.address_space<ub>>
      %59 = arith.index_cast %arg39 : i32 to index
      %60 = arith.index_cast %29 : i32 to index
      %61 = affine.apply #map2()[%59, %60]
      %62 = arith.index_cast %61 : index to i64
      %63 = arith.cmpi eq, %62, %c1_i64 : i64
      %64 = arith.select %63, %7, %6 : memref<64xf32, #hivm.address_space<ub>>
      %65 = arith.index_cast %arg39 : i32 to index
      %66 = arith.index_cast %29 : i32 to index
      %67 = affine.apply #map2()[%65, %66]
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.cmpi eq, %68, %c1_i64 : i64
      %70 = arith.select %69, %9, %8 : memref<1x16xf32, #hivm.address_space<ub>>
      %71 = arith.index_cast %arg39 : i32 to index
      %72 = arith.index_cast %29 : i32 to index
      %73 = affine.apply #map2()[%71, %72]
      %74 = arith.index_cast %73 : index to i64
      %75 = arith.cmpi eq, %74, %c1_i64 : i64
      %76 = arith.select %75, %11, %10 : memref<1x16xf32, #hivm.address_space<ub>>
      %77 = arith.index_cast %arg39 : i32 to index
      %78 = arith.index_cast %29 : i32 to index
      %79 = affine.apply #map2()[%77, %78]
      %80 = arith.index_cast %79 : index to i64
      %81 = arith.cmpi eq, %80, %c1_i64 : i64
      %82 = arith.select %81, %13, %12 : memref<64x16xf32, #hivm.address_space<ub>>
      %83 = arith.index_cast %arg39 : i32 to index
      %84 = arith.index_cast %29 : i32 to index
      %85 = affine.apply #map2()[%83, %84]
      %86 = arith.index_cast %85 : index to i64
      %87 = arith.cmpi eq, %86, %c1_i64 : i64
      %88 = arith.select %87, %15, %14 : memref<64xf32, #hivm.address_space<ub>>
      %89 = arith.index_cast %arg39 : i32 to index
      %90 = arith.index_cast %29 : i32 to index
      %91 = affine.apply #map2()[%89, %90]
      %92 = arith.index_cast %91 : index to i64
      %93 = arith.cmpi eq, %92, %c1_i64 : i64
      %94 = arith.select %93, %17, %16 : memref<64xf32, #hivm.address_space<ub>>
      %95 = arith.index_cast %arg39 : i32 to index
      %96 = arith.index_cast %29 : i32 to index
      %97 = affine.apply #map2()[%95, %96]
      %98 = arith.index_cast %97 : index to i64
      %99 = arith.cmpi eq, %98, %c1_i64 : i64
      %100 = arith.select %99, %19, %18 : memref<64x1xf32, #hivm.address_space<ub>>
      %101 = arith.index_cast %arg39 : i32 to index
      %102 = arith.index_cast %29 : i32 to index
      %103 = affine.apply #map2()[%101, %102]
      %104 = arith.index_cast %103 : index to i64
      %105 = arith.cmpi eq, %104, %c1_i64 : i64
      %106 = arith.select %105, %21, %20 : memref<64x16xf32, #hivm.address_space<ub>>
      %107 = arith.index_cast %arg39 : i32 to index
      %108 = arith.index_cast %29 : i32 to index
      %109 = affine.apply #map2()[%107, %108]
      %110 = arith.index_cast %109 : index to i64
      %111 = arith.cmpi eq, %110, %c1_i64 : i64
      %112 = arith.select %111, %23, %22 : memref<64x1xf32, #hivm.address_space<ub>>
      %113 = arith.index_cast %arg39 : i32 to index
      %114 = arith.index_cast %29 : i32 to index
      %115 = affine.apply #map2()[%113, %114]
      %116 = arith.index_cast %115 : index to i64
      %117 = arith.cmpi eq, %116, %c1_i64 : i64
      %118 = arith.select %117, %25, %24 : memref<64xf32, #hivm.address_space<ub>>
      %119 = arith.index_cast %arg39 : i32 to index
      %120 = arith.index_cast %29 : i32 to index
      %121 = affine.apply #map2()[%119, %120]
      %122 = arith.index_cast %121 : index to i1
      %123 = arith.select %122, %c5_i64, %c6_i64 : i64
      %124 = arith.index_cast %121 : index to i1
      %125 = arith.select %124, %c3_i64, %c4_i64 : i64
      %126 = arith.index_cast %121 : index to i1
      %127 = arith.select %126, %c1_i64, %c2_i64 : i64
      hivm.hir.set_mask_norm
      %128 = arith.divsi %arg39, %arg38 : i32
      %129 = arith.remsi %128, %arg37 : i32
      %130 = arith.divsi %arg39, %30 : i32
      %131 = arith.remsi %130, %arg36 : i32
      %132 = arith.divsi %129, %c4_i32 : i32
      %133 = arith.remsi %129, %c4_i32 : i32
      %134 = arith.muli %132, %arg17 : i32
      %135 = arith.muli %133, %arg18 : i32
      %136 = arith.addi %134, %135 : i32
      %137 = arith.muli %132, %arg20 : i32
      %138 = arith.muli %133, %arg21 : i32
      %139 = arith.addi %137, %138 : i32
      %140 = arith.muli %132, %arg22 : i32
      %141 = arith.muli %133, %arg23 : i32
      %142 = arith.addi %140, %141 : i32
      %143 = arith.muli %133, %arg24 : i32
      %144 = arith.muli %133, %arg25 : i32
      %145 = arith.muli %132, %arg27 : i32
      %146 = arith.divsi %133, %c2_i32 : i32
      %147 = arith.muli %146, %arg28 : i32
      %148 = arith.addi %145, %147 : i32
      %149 = arith.muli %132, %arg29 : i32
      %150 = arith.muli %146, %arg30 : i32
      %151 = arith.addi %149, %150 : i32
      %152 = arith.muli %132, %arg32 : i32
      %153 = arith.muli %133, %arg33 : i32
      %154 = arith.addi %152, %153 : i32
      %155 = arith.muli %132, %arg34 : i32
      %156 = arith.muli %133, %arg35 : i32
      %157 = arith.addi %155, %156 : i32
      %158 = arith.muli %131, %c64_i32 : i32
      %159 = arith.index_cast %136 : i32 to index
      %160 = arith.index_cast %158 : i32 to index
      %161 = affine.apply #map3()[%159, %160, %32]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%161], sizes: [64, 16], strides: [%32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %162 = arith.index_cast %139 : i32 to index
      %163 = affine.apply #map4()[%162, %160]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%163], sizes: [64, 1], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %164 = arith.index_cast %142 : i32 to index
      %165 = affine.apply #map4()[%164, %160]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%165], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %166 = arith.index_cast %143 : i32 to index
      %167 = affine.apply #map4()[%166, %160]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%167], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %168 = arith.muli %133, %arg31 : i32
      %169 = arith.index_cast %144 : i32 to index
      %170 = affine.apply #map3()[%169, %160, %33]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%170], sizes: [64, 16], strides: [%33, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %171 = arith.index_cast %148 : i32 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%171], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %172 = arith.index_cast %151 : i32 to index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg9 to offset: [%172], sizes: [1, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %173 = arith.index_cast %168 : i32 to index
      %174 = affine.apply #map4()[%173, %160]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg10 to offset: [%174], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %175 = arith.index_cast %154 : i32 to index
      %176 = affine.apply #map4()[%175, %160]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg11 to offset: [%176], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %177 = arith.index_cast %157 : i32 to index
      %178 = affine.apply #map4()[%177, %160]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg12 to offset: [%178], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %179 = affine.max #map5()[%160, %34]
      %180 = affine.min #map6()[%179, %160]
      %181 = affine.apply #map7()[%180, %160]
      %182 = affine.min #map8()[%180, %160]
      %183 = affine.max #map()[%182]
      %184 = arith.cmpi slt, %183, %c64 : index
      %185 = arith.ori %184, %39 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%183, %38] [1, 1] : memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %106[0, 0] [%183, %38] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %185 {
        %collapse_shape_146 = memref.collapse_shape %106 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %127]
      %cast_13 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %subview_12 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_13, %cast_14, %c2_i32, %cst_2, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %186 = arith.cmpi slt, %181, %c64 : index
      %subview_15 = memref.subview %reinterpret_cast_3[0, 0] [%181, 1] [1, 1] : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_16 = memref.subview %100[0, 0] [%181, 1] [1, 1] : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      scf.if %186 {
        %collapse_shape_146 = memref.collapse_shape %100 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_15 [[0, 1]] : memref<?x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1]] : memref<?x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %125]
      %cast_18 = memref.cast %collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_19 = memref.cast %collapse_shape_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_18, %cast_19, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %collapse_shape_20 = memref.collapse_shape %100 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      %subview_21 = memref.subview %reinterpret_cast_4[0] [%181] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_22 = memref.subview %94[0] [%181] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %186 {
        %cast_146 = memref.cast %94 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_23 = memref.cast %subview_21 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_24 = memref.cast %subview_22 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_23, %cast_24, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %subview_25 = memref.subview %reinterpret_cast_5[0] [%181] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_26 = memref.subview %88[0] [%181] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %186 {
        %cast_146 = memref.cast %88 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_27 = memref.cast %subview_25 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_28 = memref.cast %subview_26 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_27, %cast_28, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %187 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_29 = memref.cast %94 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %88 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %88 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %187 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_29, %cast_30, %cast_31, %cast_32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %188 = hivm.hir.pointer_cast(%c5120_i64) : memref<64xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_33 = memref.cast %88 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %31 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %188 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_le_1d_float(%cast_33, %cast_34, %cast_35) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %189 = hivm.hir.pointer_cast(%c5152_i64) : memref<64xf32, #hivm.address_space<ub>>
      %190 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_36 = memref.cast %88 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %189 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %190 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_36, %cast_37, %cast_38) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %191 = hivm.hir.pointer_cast(%c5152_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %192 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_39 = memref.cast %189 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %191 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %192 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_39, %cst_1, %cast_40, %cast_41) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %193 = hivm.hir.pointer_cast(%c5152_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %194 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_42 = memref.cast %191 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %193 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %194 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vln_1d_float(%cast_42, %cast_43, %cast_44) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %195 = hivm.hir.pointer_cast(%c5408_i64) : memref<64xf32, #hivm.address_space<ub>>
      %196 = hivm.hir.pointer_cast(%c5664_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_45 = memref.cast %188 : memref<64xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %193 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_47 = memref.cast %88 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %195 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %196 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_45, %cast_46, %cast_47, %cast_48, %cast_49) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_50 = memref.subview %reinterpret_cast_6[0, 0] [%183, %38] [1, 1] : memref<64x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_51 = memref.subview %82[0, 0] [%183, %38] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %185 {
        %collapse_shape_146 = memref.collapse_shape %82 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_146 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_147) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_52 = memref.cast %subview_50 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_53 = memref.cast %subview_51 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_52, %cast_53, %c2_i32, %cst_2, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %expand_shape = memref.expand_shape %195 [[0, 1]] output_shape [64, 1] : memref<64xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
      %197 = hivm.hir.pointer_cast(%c9792_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_54 = memref.cast %82 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_55 = memref.cast %expand_shape : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %82 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_57 = memref.cast %197 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_54, %cast_55, %cast_56, %cast_57) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_58 = memref.collapse_shape %82 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_59 = memref.collapse_shape %82 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %198 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_60 = memref.cast %collapse_shape_58 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %collapse_shape_59 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_62 = memref.cast %198 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_60, %cast_61, %cast_62) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %subview_63 = memref.subview %reinterpret_cast_7[0, 0] [1, %37] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_64 = memref.subview %76[0, 0] [1, %37] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %40 {
        %collapse_shape_146 = memref.collapse_shape %76 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
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
      %subview_69 = memref.subview %reinterpret_cast_8[0, 0] [1, %37] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_70 = memref.subview %70[0, 0] [1, %37] [1, 1] : memref<1x16xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %40 {
        %collapse_shape_146 = memref.collapse_shape %70 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
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
      %subview_75 = memref.subview %reinterpret_cast_9[0] [%181] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_76 = memref.subview %64[0] [%181] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %186 {
        %cast_146 = memref.cast %64 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_77 = memref.cast %subview_75 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_78 = memref.cast %subview_76 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_77, %cast_78, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %subview_79 = memref.subview %reinterpret_cast_10[0] [%181] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_80 = memref.subview %58[0] [%181] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %186 {
        %cast_146 = memref.cast %58 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_146) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %123]
      %cast_81 = memref.cast %subview_79 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_82 = memref.cast %subview_80 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_81, %cast_82, %c2_i32, %cst_2, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %199 = hivm.hir.pointer_cast(%c12480_i64) : memref<64x16xf32, #hivm.address_space<ub>>
      %200 = hivm.hir.pointer_cast(%c16576_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
      %cast_83 = memref.cast %76 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_84 = memref.cast %expand_shape : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %199 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_86 = memref.cast %200 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_83, %cast_84, %cast_85, %cast_86) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_87 = memref.collapse_shape %106 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_88 = memref.collapse_shape %106 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %201 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_89 = memref.cast %collapse_shape_87 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_90 = memref.cast %collapse_shape_59 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_91 = memref.cast %collapse_shape_88 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_92 = memref.cast %201 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_89, %cast_90, %cast_91, %cast_92) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %202 = hivm.hir.pointer_cast(%c12480_i64) : memref<64x16xf32, #hivm.address_space<ub>>
      %203 = hivm.hir.pointer_cast(%c18624_i64) : memref<512xf32, #hivm.address_space<ub>>
      %cast_93 = memref.cast %199 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_94 = memref.cast %100 : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_95 = memref.cast %202 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_96 = memref.cast %203 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_93, %cast_94, %cast_95, %cast_96) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %collapse_shape_97 = memref.collapse_shape %202 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_98 = memref.collapse_shape %52 [[0, 1]] : memref<64x16xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %204 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_99 = memref.cast %collapse_shape_88 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_100 = memref.cast %collapse_shape_97 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_101 = memref.cast %collapse_shape_98 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_102 = memref.cast %204 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_99, %cast_100, %cast_101, %cast_102) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_103 = memref.subview %52[0, 0] [%183, %38] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_104 = memref.cast %subview_103 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_105 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_104, %cast_105, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %205 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_106 = memref.cast %52 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_107 = memref.cast %70 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_108 = memref.cast %106 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_109 = memref.cast %205 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_106, %cast_107, %cast_108, %cast_109) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %206 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_110 = memref.cast %106 : memref<64x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_111 = memref.cast %112 : memref<64x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_112 = memref.cast %206 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_sum_ar_float(%cast_110, %cast_111, %cast_112, %cst_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %127]
      %collapse_shape_113 = memref.collapse_shape %112 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %207 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_114 = memref.cast %collapse_shape_20 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_115 = memref.cast %64 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_116 = memref.cast %118 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_117 = memref.cast %207 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_114, %cast_115, %cast_116, %cast_117) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %208 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_118 = memref.cast %collapse_shape_113 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_119 = memref.cast %118 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_120 = memref.cast %46 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_121 = memref.cast %208 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_118, %cast_119, %cast_120, %cast_121) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %125]
      %209 = hivm.hir.pointer_cast(%c20928_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %210 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_122 = memref.cast %58 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_123 = memref.cast %209 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_124 = memref.cast %210 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_122, %cst, %cast_123, %cast_124) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %211 = hivm.hir.pointer_cast(%c20928_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %212 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_125 = memref.cast %209 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_126 = memref.cast %211 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_127 = memref.cast %212 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_125, %cst_2, %cast_126, %cast_127) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %213 = hivm.hir.pointer_cast(%c20928_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %214 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_128 = memref.cast %211 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_129 = memref.cast %213 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_130 = memref.cast %214 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_128, %cast_129, %cast_130) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %215 = hivm.hir.pointer_cast(%c20928_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %216 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_131 = memref.cast %213 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_132 = memref.cast %215 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_133 = memref.cast %216 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_131, %cst_1, %cast_132, %cast_133) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %217 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_134 = memref.cast %58 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_135 = memref.cast %215 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_136 = memref.cast %58 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_137 = memref.cast %217 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_1d_float(%cast_134, %cast_135, %cast_136, %cast_137) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %218 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_138 = memref.cast %46 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_139 = memref.cast %58 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_140 = memref.cast %46 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_141 = memref.cast %218 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_138, %cast_139, %cast_140, %cast_141) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %123]
      %subview_142 = memref.subview %46[0] [%181] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_143 = memref.subview %reinterpret_cast_11[0] [%181] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
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
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
