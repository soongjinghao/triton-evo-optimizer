#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_correct_attn_cp_out_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_2d_bool(memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vor_2d_bool(memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_sv_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_max_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsub_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsub_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vor_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c4_i64 = arith.constant 4 : i64
    %cst_1 = arith.constant 0x7F800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %c160_i64 = arith.constant 160 : i64
    %c320_i64 = arith.constant 320 : i64
    %c448_i64 = arith.constant 448 : i64
    %c576_i64 = arith.constant 576 : i64
    %c704_i64 = arith.constant 704 : i64
    %c768_i64 = arith.constant 768 : i64
    %c800_i64 = arith.constant 800 : i64
    %c928_i64 = arith.constant 928 : i64
    %c992_i64 = arith.constant 992 : i64
    %c1056_i64 = arith.constant 1056 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c1248_i64 = arith.constant 1248 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c1312_i64 = arith.constant 1312 : i64
    %c1344_i64 = arith.constant 1344 : i64
    %c3456_i64 = arith.constant 3456 : i64
    %c5_i64 = arith.constant 5 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3712_i64 = arith.constant 3712 : i64
    %c192_i64 = arith.constant 192 : i64
    %c3840_i64 = arith.constant 3840 : i64
    %c960_i64 = arith.constant 960 : i64
    %c3872_i64 = arith.constant 3872 : i64
    %c1216_i64 = arith.constant 1216 : i64
    %c3904_i64 = arith.constant 3904 : i64
    %c1408_i64 = arith.constant 1408 : i64
    %c2432_i64 = arith.constant 2432 : i64
    %c4928_i64 = arith.constant 4928 : i64
    %0 = hivm.hir.pointer_cast(%c2432_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c4928_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1408_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c3904_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c1216_i64) : memref<4xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c3872_i64) : memref<4xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c960_i64) : memref<4xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c3840_i64) : memref<4xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c192_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c3712_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %10 = arith.muli %arg12, %arg13 : i32
    %11 = arith.muli %10, %arg14 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg14, %arg13 : i32
    %15 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %15[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %cast = memref.cast %subview : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_2d(%cst_1, %cast) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %16 = hivm.hir.pointer_cast(%c128_i64) : memref<4xf32, #hivm.address_space<ub>>
    %cast_2 = memref.cast %16 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast_2) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %17 = hivm.hir.pointer_cast(%c160_i64) : memref<4xf32, #hivm.address_space<ub>>
    %cast_3 = memref.cast %17 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_1, %cast_3) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %18 = arith.extsi %arg10 : i32 to i64
    %19 = arith.index_cast %arg9 : i32 to index
    %20 = arith.extsi %arg7 : i32 to i64
    %21 = arith.muli %arg11, %arg9 : i32
    %22 = arith.extsi %21 : i32 to i64
    %23 = arith.index_cast %arg8 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    scf.for %arg15 = %13 to %11 step %c48_i32  : i32 {
      %24 = arith.index_cast %arg15 : i32 to index
      %25 = arith.index_cast %13 : i32 to index
      %26 = affine.apply #map()[%24, %25]
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.cmpi eq, %27, %c1_i64 : i64
      %29 = arith.select %28, %1, %0 : memref<4x64xf32, #hivm.address_space<ub>>
      %30 = arith.index_cast %arg15 : i32 to index
      %31 = arith.index_cast %13 : i32 to index
      %32 = affine.apply #map()[%30, %31]
      %33 = arith.index_cast %32 : index to i64
      %34 = arith.cmpi eq, %33, %c1_i64 : i64
      %35 = arith.select %34, %3, %2 : memref<4x64xf32, #hivm.address_space<ub>>
      %36 = arith.index_cast %arg15 : i32 to index
      %37 = arith.index_cast %13 : i32 to index
      %38 = affine.apply #map()[%36, %37]
      %39 = arith.index_cast %38 : index to i64
      %40 = arith.cmpi eq, %39, %c1_i64 : i64
      %41 = arith.select %40, %5, %4 : memref<4xf32, #hivm.address_space<ub>>
      %42 = arith.index_cast %arg15 : i32 to index
      %43 = arith.index_cast %13 : i32 to index
      %44 = affine.apply #map()[%42, %43]
      %45 = arith.index_cast %44 : index to i64
      %46 = arith.cmpi eq, %45, %c1_i64 : i64
      %47 = arith.select %46, %7, %6 : memref<4xf32, #hivm.address_space<ub>>
      %48 = arith.index_cast %arg15 : i32 to index
      %49 = arith.index_cast %13 : i32 to index
      %50 = affine.apply #map()[%48, %49]
      %51 = arith.index_cast %50 : index to i64
      %52 = arith.cmpi eq, %51, %c1_i64 : i64
      %53 = arith.select %52, %9, %8 : memref<4x8x1xf32, #hivm.address_space<ub>>
      %54 = arith.index_cast %arg15 : i32 to index
      %55 = arith.index_cast %13 : i32 to index
      %56 = affine.apply #map()[%54, %55]
      %57 = arith.index_cast %56 : index to i1
      %58 = arith.select %57, %c4_i64, %c5_i64 : i64
      %59 = arith.index_cast %56 : index to i1
      %60 = arith.select %59, %c2_i64, %c3_i64 : i64
      %61 = arith.index_cast %56 : index to i1
      %62 = arith.xori %61, %true : i1
      %63 = arith.extui %62 : i1 to i64
      hivm.hir.set_mask_norm
      %64 = arith.divsi %arg15, %arg14 : i32
      %65 = arith.remsi %64, %arg13 : i32
      %66 = arith.divsi %arg15, %14 : i32
      %67 = arith.remsi %66, %arg12 : i32
      %68 = arith.extsi %67 : i32 to i64
      %69 = arith.extsi %65 : i32 to i64
      %70 = arith.muli %69, %c4_i64 : i64
      %71 = arith.muli %68, %18 : i64
      %72 = arith.index_cast %71 : i64 to index
      %73 = arith.index_cast %70 : i64 to index
      %74 = affine.apply #map1()[%72, %73]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%74], sizes: [4, 4], strides: [%19, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x4xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %53[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %63]
      %cast_5 = memref.cast %reinterpret_cast : memref<4x4xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_6 = memref.cast %subview_4 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_5, %cast_6, %c0_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %75 = hivm.hir.pointer_cast(%c320_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
      %subview_7 = memref.subview %75[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      scf.for %arg16 = %c0 to %c4 step %c1 {
        %subview_90 = memref.subview %subview_4[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_91 = memref.subview %subview_4[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_92 = memref.subview %subview_7[%arg16, 0] [1, 4] [1, 1] : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<4xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %cast_93 = memref.cast %subview_90 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_94 = memref.cast %subview_91 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_95 = memref.cast %subview_92 : memref<4xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_93, %cast_94, %cast_95) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      }
      %76 = hivm.hir.pointer_cast(%c320_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
      %subview_8 = memref.subview %76[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %77 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_9 = memref.cast %subview_7 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %subview_8 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %77 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_2d_bool(%cast_9, %cast_10, %cast_11) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %78 = hivm.hir.pointer_cast(%c448_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
      %subview_12 = memref.subview %78[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.for %arg16 = %c0 to %c4 step %c1 {
        %subview_90 = memref.subview %subview_4[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_91 = memref.subview %subview[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_92 = memref.subview %subview_12[%arg16, 0] [1, 4] [1, 1] : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<4xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %cast_93 = memref.cast %subview_90 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_94 = memref.cast %subview_91 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_95 = memref.cast %subview_92 : memref<4xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_93, %cast_94, %cast_95) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      }
      %79 = hivm.hir.pointer_cast(%c320_i64) : memref<4x256x1xi1, #hivm.address_space<ub>>
      %subview_13 = memref.subview %79[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x256x1xi1, #hivm.address_space<ub>> to memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %80 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_14 = memref.cast %subview_8 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %subview_12 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %subview_13 : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %80 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vor_2d_bool(%cast_14, %cast_15, %cast_16, %cast_17) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %81 = hivm.hir.pointer_cast(%c576_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
      %subview_18 = memref.subview %81[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %82 = hivm.hir.pointer_cast(%c704_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      scf.for %arg16 = %c0 to %c4 step %c1 {
        %subview_90 = memref.subview %subview_13[%arg16, 0] [1, 4] [1, 1] : memref<4x4xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<4xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_91 = memref.subview %subview_4[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_92 = memref.subview %subview_18[%arg16, 0] [1, 4] [1, 1] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %cast_93 = memref.cast %subview_90 : memref<4xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_94 = memref.cast %subview_91 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_95 = memref.cast %subview_92 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_96 = memref.cast %82 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_sv_1d_float(%cast_93, %cst_0, %cast_94, %cast_95, %cast_96) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      }
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %63]
      %83 = hivm.hir.pointer_cast(%c768_i64) : memref<1x4xf32, #hivm.address_space<ub>>
      %84 = hivm.hir.pointer_cast(%c800_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_19 = memref.cast %subview_18 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %83 : memref<1x4xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %84 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_max_ra_float(%cast_19, %cast_20, %cast_21, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape = memref.collapse_shape %83 [[0, 1]] : memref<1x4xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
      %85 = hivm.hir.pointer_cast(%c928_i64) : memref<4xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_22 = memref.cast %collapse_shape : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %16 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_24 = memref.cast %85 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_22, %cast_23, %cast_24) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %86 = hivm.hir.pointer_cast(%c992_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %63]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_25 = memref.cast %85 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %collapse_shape : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %47 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_28 = memref.cast %86 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_sv_1d_float(%cast_25, %cst, %cast_26, %cast_27, %cast_28) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape = memref.expand_shape %47 [[0, 1]] output_shape [1, 4] : memref<4xf32, #hivm.address_space<ub>> into memref<1x4xf32, #hivm.address_space<ub>>
      %87 = hivm.hir.pointer_cast(%c576_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
      %subview_29 = memref.subview %87[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %88 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_30 = memref.cast %subview_18 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %expand_shape : memref<1x4xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %subview_29 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_33 = memref.cast %88 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsub_2d_float(%cast_30, %cast_31, %cast_32, %cast_33) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %89 = hivm.hir.pointer_cast(%c576_i64) : memref<4x8x1xf32, #hivm.address_space<ub>>
      %subview_34 = memref.subview %89[0, 0, 0] [4, 4, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %90 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_35 = memref.cast %subview_29 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %subview_34 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %90 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_2d_float(%cast_35, %cast_36, %cast_37) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %91 = hivm.hir.pointer_cast(%c1056_i64) : memref<1x4xf32, #hivm.address_space<ub>>
      %92 = hivm.hir.pointer_cast(%c1088_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_38 = memref.cast %subview_34 : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_39 = memref.cast %91 : memref<1x4xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %92 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_38, %cast_39, %cast_40, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_41 = memref.collapse_shape %91 [[0, 1]] : memref<1x4xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
      %93 = hivm.hir.pointer_cast(%c1056_i64) : memref<4xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %94 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_42 = memref.cast %collapse_shape_41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %93 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %94 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vln_1d_float(%cast_42, %cast_43, %cast_44) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %95 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_45 = memref.cast %93 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %47 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_47 = memref.cast %47 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %95 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_45, %cast_46, %cast_47, %cast_48) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_49 = memref.reinterpret_cast %arg6 to offset: [%74], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_50 = memref.cast %47 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_51 = memref.cast %reinterpret_cast_49 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_50, %cast_51, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %63]
      %96 = arith.muli %68, %20 : i64
      %97 = arith.addi %22, %71 : i64
      %98 = arith.index_cast %97 : i64 to index
      %99 = affine.apply #map1()[%98, %73]
      %reinterpret_cast_52 = memref.reinterpret_cast %arg5 to offset: [%99], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %60]
      %cast_53 = memref.cast %reinterpret_cast_52 : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_54 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_53, %cast_54, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %100 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_55 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %47 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_57 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_58 = memref.cast %100 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsub_1d_float(%cast_55, %cast_56, %cast_57, %cast_58) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %101 = hivm.hir.pointer_cast(%c1248_i64) : memref<4xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_59 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_60 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %101 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_59, %cast_60, %cast_61) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %102 = hivm.hir.pointer_cast(%c1248_i64) : memref<4xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %103 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_62 = memref.cast %101 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_63 = memref.cast %102 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_64 = memref.cast %103 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_62, %cast_63, %cast_64) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %104 = hivm.hir.pointer_cast(%c1280_i64) : memref<4xi1, #hivm.address_space<ub>>
      %cast_65 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_66 = memref.cast %17 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_67 = memref.cast %104 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_65, %cast_66, %cast_67) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %105 = hivm.hir.pointer_cast(%c1248_i64) : memref<4xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %106 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_68 = memref.cast %102 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_69 = memref.cast %104 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_70 = memref.cast %105 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_71 = memref.cast %106 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vor_1d_bool(%cast_68, %cast_69, %cast_70, %cast_71) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %107 = hivm.hir.pointer_cast(%c1312_i64) : memref<4xf32, #hivm.address_space<ub>>
      %108 = hivm.hir.pointer_cast(%c1344_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_72 = memref.cast %105 : memref<4xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_73 = memref.cast %41 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_74 = memref.cast %107 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_75 = memref.cast %108 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_sv_1d_float(%cast_72, %cst_0, %cast_73, %cast_74, %cast_75) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %60]
      %109 = hivm.hir.pointer_cast(%c1312_i64) : memref<4xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %110 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_76 = memref.cast %107 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_77 = memref.cast %109 : memref<4xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_78 = memref.cast %110 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_76, %cast_77, %cast_78) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %111 = arith.index_cast %96 : i64 to index
      %112 = affine.apply #map2()[%111, %73, %23]
      %reinterpret_cast_79 = memref.reinterpret_cast %arg3 to offset: [%112], sizes: [4, 64], strides: [%23, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %58]
      %cast_80 = memref.cast %reinterpret_cast_79 : memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_81 = memref.cast %35 : memref<4x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_80, %cast_81, %c0_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %expand_shape_82 = memref.expand_shape %109 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
      %113 = hivm.hir.pointer_cast(%c3456_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %60]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_83 = memref.cast %35 : memref<4x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_84 = memref.cast %expand_shape_82 : memref<4x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %29 : memref<4x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_86 = memref.cast %113 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_83, %cast_84, %cast_85, %cast_86) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %58]
      %reinterpret_cast_87 = memref.reinterpret_cast %arg4 to offset: [%112], sizes: [4, 64], strides: [%23, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_88 = memref.cast %29 : memref<4x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_89 = memref.cast %reinterpret_cast_87 : memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_88, %cast_89, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %60]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
