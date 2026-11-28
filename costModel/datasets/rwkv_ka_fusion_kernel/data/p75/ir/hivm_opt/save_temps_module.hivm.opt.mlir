#map = affine_map<()[s0, s1, s2, s3] -> ((s0 + ((s2 - s3) floordiv 48) * s1) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 8, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @rwkv_ka_fusion_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @rwkv_ka_fusion_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 9.99999996E-13 : f32
    %cst_2 = arith.constant -1.000000e+00 : f32
    %c128_i64 = arith.constant 128 : i64
    %c160_i64 = arith.constant 160 : i64
    %c224_i64 = arith.constant 224 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i64 = arith.constant 0 : i64
    %c288_i64 = arith.constant 288 : i64
    %c32_i64 = arith.constant 32 : i64
    %c352_i64 = arith.constant 352 : i64
    %c96_i64 = arith.constant 96 : i64
    %c320_i64 = arith.constant 320 : i64
    %c64_i64 = arith.constant 64 : i64
    %c192_i64 = arith.constant 192 : i64
    %c384_i64 = arith.constant 384 : i64
    %0 = hivm.hir.pointer_cast(%c192_i64) : memref<8xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c384_i64) : memref<8xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c64_i64) : memref<8xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c320_i64) : memref<8xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c192_i64) : memref<8xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c384_i64) : memref<8xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c96_i64) : memref<8xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c352_i64) : memref<8xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c64_i64) : memref<8xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c320_i64) : memref<8xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c32_i64) : memref<8xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c288_i64) : memref<8xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c0_i64) : memref<8xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c256_i64) : memref<8xf32, #hivm.address_space<ub>>
    %14 = arith.muli %arg14, %arg15 : i32
    %15 = arith.muli %14, %arg16 : i32
    %16 = hivm.hir.get_block_idx -> i64
    %17 = arith.trunci %16 : i64 to i32
    %18 = arith.muli %arg16, %arg15 : i32
    %19 = arith.muli %arg10, %arg11 : i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = arith.index_cast %arg11 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg17 = %17 to %15 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %22 = arith.divsi %arg17, %18 : i32
      %23 = arith.remsi %22, %arg14 : i32
      %24 = arith.muli %23, %c16_i32 : i32
      scf.for %arg18 = %c0_i32 to %arg12 step %c1_i32  : i32 {
        %25 = arith.index_cast %arg18 : i32 to index
        %26 = arith.index_cast %arg12 : i32 to index
        %27 = arith.index_cast %arg17 : i32 to index
        %28 = arith.index_cast %17 : i32 to index
        %29 = affine.apply #map()[%25, %26, %27, %28]
        %30 = arith.index_cast %29 : index to i64
        %31 = arith.cmpi eq, %30, %c1_i64 : i64
        %32 = arith.select %31, %1, %0 : memref<8xf32, #hivm.address_space<ub>>
        %33 = arith.index_cast %arg18 : i32 to index
        %34 = arith.index_cast %arg12 : i32 to index
        %35 = arith.index_cast %arg17 : i32 to index
        %36 = arith.index_cast %17 : i32 to index
        %37 = affine.apply #map()[%33, %34, %35, %36]
        %38 = arith.index_cast %37 : index to i64
        %39 = arith.cmpi eq, %38, %c1_i64 : i64
        %40 = arith.select %39, %3, %2 : memref<8xf32, #hivm.address_space<ub>>
        %41 = arith.index_cast %arg18 : i32 to index
        %42 = arith.index_cast %arg12 : i32 to index
        %43 = arith.index_cast %arg17 : i32 to index
        %44 = arith.index_cast %17 : i32 to index
        %45 = affine.apply #map()[%41, %42, %43, %44]
        %46 = arith.index_cast %45 : index to i64
        %47 = arith.cmpi eq, %46, %c1_i64 : i64
        %48 = arith.select %47, %5, %4 : memref<8xf32, #hivm.address_space<ub>>
        %49 = arith.index_cast %arg18 : i32 to index
        %50 = arith.index_cast %arg12 : i32 to index
        %51 = arith.index_cast %arg17 : i32 to index
        %52 = arith.index_cast %17 : i32 to index
        %53 = affine.apply #map()[%49, %50, %51, %52]
        %54 = arith.index_cast %53 : index to i64
        %55 = arith.cmpi eq, %54, %c1_i64 : i64
        %56 = arith.select %55, %7, %6 : memref<8xf32, #hivm.address_space<ub>>
        %57 = arith.index_cast %arg18 : i32 to index
        %58 = arith.index_cast %arg12 : i32 to index
        %59 = arith.index_cast %arg17 : i32 to index
        %60 = arith.index_cast %17 : i32 to index
        %61 = affine.apply #map()[%57, %58, %59, %60]
        %62 = arith.index_cast %61 : index to i64
        %63 = arith.cmpi eq, %62, %c1_i64 : i64
        %64 = arith.select %63, %9, %8 : memref<8xf32, #hivm.address_space<ub>>
        %65 = arith.index_cast %arg18 : i32 to index
        %66 = arith.index_cast %arg12 : i32 to index
        %67 = arith.index_cast %arg17 : i32 to index
        %68 = arith.index_cast %17 : i32 to index
        %69 = affine.apply #map()[%65, %66, %67, %68]
        %70 = arith.index_cast %69 : index to i64
        %71 = arith.cmpi eq, %70, %c1_i64 : i64
        %72 = arith.select %71, %11, %10 : memref<8xf32, #hivm.address_space<ub>>
        %73 = arith.index_cast %arg18 : i32 to index
        %74 = arith.index_cast %arg12 : i32 to index
        %75 = arith.index_cast %arg17 : i32 to index
        %76 = arith.index_cast %17 : i32 to index
        %77 = affine.apply #map()[%73, %74, %75, %76]
        %78 = arith.index_cast %77 : index to i64
        %79 = arith.cmpi eq, %78, %c1_i64 : i64
        %80 = arith.select %79, %13, %12 : memref<8xf32, #hivm.address_space<ub>>
        %81 = arith.index_cast %arg18 : i32 to index
        %82 = arith.index_cast %arg12 : i32 to index
        %83 = arith.index_cast %arg17 : i32 to index
        %84 = arith.index_cast %17 : i32 to index
        %85 = affine.apply #map()[%81, %82, %83, %84]
        %86 = arith.index_cast %85 : index to i1
        %87 = arith.xori %86, %true : i1
        %88 = arith.extui %87 : i1 to i64
        %89 = arith.index_cast %85 : index to i1
        %90 = arith.select %89, %c3_i64, %c4_i64 : i64
        %91 = arith.index_cast %85 : index to i1
        %92 = arith.select %91, %c1_i64, %c2_i64 : i64
        %93 = arith.muli %arg18, %arg13 : i32
        %94 = arith.addi %24, %93 : i32
        %95 = arith.index_cast %94 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %96 = affine.max #map1()[%95, %20]
        %97 = affine.min #map2()[%96, %95]
        %98 = affine.apply #map3()[%97, %95]
        %99 = arith.cmpi slt, %98, %c8 : index
        %subview = memref.subview %reinterpret_cast[0] [%98] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %80[0] [%98] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %99 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_67 = memref.cast %80 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %92]
        %cast = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast, %cast_4, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%95], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %reinterpret_cast_5[0] [%98] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %72[0] [%98] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %99 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_67 = memref.cast %72 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %90]
        %cast_8 = memref.cast %subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_9 = memref.cast %subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_8, %cast_9, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %100 = arith.index_cast %93 : i32 to index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%100], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %101 = affine.max #map1()[%100, %21]
        %102 = affine.min #map2()[%101, %100]
        %103 = affine.apply #map3()[%102, %100]
        %104 = arith.cmpi slt, %103, %c8 : index
        %subview_11 = memref.subview %reinterpret_cast_10[0] [%103] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_12 = memref.subview %64[0] [%103] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %104 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_67 = memref.cast %64 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %88]
        %cast_13 = memref.cast %subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_14 = memref.cast %subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_13, %cast_14, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %reinterpret_cast_15 = memref.reinterpret_cast %arg4 to offset: [%100], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_16 = memref.subview %reinterpret_cast_15[0] [%103] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_17 = memref.subview %56[0] [%103] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %104 {
          %cast_67 = memref.cast %56 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %cast_18 = memref.cast %subview_16 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_19 = memref.cast %subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_18, %cast_19, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %105 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_20 = memref.cast %80 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_21 = memref.cast %56 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_22 = memref.cast %56 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_23 = memref.cast %105 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_1d_float(%cast_20, %cast_21, %cast_22, %cast_23) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %106 = hivm.hir.pointer_cast(%c128_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %107 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_24 = memref.cast %56 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %56 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_26 = memref.cast %106 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_27 = memref.cast %107 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_1d_float(%cast_24, %cast_25, %cast_26, %cast_27) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %108 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %109 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_28 = memref.cast %106 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_29 = memref.cast %108 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %109 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_sum_r_float(%cast_28, %cast_29, %cast_30, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %110 = memref.load %108[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %111 = arith.addf %110, %cst_1 : f32
        %112 = math.sqrt %111 : f32
        %113 = arith.divf %cst_0, %112 : f32
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %92]
        %114 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_31 = memref.cast %56 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_32 = memref.cast %48 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_33 = memref.cast %114 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_31, %113, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_34 = memref.reinterpret_cast %arg8 to offset: [%95], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_35 = memref.subview %48[0] [%98] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_36 = memref.subview %reinterpret_cast_34[0] [%98] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_37 = memref.cast %subview_35 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_38 = memref.cast %subview_36 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_37, %cast_38, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %115 = hivm.hir.pointer_cast(%c224_i64) : memref<8xf32, #hivm.address_space<ub>>
        %116 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_39 = memref.cast %72 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %115 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %116 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_39, %cst_2, %cast_40, %cast_41) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %117 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_42 = memref.cast %115 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_43 = memref.cast %64 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %64 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_45 = memref.cast %117 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_1d_float(%cast_42, %cast_43, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %118 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_46 = memref.cast %64 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_47 = memref.cast %64 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_48 = memref.cast %118 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_46, %cst_0, %cast_47, %cast_48) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %119 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_49 = memref.cast %80 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_50 = memref.cast %64 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_51 = memref.cast %40 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_52 = memref.cast %119 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_1d_float(%cast_49, %cast_50, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %92]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %120 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_53 = memref.cast %48 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_54 = memref.cast %72 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_55 = memref.cast %32 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_56 = memref.cast %120 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_1d_float(%cast_53, %cast_54, %cast_55, %cast_56) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %90]
        %reinterpret_cast_57 = memref.reinterpret_cast %arg7 to offset: [%95], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_58 = memref.subview %40[0] [%98] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_59 = memref.subview %reinterpret_cast_57[0] [%98] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_60 = memref.cast %subview_58 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_61 = memref.cast %subview_59 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_60, %cast_61, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %88]
        %reinterpret_cast_62 = memref.reinterpret_cast %arg9 to offset: [%95], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_63 = memref.subview %32[0] [%98] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_64 = memref.subview %reinterpret_cast_62[0] [%98] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_65 = memref.cast %subview_63 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %subview_64 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_65, %cast_66, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %92]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
