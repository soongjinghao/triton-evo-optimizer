#map = affine_map<()[s0, s1, s2] -> ((s0 + s2 * s1) mod 2)>
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
  func.func private @vdiv_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
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
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 9.99999996E-13 : f32
    %cst_2 = arith.constant -1.000000e+00 : f32
    %c128_i64 = arith.constant 128 : i64
    %c160_i64 = arith.constant 160 : i64
    %c224_i64 = arith.constant 224 : i64
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c288_i64 = arith.constant 288 : i64
    %c0_i64 = arith.constant 0 : i64
    %c320_i64 = arith.constant 320 : i64
    %c32_i64 = arith.constant 32 : i64
    %c384_i64 = arith.constant 384 : i64
    %c96_i64 = arith.constant 96 : i64
    %c352_i64 = arith.constant 352 : i64
    %c64_i64 = arith.constant 64 : i64
    %c192_i64 = arith.constant 192 : i64
    %c416_i64 = arith.constant 416 : i64
    %0 = hivm.hir.pointer_cast(%c192_i64) : memref<8xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c416_i64) : memref<8xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c64_i64) : memref<8xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c352_i64) : memref<8xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c192_i64) : memref<8xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c416_i64) : memref<8xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c96_i64) : memref<8xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c384_i64) : memref<8xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c64_i64) : memref<8xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c352_i64) : memref<8xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c32_i64) : memref<8xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c320_i64) : memref<8xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c0_i64) : memref<8xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c288_i64) : memref<8xf32, #hivm.address_space<ub>>
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
      %24 = arith.muli %23, %c4_i32 : i32
      scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %25 = arith.addi %24, %arg18 : i32
        %26 = arith.muli %25, %c16_i32 : i32
        scf.for %arg19 = %c0_i32 to %arg12 step %c1_i32  : i32 {
          %27 = arith.index_cast %arg19 : i32 to index
          %28 = arith.index_cast %arg12 : i32 to index
          %29 = arith.index_cast %arg18 : i32 to index
          %30 = affine.apply #map()[%27, %28, %29]
          %31 = arith.index_cast %30 : index to i64
          %32 = arith.cmpi eq, %31, %c1_i64 : i64
          %33 = arith.select %32, %1, %0 : memref<8xf32, #hivm.address_space<ub>>
          %34 = arith.index_cast %arg19 : i32 to index
          %35 = arith.index_cast %arg12 : i32 to index
          %36 = arith.index_cast %arg18 : i32 to index
          %37 = affine.apply #map()[%34, %35, %36]
          %38 = arith.index_cast %37 : index to i64
          %39 = arith.cmpi eq, %38, %c1_i64 : i64
          %40 = arith.select %39, %3, %2 : memref<8xf32, #hivm.address_space<ub>>
          %41 = arith.index_cast %arg19 : i32 to index
          %42 = arith.index_cast %arg12 : i32 to index
          %43 = arith.index_cast %arg18 : i32 to index
          %44 = affine.apply #map()[%41, %42, %43]
          %45 = arith.index_cast %44 : index to i64
          %46 = arith.cmpi eq, %45, %c1_i64 : i64
          %47 = arith.select %46, %5, %4 : memref<8xf32, #hivm.address_space<ub>>
          %48 = arith.index_cast %arg19 : i32 to index
          %49 = arith.index_cast %arg12 : i32 to index
          %50 = arith.index_cast %arg18 : i32 to index
          %51 = affine.apply #map()[%48, %49, %50]
          %52 = arith.index_cast %51 : index to i64
          %53 = arith.cmpi eq, %52, %c1_i64 : i64
          %54 = arith.select %53, %7, %6 : memref<8xf32, #hivm.address_space<ub>>
          %55 = arith.index_cast %arg19 : i32 to index
          %56 = arith.index_cast %arg12 : i32 to index
          %57 = arith.index_cast %arg18 : i32 to index
          %58 = affine.apply #map()[%55, %56, %57]
          %59 = arith.index_cast %58 : index to i64
          %60 = arith.cmpi eq, %59, %c1_i64 : i64
          %61 = arith.select %60, %9, %8 : memref<8xf32, #hivm.address_space<ub>>
          %62 = arith.index_cast %arg19 : i32 to index
          %63 = arith.index_cast %arg12 : i32 to index
          %64 = arith.index_cast %arg18 : i32 to index
          %65 = affine.apply #map()[%62, %63, %64]
          %66 = arith.index_cast %65 : index to i64
          %67 = arith.cmpi eq, %66, %c1_i64 : i64
          %68 = arith.select %67, %11, %10 : memref<8xf32, #hivm.address_space<ub>>
          %69 = arith.index_cast %arg19 : i32 to index
          %70 = arith.index_cast %arg12 : i32 to index
          %71 = arith.index_cast %arg18 : i32 to index
          %72 = affine.apply #map()[%69, %70, %71]
          %73 = arith.index_cast %72 : index to i64
          %74 = arith.cmpi eq, %73, %c1_i64 : i64
          %75 = arith.select %74, %13, %12 : memref<8xf32, #hivm.address_space<ub>>
          %76 = arith.index_cast %arg19 : i32 to index
          %77 = arith.index_cast %arg12 : i32 to index
          %78 = arith.index_cast %arg18 : i32 to index
          %79 = affine.apply #map()[%76, %77, %78]
          %80 = arith.index_cast %79 : index to i1
          %81 = arith.xori %80, %true : i1
          %82 = arith.extui %81 : i1 to i64
          %83 = arith.index_cast %79 : index to i1
          %84 = arith.select %83, %c3_i64, %c4_i64 : i64
          %85 = arith.index_cast %79 : index to i1
          %86 = arith.select %85, %c1_i64, %c2_i64 : i64
          %87 = arith.muli %arg19, %arg13 : i32
          %88 = arith.addi %26, %87 : i32
          %89 = arith.index_cast %88 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%89], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %90 = affine.max #map1()[%89, %20]
          %91 = affine.min #map2()[%90, %89]
          %92 = affine.apply #map3()[%91, %89]
          %93 = arith.cmpi slt, %92, %c8 : index
          %subview = memref.subview %reinterpret_cast[0] [%92] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_3 = memref.subview %75[0] [%92] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %93 {
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_67 = memref.cast %75 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %86]
          %cast = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_4 = memref.cast %subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast, %cast_4, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%89], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_6 = memref.subview %reinterpret_cast_5[0] [%92] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %68[0] [%92] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %93 {
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_67 = memref.cast %68 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %84]
          %cast_8 = memref.cast %subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_9 = memref.cast %subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_8, %cast_9, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          %94 = arith.index_cast %87 : i32 to index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%94], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %95 = affine.max #map1()[%94, %21]
          %96 = affine.min #map2()[%95, %94]
          %97 = affine.apply #map3()[%96, %94]
          %98 = arith.cmpi slt, %97, %c8 : index
          %subview_11 = memref.subview %reinterpret_cast_10[0] [%97] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_12 = memref.subview %61[0] [%97] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %98 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_67 = memref.cast %61 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %82]
          %cast_13 = memref.cast %subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_14 = memref.cast %subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_13, %cast_14, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          %reinterpret_cast_15 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_16 = memref.subview %reinterpret_cast_15[0] [%97] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_17 = memref.subview %54[0] [%97] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %98 {
            %cast_67 = memref.cast %54 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_67) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_18 = memref.cast %subview_16 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_19 = memref.cast %subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_18, %cast_19, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %99 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_20 = memref.cast %75 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %54 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_22 = memref.cast %54 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_23 = memref.cast %99 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_1d_float(%cast_20, %cast_21, %cast_22, %cast_23) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %100 = hivm.hir.pointer_cast(%c128_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %101 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_24 = memref.cast %54 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_25 = memref.cast %54 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_26 = memref.cast %100 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %101 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_1d_float(%cast_24, %cast_25, %cast_26, %cast_27) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %102 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %103 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_28 = memref.cast %100 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %102 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %103 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_sum_r_float(%cast_28, %cast_29, %cast_30, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %104 = memref.load %102[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %105 = arith.addf %104, %cst_1 : f32
          %106 = math.sqrt %105 : f32
          %107 = hivm.hir.pointer_cast(%c224_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %86]
          %cast_31 = memref.cast %54 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %47 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %107 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vdiv_vs_1d_float(%cast_31, %106, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %reinterpret_cast_34 = memref.reinterpret_cast %arg8 to offset: [%89], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_35 = memref.subview %47[0] [%92] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_36 = memref.subview %reinterpret_cast_34[0] [%92] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_37 = memref.cast %subview_35 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_38 = memref.cast %subview_36 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_37, %cast_38, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %108 = hivm.hir.pointer_cast(%c256_i64) : memref<8xf32, #hivm.address_space<ub>>
          %109 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_39 = memref.cast %68 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_40 = memref.cast %108 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_41 = memref.cast %109 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vadds_vs_1d_float(%cast_39, %cst_2, %cast_40, %cast_41) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %110 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_42 = memref.cast %108 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_43 = memref.cast %61 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_44 = memref.cast %61 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_45 = memref.cast %110 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_1d_float(%cast_42, %cast_43, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %111 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_46 = memref.cast %61 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_47 = memref.cast %61 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_48 = memref.cast %111 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vadds_vs_1d_float(%cast_46, %cst_0, %cast_47, %cast_48) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_V>]
          %112 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_49 = memref.cast %75 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_50 = memref.cast %61 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_51 = memref.cast %40 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_52 = memref.cast %112 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_1d_float(%cast_49, %cast_50, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %86]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %113 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_53 = memref.cast %47 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_54 = memref.cast %68 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_55 = memref.cast %33 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_56 = memref.cast %113 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_1d_float(%cast_53, %cast_54, %cast_55, %cast_56) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %84]
          %reinterpret_cast_57 = memref.reinterpret_cast %arg7 to offset: [%89], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_58 = memref.subview %40[0] [%92] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %reinterpret_cast_57[0] [%92] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_60 = memref.cast %subview_58 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_61 = memref.cast %subview_59 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_60, %cast_61, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %82]
          %reinterpret_cast_62 = memref.reinterpret_cast %arg9 to offset: [%89], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_63 = memref.subview %33[0] [%92] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_64 = memref.subview %reinterpret_cast_62[0] [%92] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_65 = memref.cast %subview_63 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_66 = memref.cast %subview_64 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_65, %cast_66, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %86]
        } {pipeline.veconly}
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
