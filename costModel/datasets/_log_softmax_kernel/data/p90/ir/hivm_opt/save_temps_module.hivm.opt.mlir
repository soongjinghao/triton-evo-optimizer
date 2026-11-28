#map = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 2048 + ((s2 - s3) floordiv 48) * ((s1 + 2047) floordiv 2048)) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1] -> (s1, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 2048, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_log_softmax_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_log_softmax_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c8224_i64 = arith.constant 8224 : i64
    %c8480_i64 = arith.constant 8480 : i64
    %c8512_i64 = arith.constant 8512 : i64
    %c8544_i64 = arith.constant 8544 : i64
    %c8576_i64 = arith.constant 8576 : i64
    %c8608_i64 = arith.constant 8608 : i64
    %c8640_i64 = arith.constant 8640 : i64
    %c8672_i64 = arith.constant 8672 : i64
    %c8704_i64 = arith.constant 8704 : i64
    %c8736_i64 = arith.constant 8736 : i64
    %c8768_i64 = arith.constant 8768 : i64
    %c8800_i64 = arith.constant 8800 : i64
    %c8832_i64 = arith.constant 8832 : i64
    %c12928_i64 = arith.constant 12928 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c37536_i64 = arith.constant 37536 : i64
    %c12960_i64 = arith.constant 12960 : i64
    %c45728_i64 = arith.constant 45728 : i64
    %c21152_i64 = arith.constant 21152 : i64
    %c0_i64 = arith.constant 0 : i64
    %c29344_i64 = arith.constant 29344 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c29344_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c21152_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c45728_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c12960_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c37536_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg8, %arg9 : i32
    %7 = arith.muli %6, %arg10 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg10, %arg9 : i32
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = arith.extsi %arg6 : i32 to i64
    %13 = arith.index_cast %arg7 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg11 = %9 to %7 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %14 = arith.divsi %arg11, %10 : i32
      %15 = arith.remsi %14, %arg8 : i32
      %16 = arith.extsi %15 : i32 to i64
      %17 = arith.muli %16, %11 : i64
      %18 = arith.muli %16, %12 : i64
      %19 = arith.index_cast %17 : i64 to index
      %20:2 = scf.for %arg12 = %c0_i32 to %arg7 step %c2048_i32 iter_args(%arg13 = %cst_0, %arg14 = %cst) -> (f32, f32)  : i32 {
        %27 = arith.index_cast %arg12 : i32 to index
        %28 = arith.index_cast %arg7 : i32 to index
        %29 = arith.index_cast %arg11 : i32 to index
        %30 = arith.index_cast %9 : i32 to index
        %31 = affine.apply #map()[%27, %28, %29, %30]
        %32 = arith.index_cast %31 : index to i64
        %33 = arith.cmpi eq, %32, %c1_i64 : i64
        %34 = arith.select %33, %1, %0 : memref<2048xf32, #hivm.address_space<ub>>
        %35 = arith.index_cast %arg12 : i32 to index
        %36 = affine.apply #map1()[%19, %35]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %37 = affine.max #map2()[%35, %13]
        %38 = affine.min #map3()[%37, %35]
        %39 = affine.apply #map4()[%38, %35]
        %40 = arith.cmpi slt, %39, %c2048 : index
        %subview = memref.subview %reinterpret_cast[0] [%39] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %34[0] [%39] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %40 {
          %cast_43 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_43) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %cast_4 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_5 = memref.cast %subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_4, %cast_5, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %41 = hivm.hir.pointer_cast(%c8192_i64) : memref<1xf32, #hivm.address_space<ub>>
        %42 = hivm.hir.pointer_cast(%c8224_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_6 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %41 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %42 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_max_r_float(%cast_6, %cast_7, %cast_8, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %43 = memref.load %41[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %44 = hivm.hir.pointer_cast(%c8480_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %arg14, %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
        %45 = hivm.hir.pointer_cast(%c8512_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %43, %45[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %46 = hivm.hir.pointer_cast(%c8544_i64) : memref<1xi1, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
        %cast_9 = memref.cast %44 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %44 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_11 = memref.cast %46 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_9, %cast_10, %cast_11) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %47 = hivm.hir.pointer_cast(%c8544_i64) : memref<1xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %48 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_12 = memref.cast %46 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %47 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_14 = memref.cast %48 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_12, %cast_13, %cast_14) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %49 = hivm.hir.pointer_cast(%c8576_i64) : memref<1xi1, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_15 = memref.cast %45 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_16 = memref.cast %45 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_17 = memref.cast %49 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_15, %cast_16, %cast_17) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %50 = hivm.hir.pointer_cast(%c8576_i64) : memref<1xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %51 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_18 = memref.cast %49 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_19 = memref.cast %50 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_20 = memref.cast %51 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_18, %cast_19, %cast_20) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %52 = hivm.hir.pointer_cast(%c8608_i64) : memref<1xf32, #hivm.address_space<ub>>
        %53 = arith.maximumf %arg14, %43 : f32
        memref.store %53, %52[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %54 = hivm.hir.pointer_cast(%c8640_i64) : memref<1xf32, #hivm.address_space<ub>>
        %55 = hivm.hir.pointer_cast(%c8672_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_21 = memref.cast %47 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_22 = memref.cast %45 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_23 = memref.cast %52 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %54 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %55 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_21, %cast_22, %cast_23, %cast_24, %cast_25) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %56 = hivm.hir.pointer_cast(%c8704_i64) : memref<1xf32, #hivm.address_space<ub>>
        %57 = hivm.hir.pointer_cast(%c8736_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_26 = memref.cast %50 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_27 = memref.cast %44 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_28 = memref.cast %54 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_29 = memref.cast %56 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %57 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_26, %cast_27, %cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %58 = memref.load %56[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %59 = arith.subf %arg14, %58 : f32
        %60 = hivm.hir.pointer_cast(%c8768_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %59, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %61 = hivm.hir.pointer_cast(%c8768_i64) : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %62 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_31 = memref.cast %60 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_32 = memref.cast %61 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_33 = memref.cast %62 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_1d_float(%cast_31, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %63 = memref.load %61[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %64 = arith.mulf %arg13, %63 : f32
        %65 = arith.subf %cst_0, %58 : f32
        %66 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_34 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_35 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_36 = memref.cast %66 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_34, %65, %cast_35, %cast_36) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %67 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_37 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_38 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_39 = memref.cast %67 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_1d_float(%cast_37, %cast_38, %cast_39) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %68 = hivm.hir.pointer_cast(%c8800_i64) : memref<1xf32, #hivm.address_space<ub>>
        %69 = hivm.hir.pointer_cast(%c8832_i64) : memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_40 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %68 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_42 = memref.cast %69 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_sum_r_float(%cast_40, %cast_41, %cast_42, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %70 = memref.load %68[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %71 = arith.addf %64, %70 : f32
        scf.yield %71, %58 : f32, f32
      } {pipeline.veconly}
      %21 = hivm.hir.pointer_cast(%c12928_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %20#0, %21[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %22 = hivm.hir.pointer_cast(%c12928_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %23 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast = memref.cast %21 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_1 = memref.cast %22 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_2 = memref.cast %23 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vln_1d_float(%cast, %cast_1, %cast_2) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %24 = memref.load %22[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %25 = arith.addf %24, %20#1 : f32
      %26 = arith.index_cast %18 : i64 to index
      scf.for %arg12 = %c0_i32 to %arg7 step %c2048_i32  : i32 {
        %27 = arith.index_cast %arg12 : i32 to index
        %28 = arith.index_cast %arg7 : i32 to index
        %29 = arith.index_cast %arg11 : i32 to index
        %30 = arith.index_cast %9 : i32 to index
        %31 = affine.apply #map()[%27, %28, %29, %30]
        %32 = arith.index_cast %31 : index to i64
        %33 = arith.cmpi eq, %32, %c1_i64 : i64
        %34 = arith.select %33, %3, %2 : memref<2048xf32, #hivm.address_space<ub>>
        %35 = arith.index_cast %arg12 : i32 to index
        %36 = arith.index_cast %arg7 : i32 to index
        %37 = arith.index_cast %arg11 : i32 to index
        %38 = arith.index_cast %9 : i32 to index
        %39 = affine.apply #map()[%35, %36, %37, %38]
        %40 = arith.index_cast %39 : index to i64
        %41 = arith.cmpi eq, %40, %c1_i64 : i64
        %42 = arith.select %41, %5, %4 : memref<2048xf32, #hivm.address_space<ub>>
        %43 = arith.index_cast %arg12 : i32 to index
        %44 = arith.index_cast %arg7 : i32 to index
        %45 = arith.index_cast %arg11 : i32 to index
        %46 = arith.index_cast %9 : i32 to index
        %47 = affine.apply #map()[%43, %44, %45, %46]
        %48 = arith.index_cast %47 : index to i1
        %49 = arith.select %48, %c1_i64, %c2_i64 : i64
        %50 = arith.index_cast %47 : index to i1
        %51 = arith.xori %50, %true : i1
        %52 = arith.extui %51 : i1 to i64
        %53 = arith.index_cast %arg12 : i32 to index
        %54 = affine.apply #map1()[%19, %53]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %55 = affine.max #map2()[%53, %13]
        %56 = affine.min #map3()[%55, %53]
        %57 = affine.apply #map4()[%56, %53]
        %58 = arith.cmpi slt, %57, %c2048 : index
        %subview = memref.subview %reinterpret_cast[0] [%57] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %42[0] [%57] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %58 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_14 = memref.cast %42 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %49]
        %cast_4 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_5 = memref.cast %subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_4, %cast_5, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %59 = arith.subf %cst_0, %25 : f32
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %52]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %60 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_6 = memref.cast %42 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %34 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %60 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_6, %59, %cast_7, %cast_8) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %49]
        %61 = affine.apply #map1()[%26, %53]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %34[0] [%57] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_9[0] [%57] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_12 = memref.cast %subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_12, %cast_13, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %52]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
