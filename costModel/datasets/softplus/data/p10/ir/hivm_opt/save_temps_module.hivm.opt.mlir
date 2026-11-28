#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 2048, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @softplus_kernel_default_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_gt_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @softplus_kernel_default(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2048_i32 = arith.constant 2048 : i32
    %cst_1 = arith.constant 2.000000e+01 : f32
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c16640_i64 = arith.constant 16640 : i64
    %c33024_i64 = arith.constant 33024 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c33056_i64 = arith.constant 33056 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c24832_i64 = arith.constant 24832 : i64
    %c41248_i64 = arith.constant 41248 : i64
    %0 = hivm.hir.pointer_cast(%c24832_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c41248_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c8192_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c33056_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg8, %arg9 : i32
    %5 = arith.muli %4, %arg10 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg10, %arg9 : i32
    %9 = hivm.hir.pointer_cast(%c0_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %cast = memref.cast %9 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_1, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %10 = arith.index_cast %arg5 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg11 = %7 to %5 step %c48_i32  : i32 {
      %11 = arith.index_cast %arg11 : i32 to index
      %12 = arith.index_cast %7 : i32 to index
      %13 = affine.apply #map()[%11, %12]
      %14 = arith.index_cast %13 : index to i64
      %15 = arith.cmpi eq, %14, %c1_i64 : i64
      %16 = arith.select %15, %1, %0 : memref<2048xf32, #hivm.address_space<ub>>
      %17 = arith.index_cast %arg11 : i32 to index
      %18 = arith.index_cast %7 : i32 to index
      %19 = affine.apply #map()[%17, %18]
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      %22 = arith.select %21, %3, %2 : memref<2048xf32, #hivm.address_space<ub>>
      %23 = arith.index_cast %arg11 : i32 to index
      %24 = arith.index_cast %7 : i32 to index
      %25 = affine.apply #map()[%23, %24]
      %26 = arith.index_cast %25 : index to i1
      %27 = arith.select %26, %c1_i64, %c2_i64 : i64
      %28 = arith.index_cast %25 : index to i1
      %29 = arith.xori %28, %true : i1
      %30 = arith.extui %29 : i1 to i64
      hivm.hir.set_mask_norm
      %31 = arith.divsi %arg11, %8 : i32
      %32 = arith.remsi %31, %arg8 : i32
      %33 = arith.muli %32, %c2048_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %35 = affine.max #map1()[%34, %10]
      %36 = affine.min #map2()[%35, %34]
      %37 = affine.apply #map3()[%36, %34]
      %38 = arith.cmpi slt, %37, %c2048 : index
      %subview = memref.subview %reinterpret_cast[0] [%37] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %22[0] [%37] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %38 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_27 = memref.cast %22 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_27) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %27]
      %cast_3 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_4 = memref.cast %subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %39 = hivm.hir.pointer_cast(%c16384_i64) : memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_5 = memref.cast %22 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %9 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %39 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_gt_1d_float(%cast_5, %cast_6, %cast_7) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %40 = hivm.hir.pointer_cast(%c16640_i64) : memref<2048xf32, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_8 = memref.cast %22 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_9 = memref.cast %40 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %41 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_8, %cast_9, %cast_10) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %42 = hivm.hir.pointer_cast(%c16640_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %43 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_11 = memref.cast %40 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %42 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %43 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_11, %cst_0, %cast_12, %cast_13) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %44 = hivm.hir.pointer_cast(%c16640_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %45 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_14 = memref.cast %42 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %44 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %45 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vln_1d_float(%cast_14, %cast_15, %cast_16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %46 = hivm.hir.pointer_cast(%c33024_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %30]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_17 = memref.cast %39 : memref<2048xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %22 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %44 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %16 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %46 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vv_1d_bool_float(%cast_17, %cast_18, %cast_19, %cast_20, %cast_21) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %27]
      %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_23 = memref.subview %16[0] [%37] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_24 = memref.subview %reinterpret_cast_22[0] [%37] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_25 = memref.cast %subview_23 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %subview_24 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_25, %cast_26, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %30]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
