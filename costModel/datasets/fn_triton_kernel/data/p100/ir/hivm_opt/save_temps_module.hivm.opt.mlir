#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 * 512 + s1 * 64)>
#map2 = affine_map<()[s0, s1] -> (s1, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
#map5 = affine_map<()[s0] -> (8, s0)>
#map6 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map7 = affine_map<()[s0] -> (0, s0)>
#map8 = affine_map<()[s0, s1] -> (s0 * 768 + s1 * 96)>
#map9 = affine_map<()[s0] -> (s0 * 32)>
#map10 = affine_map<()[s0, s1] -> (s0 * 768 + s1 * 96 + 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fn_triton_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_3d_float(memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_middle_axis_align_3d_float(memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fn_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c16_i32 = arith.constant 16 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %c133120_i64 = arith.constant 133120 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c135168_i64 = arith.constant 135168 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c135168_i64) : memref<16x16x32xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x32xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<16x1x32xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c133120_i64) : memref<16x1x32xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c65536_i64) : memref<16x16x64xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg7, %arg8 : i32
    %7 = arith.muli %6, %arg9 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg9, %arg8 : i32
    %11 = arith.index_cast %arg6 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg10 = %9 to %7 step %c48_i32  : i32 {
      %12 = arith.index_cast %arg10 : i32 to index
      %13 = arith.index_cast %9 : i32 to index
      %14 = affine.apply #map()[%12, %13]
      %15 = arith.index_cast %14 : index to i64
      %16 = arith.cmpi eq, %15, %c1_i64 : i64
      %17 = arith.select %16, %1, %0 : memref<16x16x32xf32, #hivm.address_space<ub>>
      %18 = arith.index_cast %arg10 : i32 to index
      %19 = arith.index_cast %9 : i32 to index
      %20 = affine.apply #map()[%18, %19]
      %21 = arith.index_cast %20 : index to i64
      %22 = arith.cmpi eq, %21, %c1_i64 : i64
      %23 = arith.select %22, %3, %2 : memref<16x1x32xf32, #hivm.address_space<ub>>
      %24 = arith.index_cast %arg10 : i32 to index
      %25 = arith.index_cast %9 : i32 to index
      %26 = affine.apply #map()[%24, %25]
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.cmpi eq, %27, %c1_i64 : i64
      %29 = arith.select %28, %5, %4 : memref<16x16x64xf32, #hivm.address_space<ub>>
      hivm.hir.set_mask_norm
      %30 = arith.divsi %arg10, %arg9 : i32
      %31 = arith.remsi %30, %arg8 : i32
      %32 = arith.divsi %arg10, %10 : i32
      %33 = arith.remsi %32, %arg7 : i32
      %34 = arith.muli %33, %c16_i32 : i32
      %35 = arith.muli %31, %c16_i32 : i32
      %36 = arith.index_cast %34 : i32 to index
      %37 = arith.index_cast %35 : i32 to index
      %38 = affine.apply #map1()[%36, %37]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [16, 16, 64], strides: [512, 64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %39 = affine.max #map2()[%36, %11]
      %40 = affine.min #map3()[%39, %36]
      %41 = affine.apply #map4()[%40, %36]
      %42 = affine.max #map5()[%37]
      %43 = affine.min #map3()[%42, %37]
      %44 = affine.min #map6()[%40, %36]
      %45 = affine.max #map7()[%44]
      %46 = affine.min #map6()[%43, %37]
      %47 = affine.max #map7()[%46]
      %48 = arith.cmpi slt, %45, %c16 : index
      %49 = arith.cmpi slt, %47, %c16 : index
      %50 = arith.ori %48, %49 : i1
      %subview = memref.subview %reinterpret_cast[0, 0, 0] [%45, %47, 64] [1, 1, 1] : memref<16x16x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %29[0, 0, 0] [%45, %47, 64] [1, 1, 1] : memref<16x16x64xf32, #hivm.address_space<ub>> to memref<?x?x64xf32, strided<[1024, 64, 1]>, #hivm.address_space<ub>>
      scf.if %50 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_21 = memref.collapse_shape %29 [[0, 1, 2]] : memref<16x16x64xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_22 = memref.cast %collapse_shape_21 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_22) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview [[0], [1, 2]] : memref<?x?x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_1 = memref.collapse_shape %subview_0 [[0], [1, 2]] : memref<?x?x64xf32, strided<[1024, 64, 1]>, #hivm.address_space<ub>> into memref<?x?xf32, strided<[1024, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %collapse_shape : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %collapse_shape_1 : memref<?x?xf32, strided<[1024, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast, %cast_2, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %51 = affine.apply #map8()[%36, %37]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [16, 16, 64], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %reinterpret_cast_3[0, 0, 0] [%45, %47, 64] [1, 1, 1] : memref<16x16x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_5 = memref.cast %subview_0 : memref<?x?x64xf32, strided<[1024, 64, 1]>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %subview_4 : memref<?x?x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_3d_float(%cast_5, %cast_6, %c0_i32) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %52 = affine.apply #map9()[%36]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [16, 1, 32], strides: [32, 32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %53 = arith.cmpi slt, %41, %c16 : index
      %subview_8 = memref.subview %reinterpret_cast_7[0, 0, 0] [%41, 1, 32] [1, 1, 1] : memref<16x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %23[0, 0, 0] [%41, 1, 32] [1, 1, 1] : memref<16x1x32xf32, #hivm.address_space<ub>> to memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
      scf.if %53 {
        %collapse_shape_21 = memref.collapse_shape %23 [[0, 1, 2]] : memref<16x1x32xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_22 = memref.cast %collapse_shape_21 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_22) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_10 = memref.collapse_shape %subview_8 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %collapse_shape_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_13 = memref.cast %collapse_shape_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_12, %cast_13, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %54 = affine.apply #map10()[%36, %37]
      %reinterpret_cast_14 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [16, 16, 32], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_15 = memref.cast %23 : memref<16x1x32xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %17 : memref<16x16x32xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_middle_axis_align_3d_float(%cast_15, %cast_16) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_17 = memref.subview %17[0, 0, 0] [%45, %47, 32] [1, 1, 1] : memref<16x16x32xf32, #hivm.address_space<ub>> to memref<?x?x32xf32, strided<[512, 32, 1]>, #hivm.address_space<ub>>
      %subview_18 = memref.subview %reinterpret_cast_14[0, 0, 0] [%45, %47, 32] [1, 1, 1] : memref<16x16x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_19 = memref.cast %subview_17 : memref<?x?x32xf32, strided<[512, 32, 1]>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %subview_18 : memref<?x?x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_3d_float(%cast_19, %cast_20, %c0_i32) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
