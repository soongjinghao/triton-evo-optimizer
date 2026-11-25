#map = affine_map<()[s0] -> (s0 * 512)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0] -> (s0 * 768)>
#map5 = affine_map<()[s0] -> (s0 * 32)>
#map6 = affine_map<()[s0] -> (s0 * 768 + 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fn_triton_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_middle_axis_align_3d_float(memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fn_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c48_i32 = arith.constant 48 : i32
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %4 = arith.muli %arg9, %arg8 : i32
      %5 = arith.divsi %arg10, %4 : i32
      %6 = arith.remsi %5, %arg7 : i32
      %7 = arith.muli %6, %c64_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = affine.apply #map()[%8]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [64, 8, 64], strides: [512, 64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %10 = hivm.hir.pointer_cast(%c0_i64) : memref<64x8x64xf32, #hivm.address_space<ub>>
      %11 = arith.index_cast %arg6 : i32 to index
      %12 = affine.max #map1()[%8, %11]
      %13 = affine.min #map2()[%12, %8]
      %14 = affine.apply #map3()[%13, %8]
      %15 = arith.cmpi slt, %14, %c64 : index
      %subview = memref.subview %reinterpret_cast[0, 0, 0] [%14, 8, 64] [1, 1, 1] : memref<64x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %10[0, 0, 0] [%14, 8, 64] [1, 1, 1] : memref<64x8x64xf32, #hivm.address_space<ub>> to memref<?x8x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>>
      scf.if %15 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_25 = memref.collapse_shape %10 [[0, 1, 2]] : memref<64x8x64xf32, #hivm.address_space<ub>> into memref<32768xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_26 = memref.cast %collapse_shape_25 : memref<32768xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_26) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview [[0, 1, 2]] : memref<?x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1, 2]] : memref<?x8x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %collapse_shape_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast, %cast_2, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %16 = affine.apply #map4()[%8]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [64, 8, 64], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %reinterpret_cast_3[0, 0, 0] [%14, 8, 64] [1, 1, 1] : memref<64x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_5 = memref.collapse_shape %subview_0 [[0, 1], [2]] : memref<?x8x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> into memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %collapse_shape_6 = memref.collapse_shape %subview_4 [[0, 1], [2]] : memref<?x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x64xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_7 = memref.cast %collapse_shape_5 : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %collapse_shape_6 : memref<?x64xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_7, %cast_8, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %17 = affine.apply #map5()[%8]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%17], sizes: [64, 1, 32], strides: [32, 32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %18 = hivm.hir.pointer_cast(%c131072_i64) : memref<64x1x32xf32, #hivm.address_space<ub>>
      %subview_10 = memref.subview %reinterpret_cast_9[0, 0, 0] [%14, 1, 32] [1, 1, 1] : memref<64x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_11 = memref.subview %18[0, 0, 0] [%14, 1, 32] [1, 1, 1] : memref<64x1x32xf32, #hivm.address_space<ub>> to memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
      scf.if %15 {
        %collapse_shape_25 = memref.collapse_shape %18 [[0, 1, 2]] : memref<64x1x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_26 = memref.cast %collapse_shape_25 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_26) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_12 = memref.collapse_shape %subview_10 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_13 = memref.collapse_shape %subview_11 [[0, 1, 2]] : memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %collapse_shape_12 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_15 = memref.cast %collapse_shape_13 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_14, %cast_15, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %19 = affine.apply #map6()[%8]
      %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [64, 8, 32], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      %20 = hivm.hir.pointer_cast(%c0_i64) : memref<64x8x32xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_17 = memref.cast %18 : memref<64x1x32xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %20 : memref<64x8x32xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_middle_axis_align_3d_float(%cast_17, %cast_18) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_19 = memref.subview %20[0, 0, 0] [%14, 8, 32] [1, 1, 1] : memref<64x8x32xf32, #hivm.address_space<ub>> to memref<?x8x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>>
      %subview_20 = memref.subview %reinterpret_cast_16[0, 0, 0] [%14, 8, 32] [1, 1, 1] : memref<64x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_21 = memref.collapse_shape %subview_19 [[0, 1], [2]] : memref<?x8x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>> into memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
      %collapse_shape_22 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<?x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x32xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_23 = memref.cast %collapse_shape_21 : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_24 = memref.cast %collapse_shape_22 : memref<?x32xf32, strided<[96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_23, %cast_24, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
