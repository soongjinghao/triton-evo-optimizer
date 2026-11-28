#map = affine_map<()[s0] -> (256, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
#map4 = affine_map<()[s0, s1] -> (s1, s0)>
#map5 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map6 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map7 = affine_map<()[s0] -> (0, s0)>
#map8 = affine_map<()[s0, s1] -> (128, s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fused_gdn_gating_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_gt_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_half_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_max_ra_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_r_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_int8_t_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_ne_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fused_gdn_gating_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0xFC00 : f16
    %c0_i32 = arith.constant 0 : i32
    %c99328_i64 = arith.constant 99328 : i64
    %c181376_i64 = arith.constant 181376 : i64
    %c164896_i64 = arith.constant 164896 : i64
    %c82944_i64 = arith.constant 82944 : i64
    %c66560_i64 = arith.constant 66560 : i64
    %c181248_i64 = arith.constant 181248 : i64
    %c164864_i64 = arith.constant 164864 : i64
    %c181344_i64 = arith.constant 181344 : i64
    %c33792_i64 = arith.constant 33792 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %c132096_i64 = arith.constant 132096 : i64
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %false = arith.constant false
    %c128 = arith.constant 128 : index
    %cst_1 = arith.constant 2.000000e+01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant -1.000000e+00 : f32
    %c48_i32 = arith.constant 48 : i32
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %4 = arith.muli %arg12, %arg11 : i32
      %5 = arith.divsi %arg13, %4 : i32
      %6 = arith.remsi %5, %arg10 : i32
      %7 = hivm.hir.pointer_cast(%c132096_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %cast = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst_1, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %8 = arith.divsi %6, %c2_i32 : i32
      %9 = arith.remsi %6, %c2_i32 : i32
      %10 = arith.muli %8, %c128_i32 : i32
      %11 = arith.muli %9, %c64_i32 : i32
      %12 = arith.index_cast %10 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %13 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
      %14 = affine.max #map()[%12]
      %15 = affine.min #map1()[%14, %12]
      %16 = affine.apply #map2()[%15, %12]
      %17 = arith.cmpi slt, %16, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%16] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %13[0] [%16] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %17 {
        %cast_107 = memref.cast %13 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_3, %cast_107) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %cast_6 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_7 = memref.cast %subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_6, %cast_7, %c2_i32, %cst_3, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%12], sizes: [1, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %18 = hivm.hir.pointer_cast(%c512_i64) : memref<1x128xf32, #hivm.address_space<ub>>
      %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [1, %16] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %18[0, 0] [1, %16] [1, 1] : memref<1x128xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      scf.if %17 {
        %collapse_shape_107 = memref.collapse_shape %18 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        %cast_108 = memref.cast %collapse_shape_107 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_3, %cast_108) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_12 = memref.collapse_shape %subview_10 [[0, 1]] : memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %collapse_shape_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_14 = memref.cast %collapse_shape_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_13, %cast_14, %c2_i32, %cst_3, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %19 = arith.index_cast %11 : i32 to index
      %20 = affine.apply #map3()[%12, %19]
      %reinterpret_cast_15 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %21 = hivm.hir.pointer_cast(%c1024_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %22 = arith.index_cast %arg9 : i32 to index
      %23 = affine.max #map4()[%19, %22]
      %24 = affine.min #map5()[%23, %19]
      %25 = affine.min #map6()[%24, %19]
      %26 = affine.max #map7()[%25]
      %27 = affine.min #map8()[%15, %12]
      %28 = affine.max #map7()[%27]
      %29 = arith.cmpi slt, %26, %c64 : index
      %30 = arith.cmpi slt, %28, %c128 : index
      %31 = arith.ori %29, %30 : i1
      %subview_16 = memref.subview %reinterpret_cast_15[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_17 = memref.subview %21[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      scf.if %31 {
        %collapse_shape_107 = memref.collapse_shape %21 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %cast_108 = memref.cast %collapse_shape_107 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_3, %cast_108) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_18 = memref.cast %subview_16 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_19 = memref.cast %subview_17 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_18, %cast_19, %c2_i32, %cst_3, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %reinterpret_cast_20 = memref.reinterpret_cast %arg7 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %32 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %subview_21 = memref.subview %reinterpret_cast_20[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_22 = memref.subview %32[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      scf.if %31 {
        %collapse_shape_107 = memref.collapse_shape %32 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_108 = memref.cast %collapse_shape_107 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_3, %cast_108) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %cast_23 = memref.cast %subview_21 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_24 = memref.cast %subview_22 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_23, %cast_24, %c2_i32, %cst_3, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %33 = hivm.hir.pointer_cast(%c1024_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %34 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_25 = memref.cast %21 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %18 : memref<1x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %33 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_28 = memref.cast %34 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_2d_float(%cast_25, %cast_26, %cast_27, %cast_28) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %35 = hivm.hir.pointer_cast(%c132096_i64) : memref<64x128xi1, #hivm.address_space<ub>>
      %collapse_shape_29 = memref.collapse_shape %33 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %collapse_shape_30 = memref.collapse_shape %35 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_31 = memref.cast %collapse_shape_29 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_33 = memref.cast %collapse_shape_30 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_gt_1d_float(%cast_31, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %36 = hivm.hir.pointer_cast(%c181344_i64) : memref<1x1xi1, #hivm.address_space<ub>>
      memref.store %false, %36[%c0, %c0] : memref<1x1xi1, #hivm.address_space<ub>>
      %37 = hivm.hir.pointer_cast(%c164864_i64) : memref<64x128xf16, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %37 [[0, 1]] : memref<64x128xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c181248_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_35 = memref.cast %collapse_shape_30 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %collapse_shape_34 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %38 : memref<48xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_half_1d_with_temp(%cast_35, %cast_36, %cast_37, %c0_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %39 = hivm.hir.pointer_cast(%c164864_i64) : memref<1x1xf16, #hivm.address_space<ub>>
      %40 = hivm.hir.pointer_cast(%c66560_i64) : memref<64x128xf16, #hivm.address_space<ub>>
      %subview_38 = memref.subview %40[0, 0] [1, 128] [1, 1] : memref<64x128xf16, #hivm.address_space<ub>> to memref<1x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c82944_i64) : memref<8192xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_39 = memref.cast %37 : memref<64x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %subview_38 : memref<1x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %41 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_max_ra_half(%cast_39, %cast_40, %cast_41, %cst) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16) -> ()
      %collapse_shape_42 = memref.collapse_shape %subview_38 [[0, 1]] : memref<1x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> into memref<128xf16, strided<[1]>, #hivm.address_space<ub>>
      %collapse_shape_43 = memref.collapse_shape %39 [[0, 1]] : memref<1x1xf16, #hivm.address_space<ub>> into memref<1xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %42 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
      %cast_44 = memref.cast %collapse_shape_42 : memref<128xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %collapse_shape_43 : memref<1xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %42 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_max_r_half(%cast_44, %cast_45, %cast_46, %cst) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16) -> ()
      %43 = hivm.hir.pointer_cast(%c164864_i64) : memref<1x1xi8, #hivm.address_space<ub>>
      %collapse_shape_47 = memref.collapse_shape %43 [[0, 1]] : memref<1x1xi8, #hivm.address_space<ub>> into memref<1xi8, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_48 = memref.cast %collapse_shape_43 : memref<1xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %collapse_shape_47 : memref<1xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_half_to_int8_t_1d_with_mode(%cast_48, %cast_49, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %44 = hivm.hir.pointer_cast(%c164896_i64) : memref<1x1xf16, #hivm.address_space<ub>>
      %collapse_shape_50 = memref.collapse_shape %44 [[0, 1]] : memref<1x1xf16, #hivm.address_space<ub>> into memref<1xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_51 = memref.cast %collapse_shape_47 : memref<1xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_52 = memref.cast %collapse_shape_50 : memref<1xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_int8_t_to_half_1d_with_mode(%cast_51, %cast_52, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %45 = hivm.hir.pointer_cast(%c181376_i64) : memref<1x1xf16, #hivm.address_space<ub>>
      %collapse_shape_53 = memref.collapse_shape %45 [[0, 1]] : memref<1x1xf16, #hivm.address_space<ub>> into memref<1xf16, #hivm.address_space<ub>>
      memref.store %cst_0, %collapse_shape_53[%c0] : memref<1xf16, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_54 = memref.collapse_shape %36 [[0, 1]] : memref<1x1xi1, #hivm.address_space<ub>> into memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_55 = memref.cast %collapse_shape_50 : memref<1xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_56 = memref.cast %collapse_shape_53 : memref<1xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_57 = memref.cast %collapse_shape_54 : memref<1xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_ne_1d_half(%cast_55, %cast_56, %cast_57) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %46 = memref.load %36[%c0, %c0] : memref<1x1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      %47 = scf.if %46 -> (memref<64x128xf32, #hivm.address_space<ub>>) {
        %67 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        %collapse_shape_107 = memref.collapse_shape %67 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %68 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_108 = memref.cast %collapse_shape_29 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_109 = memref.cast %collapse_shape_107 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_110 = memref.cast %68 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_1d_float(%cast_108, %cast_109, %cast_110) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %69 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        %collapse_shape_111 = memref.collapse_shape %69 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %70 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_112 = memref.cast %collapse_shape_107 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_113 = memref.cast %collapse_shape_111 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_114 = memref.cast %70 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_112, %cst_2, %cast_113, %cast_114) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %71 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        %collapse_shape_115 = memref.collapse_shape %71 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %72 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_116 = memref.cast %collapse_shape_111 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_117 = memref.cast %collapse_shape_115 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_118 = memref.cast %72 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vln_1d_float(%cast_116, %cast_117, %cast_118) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %73 = hivm.hir.pointer_cast(%c66560_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        %collapse_shape_119 = memref.collapse_shape %73 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        %74 = hivm.hir.pointer_cast(%c164864_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_120 = memref.cast %collapse_shape_30 : memref<8192xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_121 = memref.cast %collapse_shape_29 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_122 = memref.cast %collapse_shape_115 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_123 = memref.cast %collapse_shape_119 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_124 = memref.cast %74 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_120, %cast_121, %cast_122, %cast_123, %cast_124) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %73 : memref<64x128xf32, #hivm.address_space<ub>>
      } else {
        scf.yield %33 : memref<64x128xf32, #hivm.address_space<ub>>
      }
      %48 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_58 = memref.cast %13 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_59 = memref.cast %48 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_60 = memref.cast %49 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_58, %cast_59, %cast_60) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %50 = hivm.hir.pointer_cast(%c0_i64) : memref<1x128xf32, #hivm.address_space<ub>>
      %collapse_shape_61 = memref.collapse_shape %50 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %51 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_62 = memref.cast %48 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_63 = memref.cast %collapse_shape_61 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_64 = memref.cast %51 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_62, %cst_4, %cast_63, %cast_64) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %52 = hivm.hir.pointer_cast(%c0_i64) : memref<1x128xf32, #hivm.address_space<ub>>
      %collapse_shape_65 = memref.collapse_shape %52 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %53 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_66 = memref.cast %collapse_shape_61 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_67 = memref.cast %collapse_shape_65 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_68 = memref.cast %53 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_66, %cst_3, %cast_67, %cast_68) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %54 = hivm.hir.pointer_cast(%c66560_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %55 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_69 = memref.cast %52 : memref<1x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_70 = memref.cast %47 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_71 = memref.cast %54 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_72 = memref.cast %55 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_69, %cast_70, %cast_71, %cast_72) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %reinterpret_cast_73 = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_74 = memref.subview %54[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %subview_75 = memref.subview %reinterpret_cast_73[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_76 = memref.cast %subview_74 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_77 = memref.cast %subview_75 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_76, %cast_77, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %56 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape_78 = memref.collapse_shape %32 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %collapse_shape_79 = memref.collapse_shape %56 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %57 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_80 = memref.cast %collapse_shape_78 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_81 = memref.cast %collapse_shape_79 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_82 = memref.cast %57 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_80, %cst_4, %cast_81, %cast_82) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %58 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape_83 = memref.collapse_shape %58 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %59 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_84 = memref.cast %collapse_shape_79 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_85 = memref.cast %collapse_shape_83 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_86 = memref.cast %59 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_84, %cst_3, %cast_85, %cast_86) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %60 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape_87 = memref.collapse_shape %60 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %61 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_88 = memref.cast %collapse_shape_83 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_89 = memref.cast %collapse_shape_87 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_90 = memref.cast %61 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_88, %cast_89, %cast_90) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %62 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape_91 = memref.collapse_shape %62 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %63 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_92 = memref.cast %collapse_shape_87 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_93 = memref.cast %collapse_shape_91 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_94 = memref.cast %63 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_92, %cst_2, %cast_93, %cast_94) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %64 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape_95 = memref.collapse_shape %64 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      %cast_96 = memref.cast %collapse_shape_95 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_96) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %65 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
      %collapse_shape_97 = memref.collapse_shape %65 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %66 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_98 = memref.cast %collapse_shape_95 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_99 = memref.cast %collapse_shape_91 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_100 = memref.cast %collapse_shape_97 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_101 = memref.cast %66 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_1d_float(%cast_98, %cast_99, %cast_100, %cast_101) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %reinterpret_cast_102 = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_103 = memref.subview %65[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %subview_104 = memref.subview %reinterpret_cast_102[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_105 = memref.cast %subview_103 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_106 = memref.cast %subview_104 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_105, %cast_106, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
