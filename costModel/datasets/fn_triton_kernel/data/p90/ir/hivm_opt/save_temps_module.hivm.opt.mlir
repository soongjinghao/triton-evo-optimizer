#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 * 512 + s1 * 64)>
#map2 = affine_map<()[s0, s1] -> (s1, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 32, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
#map5 = affine_map<()[s0] -> (8, s0)>
#map6 = affine_map<()[s0, s1] -> (s1 + 8, s0)>
#map7 = affine_map<()[s0, s1] -> (32, s0 - s1)>
#map8 = affine_map<()[s0] -> (0, s0)>
#map9 = affine_map<()[s0, s1] -> (8, s0 - s1)>
#map10 = affine_map<()[s0, s1] -> (s0 * 768 + s1 * 96)>
#map11 = affine_map<()[s0] -> (s0 * 32)>
#map12 = affine_map<()[s0, s1] -> (s0 * 768 + s1 * 96 + 64)>
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
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c31_i32 = arith.constant 31 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %c135168_i64 = arith.constant 135168 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c139264_i64 = arith.constant 139264 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c139264_i64) : memref<32x8x32xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8x32xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<32x1x32xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c135168_i64) : memref<32x1x32xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8x64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c65536_i64) : memref<32x8x64xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg7, %arg8 : i32
    %7 = arith.muli %6, %arg9 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg9, %arg8 : i32
    %11 = arith.addi %arg6, %c31_i32 : i32
    %12 = arith.divsi %11, %c32_i32 : i32
    %13 = arith.index_cast %arg6 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg10 = %9 to %7 step %c48_i32  : i32 {
      %14 = arith.index_cast %arg10 : i32 to index
      %15 = arith.index_cast %9 : i32 to index
      %16 = affine.apply #map()[%14, %15]
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.cmpi eq, %17, %c1_i64 : i64
      %19 = arith.select %18, %1, %0 : memref<32x8x32xf32, #hivm.address_space<ub>>
      %20 = arith.index_cast %arg10 : i32 to index
      %21 = arith.index_cast %9 : i32 to index
      %22 = affine.apply #map()[%20, %21]
      %23 = arith.index_cast %22 : index to i64
      %24 = arith.cmpi eq, %23, %c1_i64 : i64
      %25 = arith.select %24, %3, %2 : memref<32x1x32xf32, #hivm.address_space<ub>>
      %26 = arith.index_cast %arg10 : i32 to index
      %27 = arith.index_cast %9 : i32 to index
      %28 = affine.apply #map()[%26, %27]
      %29 = arith.index_cast %28 : index to i64
      %30 = arith.cmpi eq, %29, %c1_i64 : i64
      %31 = arith.select %30, %5, %4 : memref<32x8x64xf32, #hivm.address_space<ub>>
      hivm.hir.set_mask_norm
      %32 = arith.divsi %arg10, %10 : i32
      %33 = arith.remsi %32, %arg7 : i32
      %34 = arith.remsi %33, %12 : i32
      %35 = arith.divsi %33, %12 : i32
      %36 = arith.muli %34, %c32_i32 : i32
      %37 = arith.muli %35, %c8_i32 : i32
      %38 = arith.index_cast %36 : i32 to index
      %39 = arith.index_cast %37 : i32 to index
      %40 = affine.apply #map1()[%38, %39]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [32, 8, 64], strides: [512, 64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %41 = affine.max #map2()[%38, %13]
      %42 = affine.min #map3()[%41, %38]
      %43 = affine.apply #map4()[%42, %38]
      %44 = affine.max #map5()[%39]
      %45 = affine.min #map6()[%44, %39]
      %46 = affine.min #map7()[%42, %38]
      %47 = affine.max #map8()[%46]
      %48 = affine.min #map9()[%45, %39]
      %49 = affine.max #map8()[%48]
      %50 = arith.cmpi slt, %47, %c32 : index
      %51 = arith.cmpi slt, %49, %c8 : index
      %52 = arith.ori %50, %51 : i1
      %subview = memref.subview %reinterpret_cast[0, 0, 0] [%47, %49, 64] [1, 1, 1] : memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %31[0, 0, 0] [%47, %49, 64] [1, 1, 1] : memref<32x8x64xf32, #hivm.address_space<ub>> to memref<?x?x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>>
      scf.if %52 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_21 = memref.collapse_shape %31 [[0, 1, 2]] : memref<32x8x64xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_22 = memref.cast %collapse_shape_21 : memref<16384xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_22) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview [[0], [1, 2]] : memref<?x?x64xf32, strided<[512, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_1 = memref.collapse_shape %subview_0 [[0], [1, 2]] : memref<?x?x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> into memref<?x?xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %collapse_shape : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %collapse_shape_1 : memref<?x?xf32, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast, %cast_2, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %53 = affine.apply #map10()[%38, %39]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%53], sizes: [32, 8, 64], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %reinterpret_cast_3[0, 0, 0] [%47, %49, 64] [1, 1, 1] : memref<32x8x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_5 = memref.cast %subview_0 : memref<?x?x64xf32, strided<[512, 64, 1]>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %subview_4 : memref<?x?x64xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_3d_float(%cast_5, %cast_6, %c0_i32) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %54 = affine.apply #map11()[%38]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 1, 32], strides: [32, 32, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %55 = arith.cmpi slt, %43, %c32 : index
      %subview_8 = memref.subview %reinterpret_cast_7[0, 0, 0] [%43, 1, 32] [1, 1, 1] : memref<32x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %25[0, 0, 0] [%43, 1, 32] [1, 1, 1] : memref<32x1x32xf32, #hivm.address_space<ub>> to memref<?x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
      scf.if %55 {
        %collapse_shape_21 = memref.collapse_shape %25 [[0, 1, 2]] : memref<32x1x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_22 = memref.cast %collapse_shape_21 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
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
      %56 = affine.apply #map12()[%38, %39]
      %reinterpret_cast_14 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [32, 8, 32], strides: [768, 96, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_15 = memref.cast %25 : memref<32x1x32xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %19 : memref<32x8x32xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_middle_axis_align_3d_float(%cast_15, %cast_16) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_17 = memref.subview %19[0, 0, 0] [%47, %49, 32] [1, 1, 1] : memref<32x8x32xf32, #hivm.address_space<ub>> to memref<?x?x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>>
      %subview_18 = memref.subview %reinterpret_cast_14[0, 0, 0] [%47, %49, 32] [1, 1, 1] : memref<32x8x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_19 = memref.cast %subview_17 : memref<?x?x32xf32, strided<[256, 32, 1]>, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %subview_18 : memref<?x?x32xf32, strided<[768, 96, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_3d_float(%cast_19, %cast_20, %c0_i32) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
