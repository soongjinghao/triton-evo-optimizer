#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 4)>
#map2 = affine_map<()[s0] -> (4, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map4 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map5 = affine_map<()[s0] -> (0, s0)>
#map6 = affine_map<()[s0] -> (4, 16, s0)>
#map7 = affine_map<()[s0] -> (16, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_pack_seq_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_pack_seq_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %0 = hivm.hir.pointer_cast(%c4096_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c8192_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %2 = arith.muli %arg7, %arg8 : i32
    %3 = arith.muli %2, %arg9 : i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.muli %arg9, %arg8 : i32
    %7 = hivm.hir.pointer_cast(%c0_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %cast = memref.cast %collapse_shape : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg10 = %5 to %3 step %c48_i32  : i32 {
      %8 = arith.index_cast %arg10 : i32 to index
      %9 = arith.index_cast %5 : i32 to index
      %10 = affine.apply #map()[%8, %9]
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.cmpi eq, %11, %c1_i64 : i64
      %13 = arith.select %12, %1, %0 : memref<16x64xf32, #hivm.address_space<ub>>
      %14 = arith.index_cast %arg10 : i32 to index
      %15 = arith.index_cast %5 : i32 to index
      %16 = affine.apply #map()[%14, %15]
      %17 = arith.index_cast %16 : index to i1
      %18 = arith.xori %17, %true : i1
      %19 = arith.extui %18 : i1 to i64
      hivm.hir.set_mask_norm
      %20 = arith.divsi %arg10, %arg9 : i32
      %21 = arith.remsi %20, %arg8 : i32
      %22 = arith.divsi %arg10, %6 : i32
      %23 = arith.remsi %22, %arg7 : i32
      %24 = arith.muli %21, %c64_i32 : i32
      %25 = arith.index_cast %23 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %26 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %27 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %28 = arith.index_cast %26 : i32 to index
      %29 = arith.index_cast %24 : i32 to index
      %30 = affine.apply #map1()[%29, %28]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [16, 64], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.muli %23, %c4_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = affine.apply #map1()[%29, %32]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [16, 64], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max #map2()[%29]
      %35 = affine.min #map3()[%34, %29]
      %36 = affine.min #map4()[%35, %29]
      %37 = affine.max #map5()[%36]
      %subview = memref.subview %7[0, 0] [4, %37] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<4x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [4, %37] [1, 1] : memref<16x64xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<4x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_4 = memref.cast %subview : memref<4x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %subview_3 : memref<4x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_4, %cast_5, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      %38 = arith.index_cast %27 : i32 to index
      %39 = affine.max #map5()[%38]
      %40 = affine.min #map6()[%39]
      %41 = affine.max #map5()[%40]
      %42 = affine.min #map7()[%41]
      %43 = affine.max #map5()[%42]
      %44 = arith.cmpi slt, %43, %c16 : index
      %45 = arith.cmpi slt, %37, %c64 : index
      %46 = arith.ori %44, %45 : i1
      %subview_6 = memref.subview %reinterpret_cast_1[0, 0] [%43, %37] [1, 1] : memref<16x64xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %13[0, 0] [%43, %37] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      scf.if %46 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_13 = memref.collapse_shape %13 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_14 = memref.cast %collapse_shape_13 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %19]
      %cast_8 = memref.cast %subview_6 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_9 = memref.cast %subview_7 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_8, %cast_9, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_10 = memref.subview %reinterpret_cast_2[0, 0] [%43, %37] [1, 1] : memref<16x64xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_11 = memref.cast %subview_7 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %subview_10 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_11, %cast_12, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %19]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
