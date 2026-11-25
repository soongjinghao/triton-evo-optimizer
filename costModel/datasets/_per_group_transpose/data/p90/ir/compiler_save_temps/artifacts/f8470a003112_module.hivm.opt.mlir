#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map3 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1, s2, s3, s4] -> ((s0 floordiv s2 + ((s3 - s4) floordiv 48) * ((s1 + s2 - 1) floordiv s2)) mod 2)>
#map6 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>
#map7 = affine_map<()[s0, s1] -> (s1 + 32, s0)>
#map8 = affine_map<()[s0, s1] -> (32, s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_per_group_transpose_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @transpose_2d_with_last_axis_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_per_group_transpose(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %0 = hivm.hir.pointer_cast(%c2048_i64) : memref<16x32xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c6144_i64) : memref<16x32xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<32x16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c4096_i64) : memref<32x16xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg7, %arg8 : i32
    %5 = arith.muli %4, %arg9 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg9, %arg8 : i32
    %9 = arith.muli %arg8, %c32_i32 : i32
    %10 = arith.index_cast %arg6 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg10 = %7 to %5 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %11 = arith.remsi %arg10, %arg9 : i32
      %12 = arith.divsi %arg10, %arg9 : i32
      %13 = arith.remsi %12, %arg8 : i32
      %14 = arith.divsi %arg10, %8 : i32
      %15 = arith.remsi %14, %arg7 : i32
      %16 = arith.index_cast %15 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %17 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %18 = affine.apply #map()[%16]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %19 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %20 = arith.subi %19, %17 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.muli %17, %arg6 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.muli %11, %c16_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.muli %13, %c32_i32 : i32
      %27 = affine.max #map1()[%25, %10]
      %28 = affine.min #map2()[%27, %25]
      %29 = affine.min #map3()[%28, %25]
      %30 = affine.max #map4()[%29]
      %31 = arith.cmpi slt, %30, %c16 : index
      scf.for %arg11 = %c0_i32 to %20 step %9  : i32 {
        %32 = arith.index_cast %arg11 : i32 to index
        %33 = arith.index_cast %20 : i32 to index
        %34 = arith.index_cast %9 : i32 to index
        %35 = arith.index_cast %arg10 : i32 to index
        %36 = arith.index_cast %7 : i32 to index
        %37 = affine.apply #map5()[%32, %33, %34, %35, %36]
        %38 = arith.index_cast %37 : index to i64
        %39 = arith.cmpi eq, %38, %c1_i64 : i64
        %40 = arith.select %39, %1, %0 : memref<16x32xf32, #hivm.address_space<ub>>
        %41 = arith.index_cast %arg11 : i32 to index
        %42 = arith.index_cast %20 : i32 to index
        %43 = arith.index_cast %9 : i32 to index
        %44 = arith.index_cast %arg10 : i32 to index
        %45 = arith.index_cast %7 : i32 to index
        %46 = affine.apply #map5()[%41, %42, %43, %44, %45]
        %47 = arith.index_cast %46 : index to i64
        %48 = arith.cmpi eq, %47, %c1_i64 : i64
        %49 = arith.select %48, %3, %2 : memref<32x16xf32, #hivm.address_space<ub>>
        %50 = arith.index_cast %arg11 : i32 to index
        %51 = arith.index_cast %20 : i32 to index
        %52 = arith.index_cast %9 : i32 to index
        %53 = arith.index_cast %arg10 : i32 to index
        %54 = arith.index_cast %7 : i32 to index
        %55 = affine.apply #map5()[%50, %51, %52, %53, %54]
        %56 = arith.index_cast %55 : index to i1
        %57 = arith.select %56, %c1_i64, %c2_i64 : i64
        %58 = arith.index_cast %55 : index to i1
        %59 = arith.xori %58, %true : i1
        %60 = arith.extui %59 : i1 to i64
        %61 = arith.addi %arg11, %26 : i32
        %62 = arith.index_cast %61 : i32 to index
        %63 = affine.apply #map6()[%25, %23, %62, %10]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [32, 16], strides: [%10, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %64 = affine.max #map1()[%62, %21]
        %65 = affine.min #map7()[%64, %62]
        %66 = affine.min #map8()[%65, %62]
        %67 = affine.max #map4()[%66]
        %68 = arith.cmpi slt, %67, %c32 : index
        %69 = arith.ori %68, %31 : i1
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%67, %30] [1, 1] : memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %49[0, 0] [%67, %30] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        scf.if %69 {
          %collapse_shape = memref.collapse_shape %49 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_11 = memref.cast %collapse_shape : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_11) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %57]
        %cast = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_3 = memref.cast %subview_2 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast, %cast_3, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %70 = affine.apply #map6()[%23, %62, %25, %21]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%70], sizes: [16, 32], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %60]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_5 = memref.cast %49 : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %40 : memref<16x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @transpose_2d_with_last_axis_float(%cast_5, %cast_6) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %57]
        %subview_7 = memref.subview %40[0, 0] [%30, %67] [1, 1] : memref<16x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_8 = memref.subview %reinterpret_cast_4[0, 0] [%30, %67] [1, 1] : memref<16x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_9 = memref.cast %subview_7 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_9, %cast_10, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %60]
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
