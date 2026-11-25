#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map3 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 floordiv 8 + (s2 floordiv s4) * ((s1 + 7) floordiv 8) + ((s5 - s6) floordiv 48) * (((s1 + 7) floordiv 8) * ((s3 + s4 - 1) floordiv s4))) mod 2)>
#map6 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>
#map7 = affine_map<()[s0, s1] -> (s1 + 8, s0)>
#map8 = affine_map<()[s0, s1] -> (8, s0 - s1)>
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
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %0 = hivm.hir.pointer_cast(%c512_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1536_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg7, %arg8 : i32
    %5 = arith.muli %4, %arg9 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg9, %arg8 : i32
    %9 = arith.muli %arg8, %c16_i32 : i32
    %10 = arith.index_cast %arg6 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg10 = %7 to %5 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %11 = arith.divsi %arg10, %arg9 : i32
      %12 = arith.remsi %11, %arg8 : i32
      %13 = arith.divsi %arg10, %8 : i32
      %14 = arith.remsi %13, %arg7 : i32
      %15 = arith.index_cast %14 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %16 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %17 = affine.apply #map()[%15]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %18 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %19 = arith.subi %18, %16 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.muli %16, %arg6 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %12, %c16_i32 : i32
      scf.for %arg11 = %c0_i32 to %19 step %9  : i32 {
        %24 = arith.addi %arg11, %23 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = affine.max #map1()[%25, %20]
        %27 = affine.min #map2()[%26, %25]
        %28 = affine.min #map3()[%27, %25]
        %29 = affine.max #map4()[%28]
        %30 = arith.cmpi slt, %29, %c16 : index
        scf.for %arg12 = %c0_i32 to %arg6 step %c8_i32  : i32 {
          %31 = arith.index_cast %arg12 : i32 to index
          %32 = arith.index_cast %arg6 : i32 to index
          %33 = arith.index_cast %arg11 : i32 to index
          %34 = arith.index_cast %19 : i32 to index
          %35 = arith.index_cast %9 : i32 to index
          %36 = arith.index_cast %arg10 : i32 to index
          %37 = arith.index_cast %7 : i32 to index
          %38 = affine.apply #map5()[%31, %32, %33, %34, %35, %36, %37]
          %39 = arith.index_cast %38 : index to i64
          %40 = arith.cmpi eq, %39, %c1_i64 : i64
          %41 = arith.select %40, %1, %0 : memref<8x16xf32, #hivm.address_space<ub>>
          %42 = arith.index_cast %arg12 : i32 to index
          %43 = arith.index_cast %arg6 : i32 to index
          %44 = arith.index_cast %arg11 : i32 to index
          %45 = arith.index_cast %19 : i32 to index
          %46 = arith.index_cast %9 : i32 to index
          %47 = arith.index_cast %arg10 : i32 to index
          %48 = arith.index_cast %7 : i32 to index
          %49 = affine.apply #map5()[%42, %43, %44, %45, %46, %47, %48]
          %50 = arith.index_cast %49 : index to i64
          %51 = arith.cmpi eq, %50, %c1_i64 : i64
          %52 = arith.select %51, %3, %2 : memref<16x8xf32, #hivm.address_space<ub>>
          %53 = arith.index_cast %arg12 : i32 to index
          %54 = arith.index_cast %arg6 : i32 to index
          %55 = arith.index_cast %arg11 : i32 to index
          %56 = arith.index_cast %19 : i32 to index
          %57 = arith.index_cast %9 : i32 to index
          %58 = arith.index_cast %arg10 : i32 to index
          %59 = arith.index_cast %7 : i32 to index
          %60 = affine.apply #map5()[%53, %54, %55, %56, %57, %58, %59]
          %61 = arith.index_cast %60 : index to i1
          %62 = arith.select %61, %c1_i64, %c2_i64 : i64
          %63 = arith.index_cast %60 : index to i1
          %64 = arith.xori %63, %true : i1
          %65 = arith.extui %64 : i1 to i64
          %66 = arith.index_cast %arg12 : i32 to index
          %67 = affine.apply #map6()[%66, %22, %25, %10]
          %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [16, 8], strides: [%10, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %68 = affine.max #map1()[%66, %10]
          %69 = affine.min #map7()[%68, %66]
          %70 = affine.min #map8()[%69, %66]
          %71 = affine.max #map4()[%70]
          %72 = arith.cmpi slt, %71, %c8 : index
          %73 = arith.ori %30, %72 : i1
          %subview = memref.subview %reinterpret_cast_1[0, 0] [%29, %71] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %52[0, 0] [%29, %71] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          scf.if %73 {
            %collapse_shape = memref.collapse_shape %52 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_11 = memref.cast %collapse_shape : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_11) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %62]
          %cast = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_3 = memref.cast %subview_2 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast, %cast_3, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %74 = affine.apply #map6()[%22, %25, %66, %20]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%74], sizes: [8, 16], strides: [%20, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %65]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_5 = memref.cast %52 : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_6 = memref.cast %41 : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @transpose_2d_with_last_axis_float(%cast_5, %cast_6) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %62]
          %subview_7 = memref.subview %41[0, 0] [%71, %29] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          %subview_8 = memref.subview %reinterpret_cast_4[0, 0] [%71, %29] [1, 1] : memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_9 = memref.cast %subview_7 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_10 = memref.cast %subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_2d_float(%cast_9, %cast_10, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %65]
        } {pipeline.veconly}
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
