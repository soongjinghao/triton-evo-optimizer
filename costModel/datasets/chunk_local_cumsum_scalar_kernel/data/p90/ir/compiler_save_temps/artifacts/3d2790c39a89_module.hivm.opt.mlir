#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 32)>
#map2 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map3 = affine_map<()[s0] -> (32, s0)>
#map4 = affine_map<()[s0, s1] -> (32, s1, s0)>
#map5 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @chunk_local_cumsum_scalar_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @chunk_local_cumsum_scalar_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %0 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c3072_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c2048_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg6, %arg7 : i32
    %5 = arith.muli %4, %arg8 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg8, %arg7 : i32
    %9 = arith.index_cast %arg5 : i32 to index
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg9 = %7 to %5 step %c48_i32  : i32 {
      %10 = arith.index_cast %arg9 : i32 to index
      %11 = arith.index_cast %7 : i32 to index
      %12 = affine.apply #map()[%10, %11]
      %13 = arith.index_cast %12 : index to i64
      %14 = arith.cmpi eq, %13, %c1_i64 : i64
      %15 = arith.select %14, %1, %0 : memref<32x8xf32, #hivm.address_space<ub>>
      %16 = arith.index_cast %arg9 : i32 to index
      %17 = arith.index_cast %7 : i32 to index
      %18 = affine.apply #map()[%16, %17]
      %19 = arith.index_cast %18 : index to i64
      %20 = arith.cmpi eq, %19, %c1_i64 : i64
      %21 = arith.select %20, %3, %2 : memref<32x8xf32, #hivm.address_space<ub>>
      %22 = arith.index_cast %arg9 : i32 to index
      %23 = arith.index_cast %7 : i32 to index
      %24 = affine.apply #map()[%22, %23]
      %25 = arith.index_cast %24 : index to i1
      %26 = arith.xori %25, %true : i1
      %27 = arith.extui %26 : i1 to i64
      hivm.hir.set_mask_norm
      %28 = arith.divsi %arg9, %8 : i32
      %29 = arith.remsi %28, %arg6 : i32
      %30 = arith.divsi %29, %c128_i32 : i32
      %31 = arith.remsi %29, %c128_i32 : i32
      %32 = arith.remsi %31, %c32_i32 : i32
      %33 = arith.divsi %31, %c32_i32 : i32
      %34 = arith.muli %30, %arg5 : i32
      %35 = arith.muli %34, %c32_i32 : i32
      %36 = arith.addi %35, %32 : i32
      %37 = arith.index_cast %36 : i32 to index
      %38 = arith.muli %33, %c32_i32 : i32
      %39 = arith.maxsi %38, %c0_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = affine.apply #map1()[%37, %40]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [32], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [32], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %21[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      %42 = affine.max #map2()[%9, %40]
      %43 = affine.min #map3()[%42]
      %44 = arith.subi %c0_i32, %38 : i32
      %45 = arith.maxsi %44, %c0_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = affine.min #map4()[%46, %42]
      %48 = affine.apply #map5()[%43, %47]
      %49 = arith.cmpi slt, %48, %c32 : index
      %subview_1 = memref.subview %reinterpret_cast[0] [%48] [1] : memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %subview[%47] [%48] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %49 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %cast_20 = memref.cast %reinterpret_cast_19 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_2d(%cst, %cast_20) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %27]
      %cast = memref.cast %subview_1 : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_3 = memref.cast %subview_2 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast, %cast_3, %c2_i32, %cst, %47) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %subview_4 = memref.subview %15[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %subview_4 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %27]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %50 = memref.load %reinterpret_cast_5[%c0, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      memref.store %50, %reinterpret_cast_10[%c0, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.for %arg10 = %c1 to %c32 step %c1 {
        %51 = memref.load %reinterpret_cast_5[%arg10, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %52 = arith.subi %arg10, %c1 : index
        %53 = memref.load %reinterpret_cast_10[%52, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %54 = arith.addf %51, %53 : f32
        memref.store %54, %reinterpret_cast_10[%arg10, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      }
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %27]
      %subview_11 = memref.subview %subview_4[%47] [%48] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      %subview_12 = memref.subview %reinterpret_cast_0[0] [%48] [1] : memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_13 = memref.cast %subview_11 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %subview_12 : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_13, %cast_14, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %27]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
