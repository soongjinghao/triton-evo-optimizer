#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0] -> (128, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @reshape_and_cache_kernel_flash_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @reshape_and_cache_kernel_flash(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: i64 {tt.divisibility = 16 : i32}, %arg14: i64 {tt.divisibility = 16 : i32}, %arg15: i64 {tt.divisibility = 16 : i32}, %arg16: i64 {tt.divisibility = 16 : i32}, %arg17: i64 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<21xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %c256_i64 = arith.constant 256 : i64
    %c768_i64 = arith.constant 768 : i64
    %0 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c768_i64) : memref<128xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf16, #hivm.address_space<ub>>
    %4 = arith.muli %arg18, %arg19 : i32
    %5 = arith.muli %4, %arg20 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.index_cast %arg15 : i64 to index
    %9 = arith.index_cast %arg16 : i64 to index
    %10 = arith.index_cast %arg17 : i64 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    scf.for %arg21 = %7 to %5 step %c48_i32  : i32 {
      %11 = arith.index_cast %arg21 : i32 to index
      %12 = arith.index_cast %7 : i32 to index
      %13 = affine.apply #map()[%11, %12]
      %14 = arith.index_cast %13 : index to i64
      %15 = arith.cmpi eq, %14, %c1_i64 : i64
      %16 = arith.select %15, %1, %0 : memref<128xf16, #hivm.address_space<ub>>
      %17 = arith.index_cast %arg21 : i32 to index
      %18 = arith.index_cast %7 : i32 to index
      %19 = affine.apply #map()[%17, %18]
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      %22 = arith.select %21, %3, %2 : memref<128xf16, #hivm.address_space<ub>>
      %23 = arith.index_cast %arg21 : i32 to index
      %24 = arith.index_cast %7 : i32 to index
      %25 = affine.apply #map()[%23, %24]
      %26 = arith.index_cast %25 : index to i1
      %27 = arith.select %26, %c2_i64, %c3_i64 : i64
      %28 = arith.index_cast %25 : index to i1
      %29 = arith.xori %28, %true : i1
      %30 = arith.extui %29 : i1 to i64
      hivm.hir.set_mask_norm
      %31 = arith.divsi %arg21, %arg20 : i32
      %32 = arith.remsi %31, %arg19 : i32
      %33 = arith.muli %32, %c128_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = affine.apply #map1()[%8, %34]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %36 = affine.max #map2()[%34]
      %37 = affine.min #map3()[%36, %34]
      %38 = affine.apply #map4()[%37, %34]
      %39 = arith.cmpi slt, %38, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %22[0] [%38] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %39 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_15 = memref.cast %22 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_half_to_1d(%cst, %cast_15) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %30]
      %cast = memref.cast %subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_1 = memref.cast %subview_0 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_half(%cast, %cast_1, %c2_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      %40 = affine.apply #map1()[%9, %34]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %16[0] [%38] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %39 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_15 = memref.cast %16 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_half_to_1d(%cst, %cast_15) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %27]
      %cast_5 = memref.cast %subview_3 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_6 = memref.cast %subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_half(%cast_5, %cast_6, %c2_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %41 = affine.apply #map1()[%10, %34]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%41], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %reinterpret_cast_7[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_9 = memref.cast %subview_0 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %subview_8 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_half(%cast_9, %cast_10, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %30]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %reinterpret_cast_11[0] [%38] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_13 = memref.cast %subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %subview_12 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_half(%cast_13, %cast_14, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %27]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
