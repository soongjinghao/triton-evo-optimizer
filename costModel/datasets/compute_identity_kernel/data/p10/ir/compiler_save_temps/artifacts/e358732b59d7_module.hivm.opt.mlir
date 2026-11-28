#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<()[s0] -> (32, s0)>
#map2 = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 8 + ((s2 - s3) floordiv 48) * ((s1 + 7) floordiv 8)) mod 2)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
#map4 = affine_map<()[s0] -> (8, s0)>
#map5 = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 256 + ((s2 - s3) floordiv 48) * ((s1 + 255) floordiv 256)) mod 2)>
#map6 = affine_map<()[s0] -> (256, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @compute_identity_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c256 = arith.constant 256 : index
    %c8 = arith.constant 8 : index
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c2432_i64 = arith.constant 2432 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c35328_i64 = arith.constant 35328 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c37376_i64 = arith.constant 37376 : i64
    %c2560_i64 = arith.constant 2560 : i64
    %c128_i64 = arith.constant 128 : i64
    %c36352_i64 = arith.constant 36352 : i64
    %0 = hivm.hir.pointer_cast(%c128_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c36352_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c2560_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c37376_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c2560_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c37376_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %6 = arith.muli %arg10, %arg11 : i32
    %7 = arith.muli %6, %arg12 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = arith.muli %arg12, %arg11 : i32
    %11 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
    %cast = memref.cast %11 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %12 = arith.index_cast %arg7 : i32 to index
    %13 = arith.index_cast %arg8 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg13 = %9 to %7 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %14 = arith.divsi %arg13, %10 : i32
      %15 = arith.remsi %14, %arg10 : i32
      %16 = arith.muli %15, %c32_i32 : i32
      %17 = arith.subi %arg7, %16 : i32
      %18 = arith.index_cast %16 : i32 to index
      %19 = arith.index_cast %17 : i32 to index
      %20 = affine.max #map()[%19]
      %21 = affine.min #map1()[%20]
      %22 = affine.max #map()[%21]
      %23 = arith.cmpi slt, %22, %c32 : index
      %24 = hivm.hir.pointer_cast(%c2432_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_0 = memref.cast %11 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_1 = memref.cast %24 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_0, %cast_1, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %25 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %24) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
        %28 = arith.index_cast %arg14 : i32 to index
        %29 = arith.index_cast %arg3 : i32 to index
        %30 = arith.index_cast %arg13 : i32 to index
        %31 = arith.index_cast %9 : i32 to index
        %32 = affine.apply #map2()[%28, %29, %30, %31]
        %33 = arith.index_cast %32 : index to i64
        %34 = arith.cmpi eq, %33, %c1_i64 : i64
        %35 = arith.select %34, %1, %0 : memref<8x32xf32, #hivm.address_space<ub>>
        %36 = arith.index_cast %arg14 : i32 to index
        %37 = arith.index_cast %arg3 : i32 to index
        %38 = arith.index_cast %arg13 : i32 to index
        %39 = arith.index_cast %9 : i32 to index
        %40 = affine.apply #map2()[%36, %37, %38, %39]
        %41 = arith.index_cast %40 : index to i1
        %42 = arith.select %41, %c1_i64, %c2_i64 : i64
        %43 = arith.subi %arg3, %arg14 : i32
        %44 = arith.muli %arg14, %arg7 : i32
        %45 = arith.index_cast %44 : i32 to index
        %46 = affine.apply #map3()[%45, %18]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [8, 32], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %47 = arith.index_cast %43 : i32 to index
        %48 = affine.max #map()[%47]
        %49 = affine.min #map4()[%48]
        %50 = affine.max #map()[%49]
        %51 = arith.cmpi slt, %50, %c8 : index
        %52 = arith.ori %51, %23 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%50, %22] [1, 1] : memref<8x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %35[0, 0] [%50, %22] [1, 1] : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        scf.if %52 {
          %collapse_shape_18 = memref.collapse_shape %35 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %cast_19 = memref.cast %collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_19) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %42]
        %cast_3 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_2 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %53 = hivm.hir.pointer_cast(%c1152_i64) : memref<1x32xf32, #hivm.address_space<ub>>
        %54 = hivm.hir.pointer_cast(%c1280_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_5 = memref.cast %35 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %53 : memref<1x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %54 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_5, %cast_6, %cast_7, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %42]
        %collapse_shape = memref.collapse_shape %53 [[0, 1]] : memref<1x32xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        %subview_8 = memref.subview %collapse_shape[0] [%21] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %55 = hivm.hir.pointer_cast(%c2304_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_9 = memref.cast %11 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %55 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_9, %cast_10, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %subview_11 = memref.subview %55[0] [%21] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_12 = memref.cast %subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %subview_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_12, %cast_13, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %56 = hivm.hir.pointer_cast(%c2432_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %57 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_14 = memref.cast %arg15 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_15 = memref.cast %55 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_16 = memref.cast %56 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_17 = memref.cast %57 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_14, %cast_15, %cast_16, %cast_17) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %56 : memref<32xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      %26 = arith.muli %16, %arg8 : i32
      %expand_shape = memref.expand_shape %25 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      %27 = arith.index_cast %26 : i32 to index
      hivm.hir.pipe_barrier[<PIPE_V>]
      scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
        %28 = arith.index_cast %arg14 : i32 to index
        %29 = arith.index_cast %arg8 : i32 to index
        %30 = arith.index_cast %arg13 : i32 to index
        %31 = arith.index_cast %9 : i32 to index
        %32 = affine.apply #map5()[%28, %29, %30, %31]
        %33 = arith.index_cast %32 : index to i64
        %34 = arith.cmpi eq, %33, %c1_i64 : i64
        %35 = arith.select %34, %3, %2 : memref<32x256xf32, #hivm.address_space<ub>>
        %36 = arith.index_cast %arg14 : i32 to index
        %37 = arith.index_cast %arg8 : i32 to index
        %38 = arith.index_cast %arg13 : i32 to index
        %39 = arith.index_cast %9 : i32 to index
        %40 = affine.apply #map5()[%36, %37, %38, %39]
        %41 = arith.index_cast %40 : index to i64
        %42 = arith.cmpi eq, %41, %c1_i64 : i64
        %43 = arith.select %42, %5, %4 : memref<32x256xf32, #hivm.address_space<ub>>
        %44 = arith.index_cast %arg14 : i32 to index
        %45 = arith.index_cast %arg8 : i32 to index
        %46 = arith.index_cast %arg13 : i32 to index
        %47 = arith.index_cast %9 : i32 to index
        %48 = affine.apply #map5()[%44, %45, %46, %47]
        %49 = arith.index_cast %48 : index to i1
        %50 = arith.xori %49, %true : i1
        %51 = arith.extui %50 : i1 to i64
        %52 = arith.subi %arg8, %arg14 : i32
        %53 = arith.index_cast %arg14 : i32 to index
        %54 = affine.apply #map3()[%27, %53]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [32, 256], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %55 = arith.index_cast %52 : i32 to index
        %56 = affine.max #map()[%55]
        %57 = affine.min #map6()[%56]
        %58 = affine.max #map()[%57]
        %59 = arith.cmpi slt, %58, %c256 : index
        %60 = arith.ori %23, %59 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%22, %58] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %43[0, 0] [%22, %58] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %60 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %43 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_14 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %51]
        %cast_3 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %61 = hivm.hir.pointer_cast(%c35328_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_5 = memref.cast %43 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %35 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %61 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_5, %cast_6, %cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%54], sizes: [32, 256], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %35[0, 0] [%22, %58] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%22, %58] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_12 = memref.cast %subview_10 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_12, %cast_13, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %51]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
