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
  func.func private @enablevc_reduce_sum_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
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
    %c2176_i64 = arith.constant 2176 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c35328_i64 = arith.constant 35328 : i64
    %c69120_i64 = arith.constant 69120 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c72192_i64 = arith.constant 72192 : i64
    %c2560_i64 = arith.constant 2560 : i64
    %c104960_i64 = arith.constant 104960 : i64
    %c36352_i64 = arith.constant 36352 : i64
    %c70144_i64 = arith.constant 70144 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c71168_i64 = arith.constant 71168 : i64
    %0 = hivm.hir.pointer_cast(%c1152_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c71168_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c128_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c70144_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c36352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c104960_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c36352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c104960_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c2560_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c72192_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c2560_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c72192_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg10, %arg11 : i32
    %13 = arith.muli %12, %arg12 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg12, %arg11 : i32
    %17 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
    %cast = memref.cast %17 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    scf.for %arg13 = %15 to %13 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %18 = arith.divsi %arg13, %16 : i32
      %19 = arith.remsi %18, %arg10 : i32
      %20 = arith.muli %19, %c32_i32 : i32
      %21 = arith.subi %arg7, %20 : i32
      %22 = arith.muli %20, %arg9 : i32
      %23 = arith.addi %20, %c32_i32 : i32
      %24 = arith.cmpi sle, %23, %arg7 : i32
      %25 = arith.index_cast %21 : i32 to index
      %26 = affine.max #map()[%25]
      %27 = affine.min #map1()[%26]
      %28 = hivm.hir.pointer_cast(%c2432_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_0 = memref.cast %17 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_1 = memref.cast %28 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_0, %cast_1, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      %29 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %28) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
        %31 = arith.index_cast %arg14 : i32 to index
        %32 = arith.index_cast %arg3 : i32 to index
        %33 = arith.index_cast %arg13 : i32 to index
        %34 = arith.index_cast %15 : i32 to index
        %35 = affine.apply #map2()[%31, %32, %33, %34]
        %36 = arith.index_cast %35 : index to i64
        %37 = arith.cmpi eq, %36, %c1_i64 : i64
        %38 = arith.select %37, %1, %0 : memref<32x8xf32, #hivm.address_space<ub>>
        %39 = arith.index_cast %arg14 : i32 to index
        %40 = arith.index_cast %arg3 : i32 to index
        %41 = arith.index_cast %arg13 : i32 to index
        %42 = arith.index_cast %15 : i32 to index
        %43 = affine.apply #map2()[%39, %40, %41, %42]
        %44 = arith.index_cast %43 : index to i64
        %45 = arith.cmpi eq, %44, %c1_i64 : i64
        %46 = arith.select %45, %3, %2 : memref<32x8xf32, #hivm.address_space<ub>>
        %47 = arith.index_cast %arg14 : i32 to index
        %48 = arith.index_cast %arg3 : i32 to index
        %49 = arith.index_cast %arg13 : i32 to index
        %50 = arith.index_cast %15 : i32 to index
        %51 = affine.apply #map2()[%47, %48, %49, %50]
        %52 = arith.index_cast %51 : index to i1
        %53 = arith.select %52, %c1_i64, %c2_i64 : i64
        %54 = arith.subi %arg3, %arg14 : i32
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %53]
        %55 = arith.select %24, %46, %38 : memref<32x8xf32, #hivm.address_space<ub>>
        scf.if %24 {
          %61 = arith.index_cast %22 : i32 to index
          %62 = arith.index_cast %arg14 : i32 to index
          %63 = affine.apply #map3()[%61, %62]
          %64 = arith.index_cast %arg9 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [32, 8], strides: [%64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %65 = arith.index_cast %54 : i32 to index
          %66 = affine.max #map()[%65]
          %67 = affine.min #map4()[%66]
          %68 = arith.cmpi slt, %67, %c8 : index
          %subview_14 = memref.subview %reinterpret_cast[0, 0] [32, %67] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %46[0, 0] [32, %67] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          scf.if %68 {
            %collapse_shape_18 = memref.collapse_shape %46 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            %cast_19 = memref.cast %collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_19) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_16 = memref.cast %subview_14 : memref<32x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_17 = memref.cast %subview_15 : memref<32x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_16, %cast_17, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        } else {
          %61 = arith.index_cast %22 : i32 to index
          %62 = arith.index_cast %arg14 : i32 to index
          %63 = affine.apply #map3()[%61, %62]
          %64 = arith.index_cast %arg9 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [32, 8], strides: [%64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %65 = arith.index_cast %54 : i32 to index
          %66 = affine.max #map()[%65]
          %67 = affine.max #map()[%27]
          %68 = affine.min #map4()[%66]
          %69 = affine.max #map()[%68]
          %70 = arith.cmpi slt, %67, %c32 : index
          %71 = arith.cmpi slt, %69, %c8 : index
          %72 = arith.ori %70, %71 : i1
          %subview_14 = memref.subview %reinterpret_cast[0, 0] [%67, %69] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %38[0, 0] [%67, %69] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          scf.if %72 {
            %collapse_shape_18 = memref.collapse_shape %38 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            %cast_19 = memref.cast %collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_19) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_16 = memref.cast %subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_17 = memref.cast %subview_15 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_16, %cast_17, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        }
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %56 = hivm.hir.pointer_cast(%c2176_i64) : memref<32x1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %57 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_2 = memref.cast %55 : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_3 = memref.cast %56 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_4 = memref.cast %57 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_sum_ar_float(%cast_2, %cast_3, %cast_4, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %53]
        %collapse_shape = memref.collapse_shape %56 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        %subview = memref.subview %collapse_shape[0] [%27] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %58 = hivm.hir.pointer_cast(%c2304_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_5 = memref.cast %17 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %58 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_5, %cast_6, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %subview_7 = memref.subview %58[0] [%27] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_8 = memref.cast %subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_9 = memref.cast %subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_8, %cast_9, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %59 = hivm.hir.pointer_cast(%c2432_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %60 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_10 = memref.cast %arg15 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_11 = memref.cast %58 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_12 = memref.cast %59 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %60 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_10, %cast_11, %cast_12, %cast_13) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %59 : memref<32xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      %30 = arith.muli %20, %arg8 : i32
      hivm.hir.pipe_barrier[<PIPE_V>]
      scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
        %31 = arith.index_cast %arg14 : i32 to index
        %32 = arith.index_cast %arg8 : i32 to index
        %33 = arith.index_cast %arg13 : i32 to index
        %34 = arith.index_cast %15 : i32 to index
        %35 = affine.apply #map5()[%31, %32, %33, %34]
        %36 = arith.index_cast %35 : index to i64
        %37 = arith.cmpi eq, %36, %c1_i64 : i64
        %38 = arith.select %37, %5, %4 : memref<32x256xf32, #hivm.address_space<ub>>
        %39 = arith.index_cast %arg14 : i32 to index
        %40 = arith.index_cast %arg8 : i32 to index
        %41 = arith.index_cast %arg13 : i32 to index
        %42 = arith.index_cast %15 : i32 to index
        %43 = affine.apply #map5()[%39, %40, %41, %42]
        %44 = arith.index_cast %43 : index to i64
        %45 = arith.cmpi eq, %44, %c1_i64 : i64
        %46 = arith.select %45, %7, %6 : memref<32x256xf32, #hivm.address_space<ub>>
        %47 = arith.index_cast %arg14 : i32 to index
        %48 = arith.index_cast %arg8 : i32 to index
        %49 = arith.index_cast %arg13 : i32 to index
        %50 = arith.index_cast %15 : i32 to index
        %51 = affine.apply #map5()[%47, %48, %49, %50]
        %52 = arith.index_cast %51 : index to i64
        %53 = arith.cmpi eq, %52, %c1_i64 : i64
        %54 = arith.select %53, %9, %8 : memref<32x256xf32, #hivm.address_space<ub>>
        %55 = arith.index_cast %arg14 : i32 to index
        %56 = arith.index_cast %arg8 : i32 to index
        %57 = arith.index_cast %arg13 : i32 to index
        %58 = arith.index_cast %15 : i32 to index
        %59 = affine.apply #map5()[%55, %56, %57, %58]
        %60 = arith.index_cast %59 : index to i64
        %61 = arith.cmpi eq, %60, %c1_i64 : i64
        %62 = arith.select %61, %11, %10 : memref<32x256xf32, #hivm.address_space<ub>>
        %63 = arith.index_cast %arg14 : i32 to index
        %64 = arith.index_cast %arg8 : i32 to index
        %65 = arith.index_cast %arg13 : i32 to index
        %66 = arith.index_cast %15 : i32 to index
        %67 = affine.apply #map5()[%63, %64, %65, %66]
        %68 = arith.index_cast %67 : index to i1
        %69 = arith.xori %68, %true : i1
        %70 = arith.extui %69 : i1 to i64
        %71 = arith.index_cast %67 : index to i1
        %72 = arith.select %71, %c2_i64, %c3_i64 : i64
        %73 = arith.subi %arg8, %arg14 : i32
        scf.if %24 {
          %74 = arith.index_cast %30 : i32 to index
          %75 = arith.index_cast %arg14 : i32 to index
          %76 = affine.apply #map3()[%74, %75]
          %77 = arith.index_cast %arg8 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%76], sizes: [32, 256], strides: [%77, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %78 = arith.index_cast %73 : i32 to index
          %79 = affine.max #map()[%78]
          %80 = affine.min #map6()[%79]
          %81 = arith.cmpi slt, %80, %c256 : index
          %subview = memref.subview %reinterpret_cast[0, 0] [32, %80] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %62[0, 0] [32, %80] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<32x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          scf.if %81 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape = memref.collapse_shape %62 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_14 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %72]
          %cast_3 = memref.cast %subview : memref<32x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_4 = memref.cast %subview_2 : memref<32x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %expand_shape = memref.expand_shape %29 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
          %82 = hivm.hir.pointer_cast(%c35328_i64) : memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_5 = memref.cast %62 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_6 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_7 = memref.cast %54 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_8 = memref.cast %82 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_5, %cast_6, %cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%76], sizes: [32, 256], strides: [%77, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %54[0, 0] [32, %80] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<32x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [32, %80] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_12 = memref.cast %subview_10 : memref<32x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_13 = memref.cast %subview_11 : memref<32x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_2d_float(%cast_12, %cast_13, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %72]
        } else {
          %74 = arith.index_cast %30 : i32 to index
          %75 = arith.index_cast %arg14 : i32 to index
          %76 = affine.apply #map3()[%74, %75]
          %77 = arith.index_cast %arg8 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%76], sizes: [32, 256], strides: [%77, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %78 = arith.index_cast %73 : i32 to index
          %79 = affine.max #map()[%78]
          %80 = affine.max #map()[%27]
          %81 = affine.min #map6()[%79]
          %82 = affine.max #map()[%81]
          %83 = arith.cmpi slt, %80, %c32 : index
          %84 = arith.cmpi slt, %82, %c256 : index
          %85 = arith.ori %83, %84 : i1
          %subview = memref.subview %reinterpret_cast[0, 0] [%80, %82] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %46[0, 0] [%80, %82] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          scf.if %85 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape = memref.collapse_shape %46 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_14 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %70]
          %cast_3 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_4 = memref.cast %subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %expand_shape = memref.expand_shape %29 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
          %86 = hivm.hir.pointer_cast(%c69120_i64) : memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_5 = memref.cast %46 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_6 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_7 = memref.cast %38 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_8 = memref.cast %86 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_5, %cast_6, %cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%76], sizes: [32, 256], strides: [%77, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %38[0, 0] [%80, %82] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%80, %82] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_12 = memref.cast %subview_10 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_13 = memref.cast %subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_2d_float(%cast_12, %cast_13, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %70]
        } {pipeline.veconly}
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
