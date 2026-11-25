#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (0, s0)>
#map2 = affine_map<()[s0] -> (32, s0)>
#map3 = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 8 + ((s2 - s3) floordiv 48) * ((s1 + 7) floordiv 8)) mod 2)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
#map5 = affine_map<()[s0] -> (8, s0)>
#map6 = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 512 + ((s2 - s3) floordiv 48) * ((s1 + 511) floordiv 512)) mod 2)>
#map7 = affine_map<()[s0] -> (256, s0)>
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
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c165888_i64 = arith.constant 165888 : i64
    %c166272_i64 = arith.constant 166272 : i64
    %c166016_i64 = arith.constant 166016 : i64
    %c166144_i64 = arith.constant 166144 : i64
    %c166400_i64 = arith.constant 166400 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c34816_i64 = arith.constant 34816 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c100352_i64 = arith.constant 100352 : i64
    %c67584_i64 = arith.constant 67584 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c133120_i64 = arith.constant 133120 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c133120_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c133120_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c1024_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c100352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c100352_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c2048_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c34816_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c2048_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c34816_i64) : memref<32x256xf32, #hivm.address_space<ub>>
    %14 = arith.muli %arg10, %arg11 : i32
    %15 = arith.muli %14, %arg12 : i32
    %16 = hivm.hir.get_block_idx -> i64
    %17 = arith.trunci %16 : i64 to i32
    %18 = arith.muli %arg12, %arg11 : i32
    %19 = hivm.hir.pointer_cast(%c165888_i64) : memref<32xf32, #hivm.address_space<ub>>
    %cast = memref.cast %19 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %20 = arith.index_cast %arg9 : i32 to index
    %21 = arith.index_cast %arg8 : i32 to index
    %22 = arith.remsi %arg8, %c512_i32 : i32
    %23 = arith.cmpi ne, %22, %c0_i32 : i32
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg13 = %17 to %15 step %c48_i32  : i32 {
      %24 = arith.index_cast %arg13 : i32 to index
      %25 = arith.index_cast %17 : i32 to index
      %26 = affine.apply #map()[%24, %25]
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.cmpi eq, %27, %c1_i64 : i64
      %29 = arith.select %28, %1, %0 : memref<32x256xf32, #hivm.address_space<ub>>
      %30 = arith.index_cast %arg13 : i32 to index
      %31 = arith.index_cast %17 : i32 to index
      %32 = affine.apply #map()[%30, %31]
      %33 = arith.index_cast %32 : index to i64
      %34 = arith.cmpi eq, %33, %c1_i64 : i64
      %35 = arith.select %34, %3, %2 : memref<32x256xf32, #hivm.address_space<ub>>
      hivm.hir.set_mask_norm
      %36 = arith.divsi %arg13, %18 : i32
      %37 = arith.remsi %36, %arg10 : i32
      %38 = arith.muli %37, %c32_i32 : i32
      %39 = arith.subi %arg7, %38 : i32
      %40 = arith.muli %38, %arg9 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = arith.index_cast %39 : i32 to index
      %43 = affine.max #map1()[%42]
      %44 = affine.min #map2()[%43]
      %45 = affine.max #map1()[%44]
      %46 = arith.cmpi slt, %45, %c32 : index
      %47 = hivm.hir.pointer_cast(%c166272_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_0 = memref.cast %19 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_1 = memref.cast %47 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_0, %cast_1, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %48 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %47) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
        %51 = arith.index_cast %arg14 : i32 to index
        %52 = arith.index_cast %arg3 : i32 to index
        %53 = arith.index_cast %arg13 : i32 to index
        %54 = arith.index_cast %17 : i32 to index
        %55 = affine.apply #map3()[%51, %52, %53, %54]
        %56 = arith.index_cast %55 : index to i64
        %57 = arith.cmpi eq, %56, %c1_i64 : i64
        %58 = arith.select %57, %5, %4 : memref<32x8xf32, #hivm.address_space<ub>>
        %59 = arith.index_cast %arg14 : i32 to index
        %60 = arith.index_cast %arg3 : i32 to index
        %61 = arith.index_cast %arg13 : i32 to index
        %62 = arith.index_cast %17 : i32 to index
        %63 = affine.apply #map3()[%59, %60, %61, %62]
        %64 = arith.index_cast %63 : index to i1
        %65 = arith.select %64, %c2_i64, %c3_i64 : i64
        %66 = arith.subi %arg3, %arg14 : i32
        %67 = arith.index_cast %arg14 : i32 to index
        %68 = affine.apply #map4()[%41, %67]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%68], sizes: [32, 8], strides: [%20, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %69 = arith.index_cast %66 : i32 to index
        %70 = affine.max #map1()[%69]
        %71 = affine.min #map5()[%70]
        %72 = affine.max #map1()[%71]
        %73 = arith.cmpi slt, %72, %c8 : index
        %74 = arith.ori %46, %73 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%45, %72] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %58[0, 0] [%45, %72] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        scf.if %74 {
          %collapse_shape_18 = memref.collapse_shape %58 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %cast_19 = memref.cast %collapse_shape_18 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_19) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %65]
        %cast_3 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_2 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %75 = hivm.hir.pointer_cast(%c166016_i64) : memref<32x1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %76 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_5 = memref.cast %58 : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %75 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %76 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_sum_ar_float(%cast_5, %cast_6, %cast_7, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %65]
        %collapse_shape = memref.collapse_shape %75 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        %subview_8 = memref.subview %collapse_shape[0] [%44] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %77 = hivm.hir.pointer_cast(%c166144_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_9 = memref.cast %19 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %77 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_9, %cast_10, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %subview_11 = memref.subview %77[0] [%44] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_12 = memref.cast %subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %subview_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_float(%cast_12, %cast_13, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        %78 = hivm.hir.pointer_cast(%c166272_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %79 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_14 = memref.cast %arg15 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_15 = memref.cast %77 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_16 = memref.cast %78 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_17 = memref.cast %79 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_14, %cast_15, %cast_16, %cast_17) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %78 : memref<32xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %49 = arith.muli %38, %arg8 : i32
      %expand_shape = memref.expand_shape %48 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      %50 = arith.index_cast %49 : i32 to index
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
      scf.for %arg14 = %c0_i32 to %arg8 step %c512_i32  : i32 {
        %51 = arith.index_cast %arg14 : i32 to index
        %52 = arith.index_cast %arg8 : i32 to index
        %53 = arith.index_cast %arg13 : i32 to index
        %54 = arith.index_cast %17 : i32 to index
        %55 = affine.apply #map6()[%51, %52, %53, %54]
        %56 = arith.index_cast %55 : index to i64
        %57 = arith.cmpi eq, %56, %c1_i64 : i64
        %58 = arith.select %57, %7, %6 : memref<32x256xf32, #hivm.address_space<ub>>
        %59 = arith.index_cast %arg14 : i32 to index
        %60 = arith.index_cast %arg8 : i32 to index
        %61 = arith.index_cast %arg13 : i32 to index
        %62 = arith.index_cast %17 : i32 to index
        %63 = affine.apply #map6()[%59, %60, %61, %62]
        %64 = arith.index_cast %63 : index to i64
        %65 = arith.cmpi eq, %64, %c1_i64 : i64
        %66 = arith.select %65, %9, %8 : memref<32x256xf32, #hivm.address_space<ub>>
        %67 = arith.index_cast %arg14 : i32 to index
        %68 = arith.index_cast %arg8 : i32 to index
        %69 = arith.index_cast %arg13 : i32 to index
        %70 = arith.index_cast %17 : i32 to index
        %71 = affine.apply #map6()[%67, %68, %69, %70]
        %72 = arith.index_cast %71 : index to i64
        %73 = arith.cmpi eq, %72, %c1_i64 : i64
        %74 = arith.select %73, %11, %10 : memref<32x256xf32, #hivm.address_space<ub>>
        %75 = arith.index_cast %arg14 : i32 to index
        %76 = arith.index_cast %arg8 : i32 to index
        %77 = arith.index_cast %arg13 : i32 to index
        %78 = arith.index_cast %17 : i32 to index
        %79 = affine.apply #map6()[%75, %76, %77, %78]
        %80 = arith.index_cast %79 : index to i64
        %81 = arith.cmpi eq, %80, %c1_i64 : i64
        %82 = arith.select %81, %13, %12 : memref<32x256xf32, #hivm.address_space<ub>>
        %83 = arith.index_cast %arg14 : i32 to index
        %84 = arith.index_cast %arg8 : i32 to index
        %85 = arith.index_cast %arg13 : i32 to index
        %86 = arith.index_cast %17 : i32 to index
        %87 = affine.apply #map6()[%83, %84, %85, %86]
        %88 = arith.index_cast %87 : index to i1
        %89 = arith.xori %88, %true : i1
        %90 = arith.extui %89 : i1 to i64
        %91 = arith.index_cast %87 : index to i1
        %92 = arith.select %91, %c2_i64, %c3_i64 : i64
        %93 = arith.subi %arg8, %arg14 : i32
        %94 = arith.index_cast %arg14 : i32 to index
        %95 = affine.apply #map4()[%50, %94]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%95], sizes: [32, 256], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %96 = arith.index_cast %93 : i32 to index
        %97 = affine.max #map1()[%96]
        %98 = affine.min #map7()[%97]
        %99 = affine.max #map1()[%98]
        %100 = arith.cmpi slt, %99, %c256 : index
        %101 = arith.ori %46, %100 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%45, %99] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %82[0, 0] [%45, %99] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %101 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %82 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_14 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %92]
        %cast_3 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %102 = hivm.hir.pointer_cast(%c166400_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_5 = memref.cast %82 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %74 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %102 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_5, %cast_6, %cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%95], sizes: [32, 256], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %74[0, 0] [%45, %99] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%45, %99] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_12 = memref.cast %subview_10 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_12, %cast_13, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %92]
        %103 = arith.addi %arg14, %c256_i32 : i32
        %104 = arith.addi %arg14, %c512_i32 : i32
        %105 = arith.cmpi sle, %104, %arg8 : i32
        scf.if %105 {
          %106 = arith.subi %arg8, %103 : i32
          %107 = arith.index_cast %103 : i32 to index
          %108 = affine.apply #map4()[%50, %107]
          %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%108], sizes: [32, 256], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %109 = arith.index_cast %106 : i32 to index
          %110 = affine.max #map1()[%109]
          %111 = affine.min #map7()[%110]
          %112 = affine.max #map1()[%111]
          %113 = arith.cmpi slt, %112, %c256 : index
          %114 = arith.ori %46, %113 : i1
          %subview_15 = memref.subview %reinterpret_cast_14[0, 0] [%45, %112] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_16 = memref.subview %66[0, 0] [%45, %112] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          scf.if %114 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape = memref.collapse_shape %66 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_28 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_28) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %90]
          %cast_17 = memref.cast %subview_15 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_18 = memref.cast %subview_16 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_17, %cast_18, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %115 = hivm.hir.pointer_cast(%c166400_i64) : memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_19 = memref.cast %66 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_20 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %58 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_22 = memref.cast %115 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_19, %cast_20, %cast_21, %cast_22) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %reinterpret_cast_23 = memref.reinterpret_cast %arg6 to offset: [%108], sizes: [32, 256], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_24 = memref.subview %58[0, 0] [%45, %112] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_25 = memref.subview %reinterpret_cast_23[0, 0] [%45, %112] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_26 = memref.cast %subview_24 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %subview_25 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_2d_float(%cast_26, %cast_27, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %90]
        } {pipeline.veconly}
      } {pipeline.veconly}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      scf.if %23 {
        %51 = arith.divsi %arg8, %c512_i32 : i32
        %52 = arith.muli %51, %c512_i32 : i32
        %53 = arith.subi %arg8, %52 : i32
        %54 = arith.index_cast %52 : i32 to index
        %55 = affine.apply #map4()[%50, %54]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%55], sizes: [32, 256], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %56 = arith.index_cast %53 : i32 to index
        %57 = affine.max #map1()[%56]
        %58 = affine.min #map7()[%57]
        %59 = affine.max #map1()[%58]
        %60 = arith.cmpi slt, %59, %c256 : index
        %61 = arith.ori %46, %60 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%45, %59] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %35[0, 0] [%45, %59] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %61 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %35 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_14 = memref.cast %collapse_shape : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_14) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %cast_3 = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %62 = hivm.hir.pointer_cast(%c166400_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_5 = memref.cast %35 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_6 = memref.cast %expand_shape : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %29 : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_8 = memref.cast %62 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_5, %cast_6, %cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%55], sizes: [32, 256], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %29[0, 0] [%45, %59] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%45, %59] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_12 = memref.cast %subview_10 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_12, %cast_13, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
