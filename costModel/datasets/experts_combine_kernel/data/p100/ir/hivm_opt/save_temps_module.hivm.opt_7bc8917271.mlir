#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @experts_combine_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @experts_combine_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_0 = arith.constant 1.41421354 : f32
    %c0 = arith.constant 0 : index
    %c24576_i64 = arith.constant 24576 : i64
    %c49152_i64 = arith.constant 49152 : i64
    %c73760_i64 = arith.constant 73760 : i64
    %c98336_i64 = arith.constant 98336 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c98368_i64 = arith.constant 98368 : i64
    %c0_i64 = arith.constant 0 : i64
    %c122944_i64 = arith.constant 122944 : i64
    %c40960_i64 = arith.constant 40960 : i64
    %c114752_i64 = arith.constant 114752 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c131136_i64 = arith.constant 131136 : i64
    %c49184_i64 = arith.constant 49184 : i64
    %c155712_i64 = arith.constant 155712 : i64
    %c90144_i64 = arith.constant 90144 : i64
    %c65568_i64 = arith.constant 65568 : i64
    %c147520_i64 = arith.constant 147520 : i64
    %0 = hivm.hir.pointer_cast(%c65568_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c147520_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c90144_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c155712_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c49184_i64) : memref<2x2048xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c131136_i64) : memref<2x2048xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c16384_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c114752_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c40960_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c122944_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<2x2048xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c98368_i64) : memref<2x2048xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c16384_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c114752_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c65568_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c147520_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %16 = arith.muli %arg6, %arg7 : i32
    %17 = arith.muli %16, %arg8 : i32
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.muli %arg8, %arg7 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    scf.for %arg9 = %19 to %17 step %c48_i32  : i32 {
      %21 = arith.index_cast %arg9 : i32 to index
      %22 = arith.index_cast %19 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.cmpi eq, %24, %c1_i64 : i64
      %26 = arith.select %25, %1, %0 : memref<2048xf32, #hivm.address_space<ub>>
      %27 = arith.index_cast %arg9 : i32 to index
      %28 = arith.index_cast %19 : i32 to index
      %29 = affine.apply #map()[%27, %28]
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.cmpi eq, %30, %c1_i64 : i64
      %32 = arith.select %31, %3, %2 : memref<2048xf32, #hivm.address_space<ub>>
      %33 = arith.index_cast %arg9 : i32 to index
      %34 = arith.index_cast %19 : i32 to index
      %35 = affine.apply #map()[%33, %34]
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.cmpi eq, %36, %c1_i64 : i64
      %38 = arith.select %37, %5, %4 : memref<2x2048xf32, #hivm.address_space<ub>>
      %39 = arith.index_cast %arg9 : i32 to index
      %40 = arith.index_cast %19 : i32 to index
      %41 = affine.apply #map()[%39, %40]
      %42 = arith.index_cast %41 : index to i64
      %43 = arith.cmpi eq, %42, %c1_i64 : i64
      %44 = arith.select %43, %7, %6 : memref<2048xf32, #hivm.address_space<ub>>
      %45 = arith.index_cast %arg9 : i32 to index
      %46 = arith.index_cast %19 : i32 to index
      %47 = affine.apply #map()[%45, %46]
      %48 = arith.index_cast %47 : index to i64
      %49 = arith.cmpi eq, %48, %c1_i64 : i64
      %50 = arith.select %49, %9, %8 : memref<2048xf32, #hivm.address_space<ub>>
      %51 = arith.index_cast %arg9 : i32 to index
      %52 = arith.index_cast %19 : i32 to index
      %53 = affine.apply #map()[%51, %52]
      %54 = arith.index_cast %53 : index to i64
      %55 = arith.cmpi eq, %54, %c1_i64 : i64
      %56 = arith.select %55, %11, %10 : memref<2x2048xf32, #hivm.address_space<ub>>
      %57 = arith.index_cast %arg9 : i32 to index
      %58 = arith.index_cast %19 : i32 to index
      %59 = affine.apply #map()[%57, %58]
      %60 = arith.index_cast %59 : index to i64
      %61 = arith.cmpi eq, %60, %c1_i64 : i64
      %62 = arith.select %61, %13, %12 : memref<1x2048xf32, #hivm.address_space<ub>>
      %63 = arith.index_cast %arg9 : i32 to index
      %64 = arith.index_cast %19 : i32 to index
      %65 = affine.apply #map()[%63, %64]
      %66 = arith.index_cast %65 : index to i64
      %67 = arith.cmpi eq, %66, %c1_i64 : i64
      %68 = arith.select %67, %15, %14 : memref<1x2048xf32, #hivm.address_space<ub>>
      %69 = arith.index_cast %arg9 : i32 to index
      %70 = arith.index_cast %19 : i32 to index
      %71 = affine.apply #map()[%69, %70]
      %72 = arith.index_cast %71 : index to i1
      %73 = arith.select %72, %c2_i64, %c3_i64 : i64
      %74 = arith.index_cast %71 : index to i1
      %75 = arith.xori %74, %true : i1
      %76 = arith.extui %75 : i1 to i64
      hivm.hir.set_mask_norm
      %77 = arith.divsi %arg9, %20 : i32
      %78 = arith.remsi %77, %arg6 : i32
      %79 = arith.muli %78, %c2048_i32 : i32
      %80 = arith.muli %78, %c2_i32 : i32
      %81 = arith.addi %80, %c1_i32 : i32
      %82 = arith.muli %81, %c1024_i32 : i32
      %83 = arith.muli %78, %c4096_i32 : i32
      %84 = arith.index_cast %83 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%84], sizes: [2, 2048], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x2048xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[0, 0] [2, 1024] [1, 1] : memref<2x2048xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %56[0, 0] [2, 1024] [1, 1] : memref<2x2048xf32, #hivm.address_space<ub>> to memref<2x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %56 [[0, 1]] : memref<2x2048xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %cast = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_2 = memref.cast %subview : memref<2x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_3 = memref.cast %subview_1 : memref<2x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_2, %cast_3, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %85 = hivm.hir.pointer_cast(%c24576_i64) : memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %76]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_4 = memref.cast %56 : memref<2x2048xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %62 : memref<1x2048xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %85 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_4, %cast_5, %cast_6, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_7 = memref.collapse_shape %62 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %86 = arith.index_cast %79 : i32 to index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg5 to offset: [%86], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %reinterpret_cast_8[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %50[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %50 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast_11) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_12 = memref.cast %subview_9 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_13 = memref.cast %subview_10 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_12, %cast_13, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %87 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_14 = memref.cast %collapse_shape_7 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %50 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %44 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %87 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_14, %cast_15, %cast_16, %cast_17) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %88 = hivm.hir.pointer_cast(%c49152_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_18 = memref.cast %44 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %44 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %88 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_vs_1d_float(%cast_18, %cst_0, %cast_19, %cast_20) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_22 = memref.subview %44[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_23 = memref.subview %reinterpret_cast_21[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_24 = memref.cast %subview_22 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_25 = memref.cast %subview_23 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_24, %cast_25, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %76]
      %89 = arith.muli %81, %c2048_i32 : i32
      %90 = arith.index_cast %89 : i32 to index
      %reinterpret_cast_26 = memref.reinterpret_cast %arg4 to offset: [%90], sizes: [2, 2048], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x2048xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_27 = memref.subview %reinterpret_cast_26[0, 0] [2, 1024] [1, 1] : memref<2x2048xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %38[0, 0] [2, 1024] [1, 1] : memref<2x2048xf32, #hivm.address_space<ub>> to memref<2x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
      %collapse_shape_29 = memref.collapse_shape %38 [[0, 1]] : memref<2x2048xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %cast_30 = memref.cast %collapse_shape_29 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast_30) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_31 = memref.cast %subview_27 : memref<2x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_32 = memref.cast %subview_28 : memref<2x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_31, %cast_32, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %91 = hivm.hir.pointer_cast(%c73760_i64) : memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %73]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_33 = memref.cast %38 : memref<2x2048xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %68 : memref<1x2048xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %91 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_33, %cast_34, %cast_35, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_36 = memref.collapse_shape %68 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %92 = arith.index_cast %82 : i32 to index
      %reinterpret_cast_37 = memref.reinterpret_cast %arg5 to offset: [%92], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_38 = memref.subview %reinterpret_cast_37[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_39 = memref.subview %32[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %32 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast_40) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_41 = memref.cast %subview_38 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_42 = memref.cast %subview_39 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_41, %cast_42, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %93 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_43 = memref.cast %collapse_shape_36 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %32 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %26 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %93 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_43, %cast_44, %cast_45, %cast_46) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %94 = hivm.hir.pointer_cast(%c98336_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_47 = memref.cast %26 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_48 = memref.cast %26 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %94 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_vs_1d_float(%cast_47, %cst_0, %cast_48, %cast_49) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_50 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_51 = memref.subview %26[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_52 = memref.subview %reinterpret_cast_50[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_53 = memref.cast %subview_51 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_54 = memref.cast %subview_52 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_53, %cast_54, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %73]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
