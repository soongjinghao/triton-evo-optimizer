#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @experts_combine_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
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
    %c2048_i32 = arith.constant 2048 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_0 = arith.constant 1.41421354 : f32
    %c0 = arith.constant 0 : index
    %c16384_i64 = arith.constant 16384 : i64
    %c40992_i64 = arith.constant 40992 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c49216_i64 = arith.constant 49216 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c41024_i64 = arith.constant 41024 : i64
    %c0_i64 = arith.constant 0 : i64
    %c57408_i64 = arith.constant 57408 : i64
    %c16416_i64 = arith.constant 16416 : i64
    %c65600_i64 = arith.constant 65600 : i64
    %c24608_i64 = arith.constant 24608 : i64
    %c32800_i64 = arith.constant 32800 : i64
    %c73792_i64 = arith.constant 73792 : i64
    %0 = hivm.hir.pointer_cast(%c32800_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c73792_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c24608_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c65600_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c16416_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c0_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c41024_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c8192_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c49216_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c41024_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg6, %arg7 : i32
    %13 = arith.muli %12, %arg8 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg8, %arg7 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    scf.for %arg9 = %15 to %13 step %c48_i32  : i32 {
      %17 = arith.index_cast %arg9 : i32 to index
      %18 = arith.index_cast %15 : i32 to index
      %19 = affine.apply #map()[%17, %18]
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      %22 = arith.select %21, %1, %0 : memref<2048xf32, #hivm.address_space<ub>>
      %23 = arith.index_cast %arg9 : i32 to index
      %24 = arith.index_cast %15 : i32 to index
      %25 = affine.apply #map()[%23, %24]
      %26 = arith.index_cast %25 : index to i64
      %27 = arith.cmpi eq, %26, %c1_i64 : i64
      %28 = arith.select %27, %3, %2 : memref<2048xf32, #hivm.address_space<ub>>
      %29 = arith.index_cast %arg9 : i32 to index
      %30 = arith.index_cast %15 : i32 to index
      %31 = affine.apply #map()[%29, %30]
      %32 = arith.index_cast %31 : index to i64
      %33 = arith.cmpi eq, %32, %c1_i64 : i64
      %34 = arith.select %33, %5, %4 : memref<1x2048xf32, #hivm.address_space<ub>>
      %35 = arith.index_cast %arg9 : i32 to index
      %36 = arith.index_cast %15 : i32 to index
      %37 = affine.apply #map()[%35, %36]
      %38 = arith.index_cast %37 : index to i64
      %39 = arith.cmpi eq, %38, %c1_i64 : i64
      %40 = arith.select %39, %7, %6 : memref<2048xf32, #hivm.address_space<ub>>
      %41 = arith.index_cast %arg9 : i32 to index
      %42 = arith.index_cast %15 : i32 to index
      %43 = affine.apply #map()[%41, %42]
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.cmpi eq, %44, %c1_i64 : i64
      %46 = arith.select %45, %9, %8 : memref<2048xf32, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg9 : i32 to index
      %48 = arith.index_cast %15 : i32 to index
      %49 = affine.apply #map()[%47, %48]
      %50 = arith.index_cast %49 : index to i64
      %51 = arith.cmpi eq, %50, %c1_i64 : i64
      %52 = arith.select %51, %11, %10 : memref<1x2048xf32, #hivm.address_space<ub>>
      %53 = arith.index_cast %arg9 : i32 to index
      %54 = arith.index_cast %15 : i32 to index
      %55 = affine.apply #map()[%53, %54]
      %56 = arith.index_cast %55 : index to i1
      %57 = arith.xori %56, %true : i1
      %58 = arith.extui %57 : i1 to i64
      %59 = arith.index_cast %55 : index to i1
      %60 = arith.select %59, %c2_i64, %c3_i64 : i64
      hivm.hir.set_mask_norm
      %61 = arith.divsi %arg9, %16 : i32
      %62 = arith.remsi %61, %arg6 : i32
      %63 = arith.muli %62, %c2048_i32 : i32
      %64 = arith.muli %62, %c2_i32 : i32
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.muli %65, %c1024_i32 : i32
      %67 = arith.index_cast %63 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[0, 0] [1, 1024] [1, 1] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x1024xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %52[0, 0] [1, 1024] [1, 1] : memref<1x2048xf32, #hivm.address_space<ub>> to memref<1x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %52 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %58]
      %cast = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %collapse_shape_2 = memref.collapse_shape %subview [[0, 1]] : memref<1x1024xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_3 = memref.collapse_shape %subview_1 [[0, 1]] : memref<1x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>> into memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_4 = memref.cast %collapse_shape_2 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_5 = memref.cast %collapse_shape_3 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_4, %cast_5, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%67], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %reinterpret_cast_6[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %46[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_9 = memref.cast %46 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast_9) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_10 = memref.cast %subview_7 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_11 = memref.cast %subview_8 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_10, %cast_11, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %68 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_12 = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %46 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %40 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %68 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_12, %cast_13, %cast_14, %cast_15) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %69 = hivm.hir.pointer_cast(%c16384_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_16 = memref.cast %40 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %40 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %69 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_vs_1d_float(%cast_16, %cst_0, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_20 = memref.subview %40[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_21 = memref.subview %reinterpret_cast_19[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_22 = memref.cast %subview_20 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %subview_21 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_22, %cast_23, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %58]
      %70 = arith.index_cast %66 : i32 to index
      %reinterpret_cast_24 = memref.reinterpret_cast %arg4 to offset: [%70], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_25 = memref.subview %reinterpret_cast_24[0, 0] [1, 1024] [1, 1] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x1024xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_26 = memref.subview %34[0, 0] [1, 1024] [1, 1] : memref<1x2048xf32, #hivm.address_space<ub>> to memref<1x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
      %collapse_shape_27 = memref.collapse_shape %34 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %cast_28 = memref.cast %collapse_shape_27 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast_28) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %collapse_shape_29 = memref.collapse_shape %subview_25 [[0, 1]] : memref<1x1024xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_30 = memref.collapse_shape %subview_26 [[0, 1]] : memref<1x1024xf32, strided<[2048, 1]>, #hivm.address_space<ub>> into memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_31 = memref.cast %collapse_shape_29 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_32 = memref.cast %collapse_shape_30 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_31, %cast_32, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %reinterpret_cast_33 = memref.reinterpret_cast %arg5 to offset: [%70], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_34 = memref.subview %reinterpret_cast_33[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_35 = memref.subview %28[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %28 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_float_to_1d(%cst, %cast_36) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast_37 = memref.cast %subview_34 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_38 = memref.cast %subview_35 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_37, %cast_38, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %71 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_39 = memref.cast %collapse_shape_27 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_40 = memref.cast %28 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %28 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %71 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_39, %cast_40, %cast_41, %cast_42) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %72 = hivm.hir.pointer_cast(%c40992_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %60]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_43 = memref.cast %28 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %22 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_45 = memref.cast %72 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_vs_1d_float(%cast_43, %cst_0, %cast_44, %cast_45) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_46 = memref.reinterpret_cast %arg3 to offset: [%70], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_47 = memref.subview %22[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_48 = memref.subview %reinterpret_cast_46[0] [1024] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_49 = memref.cast %subview_47 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_50 = memref.cast %subview_48 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_49, %cast_50, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %60]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
