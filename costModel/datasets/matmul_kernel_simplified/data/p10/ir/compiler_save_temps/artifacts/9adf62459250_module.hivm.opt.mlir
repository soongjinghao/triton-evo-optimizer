#map = affine_map<()[s0, s1] -> (s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 32, s0)>
#map2 = affine_map<()[s0, s1] -> (32, s0 - s1)>
#map3 = affine_map<()[s0] -> (0, s0)>
#map4 = affine_map<()[s0, s1, s2, s3] -> ((s0 + ((s2 - s3) floordiv 24) * s1) mod 2)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (s0 + ((s2 - s3) floordiv 24) * s1)>
#map6 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map7 = affine_map<()[s0, s1] -> (s0 + s1)>
#map8 = affine_map<()[s0] -> (64, s0)>
#map9 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map10 = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
#map11 = affine_map<()[s0] -> (s0 + 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c20_i8 = arith.constant 20 : i8
    return %c20_i8 : i8
  }
  func.func private @set_l1_2d_float(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c63_i32 = arith.constant 63 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c31_i32 = arith.constant 31 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i64 = arith.constant 0 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %0 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c24576_i64) : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c16384_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %4 = arith.muli %arg12, %arg13 : i32
    %5 = arith.muli %4, %arg14 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg14, %arg13 : i32
    %9 = arith.addi %arg6, %c31_i32 : i32
    %10 = arith.divsi %9, %c32_i32 : i32
    %11 = arith.addi %arg7, %c31_i32 : i32
    %12 = arith.divsi %11, %c32_i32 : i32
    %13 = arith.muli %12, %c8_i32 : i32
    %14 = arith.addi %arg8, %c63_i32 : i32
    %15 = arith.divsi %14, %c64_i32 : i32
    %16 = arith.muli %arg10, %c64_i32 : i32
    %17 = arith.index_cast %arg9 : i32 to index
    %18 = arith.index_cast %arg10 : i32 to index
    %19 = arith.index_cast %arg6 : i32 to index
    %20 = arith.index_cast %arg7 : i32 to index
    %21 = arith.index_cast %16 : i32 to index
    %22 = arith.index_cast %arg11 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg15 = %7 to %5 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %23 = arith.divsi %arg15, %8 : i32
      %24 = arith.remsi %23, %arg12 : i32
      %25 = arith.divsi %24, %13 : i32
      %26 = arith.muli %25, %c8_i32 : i32
      %27 = arith.subi %10, %26 : i32
      %28 = arith.minsi %27, %c8_i32 : i32
      %29 = arith.remsi %24, %28 : i32
      %30 = arith.addi %26, %29 : i32
      %31 = arith.remsi %24, %13 : i32
      %32 = arith.divsi %31, %28 : i32
      %33 = arith.muli %30, %c32_i32 : i32
      %34 = arith.muli %32, %c32_i32 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %33 : i32 to index
      %37 = affine.max #map()[%36, %19]
      %38 = affine.min #map1()[%37, %36]
      %39 = affine.min #map2()[%38, %36]
      %40 = affine.max #map3()[%39]
      %41 = arith.cmpi slt, %40, %c32 : index
      %42 = affine.max #map()[%35, %20]
      %43 = affine.min #map1()[%42, %35]
      %44 = affine.min #map2()[%43, %35]
      %45 = affine.max #map3()[%44]
      %46 = arith.cmpi slt, %45, %c32 : index
      %47 = hivm.hir.pointer_cast(%c0_i64) : memref<2x2x16x16xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %48:2 = scf.for %arg16 = %c0_i32 to %15 step %c1_i32 iter_args(%arg17 = %c0, %arg18 = %c0) -> (index, index)  : i32 {
        %52 = arith.index_cast %arg16 : i32 to index
        %53 = arith.index_cast %15 : i32 to index
        %54 = arith.index_cast %arg15 : i32 to index
        %55 = arith.index_cast %7 : i32 to index
        %56 = affine.apply #map4()[%52, %53, %54, %55]
        %57 = arith.index_cast %56 : index to i64
        %58 = arith.cmpi eq, %57, %c1_i64 : i64
        %59 = arith.select %58, %1, %0 : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
        %60 = arith.index_cast %arg16 : i32 to index
        %61 = arith.index_cast %15 : i32 to index
        %62 = arith.index_cast %arg15 : i32 to index
        %63 = arith.index_cast %7 : i32 to index
        %64 = affine.apply #map4()[%60, %61, %62, %63]
        %65 = arith.index_cast %64 : index to i64
        %66 = arith.cmpi eq, %65, %c1_i64 : i64
        %67 = arith.select %66, %3, %2 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
        %68 = arith.index_cast %arg16 : i32 to index
        %69 = arith.index_cast %15 : i32 to index
        %70 = arith.index_cast %arg15 : i32 to index
        %71 = arith.index_cast %7 : i32 to index
        %72 = affine.apply #map5()[%68, %69, %70, %71]
        %73 = arith.index_cast %72 : index to i64
        %74 = arith.index_cast %arg16 : i32 to index
        %75 = arith.index_cast %15 : i32 to index
        %76 = arith.index_cast %arg15 : i32 to index
        %77 = arith.index_cast %7 : i32 to index
        %78 = affine.apply #map4()[%74, %75, %76, %77]
        %79 = arith.index_cast %78 : index to i1
        %80 = arith.select %79, %c3_i64, %c4_i64 : i64
        %81 = arith.index_cast %78 : index to i1
        %82 = arith.select %81, %c1_i64, %c2_i64 : i64
        %83 = affine.apply #map6()[%arg17, %36, %17]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [32, 64], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %84 = affine.apply #map7()[%arg18, %35]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%84], sizes: [64, 32], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %85 = arith.muli %arg16, %c64_i32 : i32
        %86 = arith.subi %arg8, %85 : i32
        %87 = arith.index_cast %86 : i32 to index
        %88 = affine.max #map3()[%87]
        %89 = affine.min #map8()[%88]
        %90 = affine.max #map3()[%89]
        %91 = arith.cmpi slt, %90, %c64 : index
        %92 = arith.ori %41, %91 : i1
        %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%40, %90] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %93 = affine.apply #map9()[%40]
        %94 = affine.apply #map10()[%90]
        %subview_5 = memref.subview %67[0, 0, 0, 0] [%94, %93, 16, 8] [1, 1, 1, 1] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>>
        scf.if %92 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %67 [[0, 1, 2, 3]] : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          %cast_15 = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_15) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %82]
        %cast_6 = memref.cast %subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_7 = memref.cast %subview_5 : memref<?x?x16x8xf32, strided<[256, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_6, %cast_7) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        %95 = arith.ori %91, %46 : i1
        %subview_8 = memref.subview %reinterpret_cast_3[0, 0] [%90, %45] [1, 1] : memref<64x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %96 = affine.apply #map9()[%90]
        %97 = affine.apply #map10()[%45]
        %subview_9 = memref.subview %59[0, 0, 0, 0] [%97, %96, 16, 8] [1, 1, 1, 1] : memref<4x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
        scf.if %95 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %59 [[0, 1, 2, 3]] : memref<4x4x16x8xf32, #hivm.address_space<cbuf>> into memref<2048xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          %cast_15 = memref.cast %collapse_shape : memref<2048xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_15) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %80]
        %cast_10 = memref.cast %subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_11 = memref.cast %subview_9 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_10, %cast_11) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        %98 = arith.cmpi eq, %arg16, %c0_i32 : i32
        %cast_12 = memref.cast %67 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_13 = memref.cast %59 : memref<4x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_14 = memref.cast %47 : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_12, %cast_13, %98, %40, %90, %45, %cast_14, %c0_i64, %c1_i64, %82, %80, %73, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        %99 = affine.apply #map11()[%arg17]
        %100 = affine.apply #map7()[%arg18, %21]
        scf.yield %99, %100 : index, index
      } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %49 = affine.apply #map6()[%35, %36, %22]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%49], sizes: [32, 32], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %50 = affine.apply #map9()[%40]
      %51 = affine.apply #map9()[%45]
      %subview = memref.subview %47[0, 0, 0, 0] [%51, %50, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>>
      %subview_0 = memref.subview %reinterpret_cast[0, 0] [%40, %45] [1, 1] : memref<32x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %cast = memref.cast %subview : memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      %cast_1 = memref.cast %subview_0 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast, %cast_1, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
