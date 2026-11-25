#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 24) mod 2)>
#map1 = affine_map<()[s0, s1] -> ((s0 - s1) floordiv 24)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map3 = affine_map<()[s0, s1] -> (s1, s0)>
#map4 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map5 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map6 = affine_map<()[s0] -> (0, s0)>
#map7 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map8 = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
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
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c8_i32 = arith.constant 8 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i64 = arith.constant 0 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c49152_i64 = arith.constant 49152 : i64
    %0 = hivm.hir.pointer_cast(%c16384_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c49152_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %4 = arith.muli %arg12, %arg13 : i32
    %5 = arith.muli %4, %arg14 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg14, %arg13 : i32
    %9 = arith.addi %arg6, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %11 = arith.addi %arg7, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %12, %c8_i32 : i32
    %14 = arith.index_cast %arg9 : i32 to index
    %15 = arith.index_cast %arg10 : i32 to index
    %16 = arith.index_cast %arg6 : i32 to index
    %17 = arith.index_cast %arg7 : i32 to index
    %18 = arith.index_cast %arg11 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg15 = %7 to %5 step %c24_i32  : i32 {
      %19 = arith.index_cast %arg15 : i32 to index
      %20 = arith.index_cast %7 : i32 to index
      %21 = affine.apply #map()[%19, %20]
      %22 = arith.index_cast %21 : index to i64
      %23 = arith.cmpi eq, %22, %c1_i64 : i64
      %24 = arith.select %23, %1, %0 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %25 = arith.index_cast %arg15 : i32 to index
      %26 = arith.index_cast %7 : i32 to index
      %27 = affine.apply #map()[%25, %26]
      %28 = arith.index_cast %27 : index to i64
      %29 = arith.cmpi eq, %28, %c1_i64 : i64
      %30 = arith.select %29, %3, %2 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %31 = arith.index_cast %arg15 : i32 to index
      %32 = arith.index_cast %7 : i32 to index
      %33 = affine.apply #map1()[%31, %32]
      %34 = arith.index_cast %33 : index to i64
      %35 = arith.index_cast %arg15 : i32 to index
      %36 = arith.index_cast %7 : i32 to index
      %37 = affine.apply #map()[%35, %36]
      %38 = arith.index_cast %37 : index to i1
      %39 = arith.select %38, %c3_i64, %c4_i64 : i64
      %40 = arith.index_cast %37 : index to i1
      %41 = arith.select %40, %c1_i64, %c2_i64 : i64
      hivm.hir.set_mask_norm
      %42 = arith.divsi %arg15, %arg14 : i32
      %43 = arith.remsi %42, %arg13 : i32
      %44 = arith.divsi %arg15, %8 : i32
      %45 = arith.remsi %44, %arg12 : i32
      %46 = arith.divsi %45, %13 : i32
      %47 = arith.muli %46, %c8_i32 : i32
      %48 = arith.subi %10, %47 : i32
      %49 = arith.minsi %48, %c8_i32 : i32
      %50 = arith.remsi %45, %49 : i32
      %51 = arith.addi %47, %50 : i32
      %52 = arith.remsi %45, %13 : i32
      %53 = arith.divsi %52, %49 : i32
      %54 = arith.muli %51, %c64_i32 : i32
      %55 = arith.muli %53, %c64_i32 : i32
      %56 = arith.muli %43, %c64_i32 : i32
      %57 = arith.index_cast %54 : i32 to index
      %58 = arith.index_cast %56 : i32 to index
      %59 = affine.apply #map2()[%58, %57, %14]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [64, 64], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %60 = arith.index_cast %55 : i32 to index
      %61 = affine.apply #map2()[%60, %58, %15]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [64, 64], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %62 = arith.subi %arg8, %56 : i32
      %63 = affine.max #map3()[%57, %16]
      %64 = affine.min #map4()[%63, %57]
      %65 = arith.index_cast %62 : i32 to index
      %66 = affine.max #map3()[%58, %65]
      %67 = affine.min #map4()[%66, %58]
      %68 = affine.min #map5()[%64, %57]
      %69 = affine.max #map6()[%68]
      %70 = affine.min #map5()[%67, %58]
      %71 = affine.max #map6()[%70]
      %72 = arith.cmpi slt, %69, %c64 : index
      %73 = arith.cmpi slt, %71, %c64 : index
      %74 = arith.ori %72, %73 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%69, %71] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %75 = affine.apply #map7()[%69]
      %76 = affine.apply #map8()[%71]
      %subview_1 = memref.subview %30[0, 0, 0, 0] [%76, %75, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      scf.if %74 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %30 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_15 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @set_l1_2d_float(%cst, %cast_15) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %41]
      %cast = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %subview_1 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @nd2nz_float(%cast, %cast_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %77 = affine.max #map3()[%60, %17]
      %78 = affine.min #map4()[%77, %60]
      %79 = affine.min #map5()[%78, %60]
      %80 = affine.max #map6()[%79]
      %81 = arith.cmpi slt, %80, %c64 : index
      %82 = arith.ori %73, %81 : i1
      %subview_3 = memref.subview %reinterpret_cast_0[0, 0] [%71, %80] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %83 = affine.apply #map7()[%71]
      %84 = affine.apply #map8()[%80]
      %subview_4 = memref.subview %24[0, 0, 0, 0] [%84, %83, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      scf.if %82 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %24 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_15 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @set_l1_2d_float(%cst, %cast_15) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %39]
      %cast_5 = memref.cast %subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_6 = memref.cast %subview_4 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @nd2nz_float(%cast_5, %cast_6) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %85 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %cast_7 = memref.cast %30 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      %cast_8 = memref.cast %24 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      %cast_9 = memref.cast %85 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      func.call @mma_tile_float_to_float(%cast_7, %cast_8, %true, %69, %71, %80, %cast_9, %c0_i64, %c1_i64, %41, %39, %34, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %86 = affine.apply #map2()[%60, %57, %18]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%86], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %87 = affine.apply #map7()[%80]
      %subview_11 = memref.subview %85[0, 0, 0, 0] [%87, %75, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
      %subview_12 = memref.subview %reinterpret_cast_10[0, 0] [%69, %80] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %cast_13 = memref.cast %subview_11 : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      %cast_14 = memref.cast %subview_12 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_13, %cast_14, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
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
