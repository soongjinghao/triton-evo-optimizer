#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 24) mod 2)>
#map1 = affine_map<()[s0, s1] -> ((s0 - s1) floordiv 24)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
#map3 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map4 = affine_map<()[s0] -> (32, s0)>
#map5 = affine_map<()[s0, s1] -> (32, s1, s0)>
#map6 = affine_map<()[s0, s1] -> (s0 - s1)>
#map7 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map8 = affine_map<()[s0] -> (s0 floordiv 16)>
#map9 = affine_map<()[s0] -> (s0 mod 16)>
#map10 = affine_map<()[s0, s1] -> (s0 + s1 * 512)>
#map11 = affine_map<(d0) -> (d0 * 40960)>
#map12 = affine_map<(d0) -> (d0 * 40960 + 8192)>
#map13 = affine_map<()[s0] -> (s0 mod 2)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 + ((s1 - s2) floordiv 24) * 2)>
#map15 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>
#map16 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>
#map17 = affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128, 0)>
#map18 = affine_map<()[s0] -> (64, s0)>
#map19 = affine_map<()[s0, s1] -> (64, s1, s0)>
#map20 = affine_map<()[s0] -> (s0 * 16)>
#map21 = affine_map<()[s0, s1, s2] -> (s0 * 8 + s1 + s2)>
#map22 = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
#map23 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>
#map24 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>
#map25 = affine_map<()[s0, s1] -> (s1 * 16, s0)>
#map26 = affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>
#map27 = affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>
#map28 = affine_map<()[s0, s1] -> (s0 - s1 * 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @recompute_w_u_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c40960 = arith.constant 40960 : index
    return %c40960 : index
  }
  func.func @recompute_w_u_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @recompute_w_u_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %c24_i32 = arith.constant 24 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %0 = hivm.hir.pointer_cast(%c4096_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c24576_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c20480_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c12288_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    %6 = arith.muli %arg11, %arg12 : i32
    %7 = arith.muli %6, %arg13 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
    %10 = arith.muli %arg13, %arg12 : i32
    %11 = arith.index_cast %arg10 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    scf.for %arg14 = %9 to %7 step %c24_i32  : i32 {
      %12 = arith.index_cast %arg14 : i32 to index
      %13 = arith.index_cast %9 : i32 to index
      %14 = affine.apply #map()[%12, %13]
      %15 = arith.index_cast %14 : index to i64
      %16 = arith.cmpi eq, %15, %c1_i64 : i64
      %17 = arith.select %16, %1, %0 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
      %18 = arith.index_cast %arg14 : i32 to index
      %19 = arith.index_cast %9 : i32 to index
      %20 = affine.apply #map()[%18, %19]
      %21 = arith.index_cast %20 : index to i64
      %22 = arith.cmpi eq, %21, %c1_i64 : i64
      %23 = arith.select %22, %3, %2 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      %24 = arith.index_cast %arg14 : i32 to index
      %25 = arith.index_cast %9 : i32 to index
      %26 = affine.apply #map1()[%24, %25]
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.index_cast %arg14 : i32 to index
      %29 = arith.index_cast %9 : i32 to index
      %30 = affine.apply #map()[%28, %29]
      %31 = arith.index_cast %30 : index to i1
      %32 = arith.xori %31, %true : i1
      %33 = arith.extui %32 : i1 to i64
      hivm.hir.set_mask_norm
      %34 = arith.remsi %arg14, %arg13 : i32
      %35 = arith.divsi %arg14, %arg13 : i32
      %36 = arith.remsi %35, %arg12 : i32
      %37 = arith.divsi %arg14, %10 : i32
      %38 = arith.remsi %37, %arg11 : i32
      %39 = arith.muli %36, %arg10 : i32
      %40 = arith.muli %39, %c8_i32 : i32
      %41 = arith.muli %38, %c32_i32 : i32
      %42 = arith.maxsi %41, %c0_i32 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.addi %40, %34 : i32
      %45 = arith.muli %44, %c32_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = affine.apply #map2()[%46, %43]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%47], sizes: [32, 32], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %48 = affine.max #map3()[%11, %43]
      %49 = affine.min #map4()[%48]
      %50 = arith.subi %c0_i32, %41 : i32
      %51 = arith.maxsi %50, %c0_i32 : i32
      %52 = arith.index_cast %51 : i32 to index
      %53 = affine.min #map5()[%52, %48]
      %54 = affine.apply #map6()[%49, %53]
      %55 = arith.cmpi slt, %54, %c32 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%54, 32] [1, 1] : memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %56 = affine.apply #map7()[%49, %53]
      %57 = affine.apply #map8()[%53]
      %58 = affine.apply #map9()[%53]
      %subview_0 = memref.subview %23[0, %57, %58, 0] [4, %56, 16, 8] [1, 1, 1, 1] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %55 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
        %collapse_shape = memref.collapse_shape %23 [[0, 1, 2, 3]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> into memref<1024xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
        %cast_13 = memref.cast %collapse_shape : memref<1024xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @set_l1_2d_float(%cst, %cast_13) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %33]
      %cast = memref.cast %subview : memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_1 = memref.cast %subview_0 : memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @nd2nz_float(%cast, %cast_1) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %59 = arith.muli %44, %c64_i32 : i32
      %60 = arith.index_cast %59 : i32 to index
      %61 = affine.apply #map10()[%60, %43]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%61], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %62 = hivm.hir.get_block_idx -> i64
      %63 = arith.index_cast %62 : i64 to index
      %64 = affine.apply #map11(%63)
      %view = memref.view %arg2[%64][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_3 = memref.cast %view : memref<32x64xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_4 = memref.cast %17 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @nd2nz_float(%cast_3, %cast_4) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
      %65 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %cast_5 = memref.cast %23 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      %cast_6 = memref.cast %17 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      %cast_7 = memref.cast %65 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      func.call @mma_tile_float_to_float(%cast_5, %cast_6, %true, %54, %c32, %c64, %cast_7, %c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %27, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %subview_8 = memref.subview %65[0, %57, %58, 0] [4, %56, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<4x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_9 = memref.subview %reinterpret_cast_2[0, 0] [%54, 64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %cast_10 = memref.cast %subview_8 : memref<4x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      %cast_11 = memref.cast %subview_9 : memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_10, %cast_11, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      %66 = arith.muli %44, %c128_i32 : i32
      %67 = arith.index_cast %66 : i32 to index
      %68 = hivm.hir.get_block_idx -> i64
      %69 = arith.index_cast %68 : i64 to index
      %70 = affine.apply #map12(%69)
      %view_12 = memref.view %arg2[%70][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
      scf.for %arg15 = %c0 to %c2 step %c1 {
        %71 = affine.apply #map13()[%arg15]
        %72 = arith.index_cast %71 : index to i64
        %73 = arith.cmpi eq, %72, %c1_i64 : i64
        %74 = arith.select %73, %5, %4 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
        %75 = arith.index_cast %arg14 : i32 to index
        %76 = arith.index_cast %9 : i32 to index
        %77 = affine.apply #map14()[%arg15, %75, %76]
        %78 = arith.index_cast %77 : index to i64
        %79 = affine.apply #map13()[%arg15]
        %80 = arith.index_cast %79 : index to i1
        %81 = arith.select %80, %c3_i64, %c4_i64 : i64
        %82 = arith.index_cast %arg15 : index to i32
        %83 = arith.muli %82, %c64_i32 : i32
        %84 = arith.maxsi %83, %c0_i32 : i32
        %85 = arith.index_cast %84 : i32 to index
        %86 = affine.apply #map15()[%85, %67, %43]
        %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%86], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %87 = affine.max #map16()[%11, %85, %43]
        %88 = affine.min #map4()[%87]
        %89 = affine.max #map17()[%85]
        %90 = affine.min #map18()[%89]
        %91 = affine.min #map5()[%52, %87]
        %92 = affine.apply #map6()[%88, %91]
        %93 = arith.subi %c0_i32, %83 : i32
        %94 = arith.maxsi %93, %c0_i32 : i32
        %95 = arith.index_cast %94 : i32 to index
        %96 = affine.min #map19()[%95, %89]
        %97 = affine.apply #map6()[%90, %96]
        %subview_14 = memref.subview %view_12[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %81]
        %cast_15 = memref.cast %subview_14 : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_16 = memref.cast %74 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_15, %cast_16) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
        %98 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        %cast_17 = memref.cast %23 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_18 = memref.cast %74 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_19 = memref.cast %98 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_17, %cast_18, %true, %54, %c32, %c64, %cast_19, %c-1_i64, %c0_i64, %c-1_i64, %81, %78, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %99 = affine.apply #map7()[%88, %91]
        %100 = affine.apply #map7()[%90, %96]
        %101 = affine.apply #map8()[%91]
        %102 = affine.apply #map9()[%91]
        %103 = affine.apply #map8()[%96]
        %104 = affine.apply #map9()[%96]
        %subview_20 = memref.subview %98[%103, %101, %102, %104] [%100, %99, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
        %subview_21 = memref.subview %reinterpret_cast_13[0, 0] [%92, %97] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_22 = memref.cast %subview_20 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_23 = memref.cast %subview_21 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_22, %cast_23, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %33]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func private @set_l1_2d_float(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @recompute_w_u_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c10240_i64 = arith.constant 10240 : i64
    %c9728_i64 = arith.constant 9728 : i64
    %c5632_i64 = arith.constant 5632 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c24_i32 = arith.constant 24 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg11, %arg12 : i32
    %1 = arith.muli %0, %arg13 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg13, %arg12 : i32
    %5 = arith.index_cast %arg10 : i32 to index
    %6 = hivm.hir.get_sub_block_idx -> i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = affine.apply #map20()[%7]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg14 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %9 = arith.remsi %arg14, %arg13 : i32
      %10 = arith.divsi %arg14, %arg13 : i32
      %11 = arith.remsi %10, %arg12 : i32
      %12 = arith.divsi %arg14, %4 : i32
      %13 = arith.remsi %12, %arg11 : i32
      %14 = arith.muli %11, %arg10 : i32
      %15 = arith.muli %14, %c8_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.index_cast %9 : i32 to index
      %18 = arith.muli %13, %c32_i32 : i32
      %19 = arith.maxsi %18, %c0_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = affine.apply #map21()[%20, %16, %17]
      %22 = affine.max #map3()[%5, %20]
      %23 = affine.min #map4()[%22]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [32, 1], strides: [8, 8] : memref<?xf32, #hivm.address_space<gm>> to memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
      %24 = arith.addi %15, %9 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = affine.apply #map22()[%25, %20]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%26], sizes: [32], strides: [8] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %27 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8x1xf32, #hivm.address_space<ub>>
      %subview = memref.subview %27[0, 0, 0] [16, 1, 1] [1, 1, 1] : memref<16x8x1xf32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %28 = arith.subi %c0_i32, %18 : i32
      %29 = arith.maxsi %28, %c0_i32 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = affine.min #map5()[%30, %22]
      %32 = affine.apply #map6()[%23, %31]
      %33 = arith.cmpi slt, %32, %c32 : index
      %34 = affine.min #map23()[%7, %23, %31]
      %35 = affine.min #map24()[%34, %23, %31]
      %subview_1 = memref.subview %reinterpret_cast[%8, 0] [16, 1] [1, 1] : memref<32x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> to memref<16x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %subview_1[0, 0] [%35, 1] [1, 1] : memref<16x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>>
      %36 = affine.max #map25()[%31, %7]
      %37 = affine.min #map26()[%22, %7]
      %38 = affine.max #map27()[%37, %31, %7]
      %39 = affine.apply #map6()[%38, %36]
      %40 = affine.apply #map28()[%36, %7]
      %subview_3 = memref.subview %subview[%40, 0] [%39, 1] [1, 1] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %33 {
        %collapse_shape_38 = memref.collapse_shape %subview [[0, 1]] : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_39, %offset_40, %sizes_41, %strides_42 = memref.extract_strided_metadata %collapse_shape_38 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_43 = memref.reinterpret_cast %base_buffer_39 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_44 = memref.cast %reinterpret_cast_43 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_2d(%cst, %cast_44) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x1xf32, strided<[8, 8], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1]] : memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %collapse_shape : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_5 = memref.cast %collapse_shape_4 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast, %cast_5, %c2_i32, %cst, %31) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %41 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %subview_6 = memref.subview %41[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_0[%8] [16] [1] : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %subview_7[0] [%35] [1] : memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %subview_6[%40] [%39] [1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %33 {
        %base_buffer_38, %offset_39, %sizes_40, %strides_41 = memref.extract_strided_metadata %subview_6 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_42 = memref.reinterpret_cast %base_buffer_38 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_43 = memref.cast %reinterpret_cast_42 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_2d(%cst, %cast_43) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_10 = memref.cast %subview_8 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_11 = memref.cast %subview_9 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_10, %cast_11, %c2_i32, %cst, %31) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %42 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %subview_12 = memref.subview %42[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_6 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %base_buffer_14, %offset_15, %sizes_16, %strides_17 = memref.extract_strided_metadata %subview_12 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_18 = memref.reinterpret_cast %base_buffer_14 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %43 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_19 = memref.cast %reinterpret_cast_13 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %reinterpret_cast_18 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %43 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_2d_float(%cast_19, %cast_20, %cast_21) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %44 = arith.muli %24, %c64_i32 : i32
      %45 = arith.index_cast %44 : i32 to index
      %46 = affine.apply #map10()[%45, %20]
      %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %subview_23 = memref.subview %reinterpret_cast_22[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_24 = memref.subview %subview_23[0, 0] [%35, 64] [1, 1] : memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_25 = memref.subview %47[%40, 0] [%39, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %33 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape_38 = memref.collapse_shape %47 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %cast_39 = memref.cast %collapse_shape_38 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_39) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      %cast_26 = memref.cast %subview_24 : memref<?x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_27 = memref.cast %subview_25 : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_26, %cast_27, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %48 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c5120_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_28 = memref.cast %47 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_29 = memref.cast %subview : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %48 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %49 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_28, %cast_29, %cast_30, %cast_31) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %50 = hivm.hir.get_block_idx -> i64
      %51 = arith.index_cast %50 : i64 to index
      %52 = affine.apply #map11(%51)
      %view = memref.view %arg2[%52][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
      %subview_32 = memref.subview %view[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      %collapse_shape_33 = memref.collapse_shape %48 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %subview_32 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_35 = memref.cast %collapse_shape_33 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_36 = memref.cast %collapse_shape_34 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_35, %cast_36, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
      %53 = arith.muli %24, %c128_i32 : i32
      %54 = arith.index_cast %53 : i32 to index
      %55 = hivm.hir.get_block_idx -> i64
      %56 = arith.index_cast %55 : i64 to index
      %57 = affine.apply #map12(%56)
      %view_37 = memref.view %arg2[%57][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x32x64xf32, #hivm.address_space<gm>>
      %expand_shape = memref.expand_shape %subview_12 [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
      scf.for %arg15 = %c0 to %c2 step %c1 {
        %58 = arith.index_cast %arg15 : index to i32
        %59 = arith.muli %58, %c64_i32 : i32
        %60 = arith.maxsi %59, %c0_i32 : i32
        %61 = arith.index_cast %60 : i32 to index
        %62 = affine.apply #map15()[%61, %54, %20]
        %63 = affine.max #map17()[%61]
        %64 = affine.min #map18()[%63]
        %65 = affine.max #map16()[%5, %61, %20]
        %66 = affine.min #map5()[%30, %65]
        %67 = affine.min #map4()[%65]
        %68 = affine.apply #map6()[%67, %66]
        %reinterpret_cast_38 = memref.reinterpret_cast %arg3 to offset: [%62], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %69 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
        %70 = arith.subi %c0_i32, %59 : i32
        %71 = arith.maxsi %70, %c0_i32 : i32
        %72 = arith.index_cast %71 : i32 to index
        %73 = affine.min #map19()[%72, %63]
        %74 = affine.apply #map6()[%64, %73]
        %75 = arith.cmpi slt, %68, %c32 : index
        %76 = arith.cmpi slt, %74, %c64 : index
        %77 = arith.ori %75, %76 : i1
        %78 = affine.min #map23()[%7, %67, %66]
        %79 = affine.min #map24()[%78, %67, %66]
        %subview_39 = memref.subview %reinterpret_cast_38[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_40 = memref.subview %subview_39[0, 0] [%79, %74] [1, 1] : memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
        %80 = affine.max #map25()[%66, %7]
        %81 = affine.min #map26()[%65, %7]
        %82 = affine.max #map27()[%81, %66, %7]
        %83 = affine.apply #map6()[%82, %80]
        %84 = affine.apply #map28()[%80, %7]
        %subview_41 = memref.subview %69[%84, %73] [%83, %74] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
        scf.if %77 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape_59 = memref.collapse_shape %69 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_60 = memref.cast %collapse_shape_59 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_60) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_42 = memref.cast %subview_40 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_43 = memref.cast %subview_41 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_42, %cast_43, %c2_i32, %cst, %73) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %85 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
        %86 = hivm.hir.pointer_cast(%c9728_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_44 = memref.cast %69 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_45 = memref.cast %subview : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_46 = memref.cast %85 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_47 = memref.cast %86 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_44, %cast_45, %cast_46, %cast_47) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %87 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
        %88 = hivm.hir.pointer_cast(%c10240_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_48 = memref.cast %85 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_49 = memref.cast %expand_shape : memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_50 = memref.cast %87 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_51 = memref.cast %88 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_48, %cast_49, %cast_50, %cast_51) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_52 = memref.subview %view_37[%arg15, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #hivm.address_space<gm>> to memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_53 = memref.collapse_shape %subview_52 [[0, 1], [2]] : memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_54 = memref.subview %collapse_shape_53[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
        %collapse_shape_55 = memref.collapse_shape %87 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
        %collapse_shape_56 = memref.collapse_shape %subview_54 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_57 = memref.cast %collapse_shape_55 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_58 = memref.cast %collapse_shape_56 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_57, %cast_58, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
