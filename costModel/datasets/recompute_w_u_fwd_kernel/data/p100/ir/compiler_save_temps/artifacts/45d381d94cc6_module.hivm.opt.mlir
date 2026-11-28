#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 24) mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
#map2 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map3 = affine_map<()[s0] -> (32, s0)>
#map4 = affine_map<()[s0, s1] -> (32, s1, s0)>
#map5 = affine_map<()[s0, s1] -> (s0 - s1)>
#map6 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map7 = affine_map<()[s0] -> (s0 floordiv 16)>
#map8 = affine_map<()[s0] -> (s0 mod 16)>
#map9 = affine_map<()[s0] -> (s0 mod 2)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 + ((s1 - s2) floordiv 24) * 2)>
#map11 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 512)>
#map12 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 512, 0)>
#map13 = affine_map<()[s0] -> (-s0 + (s0 floordiv 512) * 512 + 64, 0)>
#map14 = affine_map<()[s0] -> (64, s0)>
#map15 = affine_map<()[s0, s1] -> (64, s1, s0)>
#map16 = affine_map<(d0) -> (d0 * 16384)>
#map17 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1024)>
#map18 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 1024, 0)>
#map19 = affine_map<()[s0] -> (-s0 + (s0 floordiv 1024) * 1024 + 128, 0)>
#map20 = affine_map<(d0) -> (d0 * 16384 + 8192)>
#map21 = affine_map<()[s0] -> (s0 * 16)>
#map22 = affine_map<()[s0, s1, s2] -> (s0 * 8 + s1 + s2)>
#map23 = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
#map24 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 16)>
#map25 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 16)>
#map26 = affine_map<()[s0, s1] -> (s1 * 16, s0)>
#map27 = affine_map<()[s0, s1] -> (32, s1 * 16 + 16, s0)>
#map28 = affine_map<()[s0, s1, s2] -> (s2 * 16, s1, s0)>
#map29 = affine_map<()[s0, s1] -> (s0 - s1 * 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @recompute_w_u_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c16384 = arith.constant 16384 : index
    return %c16384 : index
  }
  func.func @recompute_w_u_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @recompute_w_u_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c24_i32 = arith.constant 24 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c0_i64 = arith.constant 0 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c20480_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c12288_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c4096_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c24576_i64) : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    %6 = arith.muli %arg11, %arg12 : i32
    %7 = arith.muli %6, %arg13 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
    %10 = arith.muli %arg13, %arg12 : i32
    %11 = arith.index_cast %arg10 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    scf.for %arg14 = %9 to %7 step %c24_i32  : i32 {
      %12 = arith.index_cast %arg14 : i32 to index
      %13 = arith.index_cast %9 : i32 to index
      %14 = affine.apply #map()[%12, %13]
      %15 = arith.index_cast %14 : index to i64
      %16 = arith.cmpi eq, %15, %c1_i64 : i64
      %17 = arith.select %16, %1, %0 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      %18 = arith.index_cast %arg14 : i32 to index
      %19 = arith.index_cast %9 : i32 to index
      %20 = affine.apply #map()[%18, %19]
      %21 = arith.index_cast %20 : index to i1
      %22 = arith.xori %21, %true : i1
      %23 = arith.extui %22 : i1 to i64
      hivm.hir.set_mask_norm
      %24 = arith.divsi %arg14, %arg13 : i32
      %25 = arith.remsi %24, %arg12 : i32
      %26 = arith.divsi %arg14, %10 : i32
      %27 = arith.remsi %26, %arg11 : i32
      %28 = arith.divsi %25, %c8_i32 : i32
      %29 = arith.remsi %25, %c8_i32 : i32
      %30 = arith.muli %28, %arg10 : i32
      %31 = arith.muli %30, %c8_i32 : i32
      %32 = arith.addi %31, %29 : i32
      %33 = arith.muli %32, %c64_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.muli %32, %c128_i32 : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.muli %27, %c32_i32 : i32
      %38 = arith.maxsi %37, %c0_i32 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.muli %32, %c32_i32 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = affine.apply #map1()[%41, %39]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%42], sizes: [32, 32], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %43 = affine.max #map2()[%11, %39]
      %44 = affine.min #map3()[%43]
      %45 = arith.subi %c0_i32, %37 : i32
      %46 = arith.maxsi %45, %c0_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = affine.min #map4()[%47, %43]
      %49 = affine.apply #map5()[%44, %48]
      %50 = arith.cmpi slt, %49, %c32 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%49, 32] [1, 1] : memref<32x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %51 = affine.apply #map6()[%44, %48]
      %52 = affine.apply #map7()[%48]
      %53 = affine.apply #map8()[%48]
      %subview_0 = memref.subview %17[0, %52, %53, 0] [4, %51, 16, 8] [1, 1, 1, 1] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %50 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
        %collapse_shape = memref.collapse_shape %17 [[0, 1, 2, 3]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> into memref<1024xf32, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
        %cast_2 = memref.cast %collapse_shape : memref<1024xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @set_l1_2d_float(%cst, %cast_2) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %23]
      %cast = memref.cast %subview : memref<?x32xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_1 = memref.cast %subview_0 : memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
      func.call @nd2nz_float(%cast, %cast_1) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
      scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %54 = arith.index_cast %arg15 : i32 to index
        %55 = affine.apply #map9()[%54]
        %56 = arith.index_cast %55 : index to i64
        %57 = arith.cmpi eq, %56, %c1_i64 : i64
        %58 = arith.select %57, %3, %2 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
        %59 = arith.index_cast %arg15 : i32 to index
        %60 = affine.apply #map9()[%59]
        %61 = arith.index_cast %60 : index to i64
        %62 = arith.cmpi eq, %61, %c1_i64 : i64
        %63 = arith.select %62, %5, %4 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>
        %64 = arith.index_cast %arg15 : i32 to index
        %65 = arith.index_cast %arg14 : i32 to index
        %66 = arith.index_cast %9 : i32 to index
        %67 = affine.apply #map10()[%64, %65, %66]
        %68 = arith.index_cast %67 : index to i64
        %69 = arith.index_cast %arg15 : i32 to index
        %70 = affine.apply #map9()[%69]
        %71 = arith.index_cast %70 : index to i1
        %72 = arith.select %71, %c2_i64, %c3_i64 : i64
        %73 = arith.index_cast %70 : index to i1
        %74 = arith.select %73, %c4_i64, %c5_i64 : i64
        %75 = arith.cmpi slt, %arg15, %c1_i32 : i32
        scf.if %75 {
          %77 = arith.muli %arg15, %c64_i32 : i32
          %78 = arith.maxsi %77, %c0_i32 : i32
          %79 = arith.index_cast %78 : i32 to index
          %80 = affine.apply #map11()[%79, %34, %39]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%80], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %81 = affine.max #map12()[%11, %79, %39]
          %82 = affine.min #map3()[%81]
          %83 = affine.max #map13()[%79]
          %84 = affine.min #map14()[%83]
          %85 = affine.min #map4()[%47, %81]
          %86 = affine.apply #map5()[%82, %85]
          %87 = arith.subi %c0_i32, %77 : i32
          %88 = arith.maxsi %87, %c0_i32 : i32
          %89 = arith.index_cast %88 : i32 to index
          %90 = affine.min #map15()[%89, %83]
          %91 = affine.apply #map5()[%84, %90]
          %92 = hivm.hir.get_block_idx -> i64
          %93 = arith.index_cast %92 : i64 to index
          %94 = affine.apply #map16(%93)
          %view = memref.view %arg2[%94][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 2
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %74]
          %cast_3 = memref.cast %view : memref<32x64xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_4 = memref.cast %63 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_float(%cast_3, %cast_4) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
          %95 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
          %cast_5 = memref.cast %17 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_6 = memref.cast %63 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_7 = memref.cast %95 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          func.call @mma_tile_float_to_float(%cast_5, %cast_6, %true, %49, %c32, %c64, %cast_7, %c-1_i64, %c0_i64, %c-1_i64, %74, %68, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %96 = affine.apply #map6()[%82, %85]
          %97 = affine.apply #map6()[%84, %90]
          %98 = affine.apply #map7()[%85]
          %99 = affine.apply #map8()[%85]
          %100 = affine.apply #map7()[%90]
          %101 = affine.apply #map8()[%90]
          %subview_8 = memref.subview %95[%100, %98, %99, %101] [%97, %96, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
          %subview_9 = memref.subview %reinterpret_cast_2[0, 0] [%86, %91] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %cast_10 = memref.cast %subview_8 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          %cast_11 = memref.cast %subview_9 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_10, %cast_11, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        }
        %76 = arith.cmpi slt, %arg15, %c2_i32 : i32
        scf.if %76 {
          %77 = arith.muli %arg15, %c64_i32 : i32
          %78 = arith.maxsi %77, %c0_i32 : i32
          %79 = arith.index_cast %78 : i32 to index
          %80 = affine.apply #map17()[%79, %36, %39]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          %81 = affine.max #map18()[%11, %79, %39]
          %82 = affine.min #map3()[%81]
          %83 = affine.max #map19()[%79]
          %84 = affine.min #map14()[%83]
          %85 = affine.min #map4()[%47, %81]
          %86 = affine.apply #map5()[%82, %85]
          %87 = arith.subi %c0_i32, %77 : i32
          %88 = arith.maxsi %87, %c0_i32 : i32
          %89 = arith.index_cast %88 : i32 to index
          %90 = affine.min #map15()[%89, %83]
          %91 = affine.apply #map5()[%84, %90]
          %92 = hivm.hir.get_block_idx -> i64
          %93 = arith.index_cast %92 : i64 to index
          %94 = affine.apply #map20(%93)
          %view = memref.view %arg2[%94][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 2
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %72]
          %cast_3 = memref.cast %view : memref<32x64xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_4 = memref.cast %58 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_float(%cast_3, %cast_4) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
          %95 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
          %cast_5 = memref.cast %17 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_6 = memref.cast %58 : memref<8x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_7 = memref.cast %95 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          func.call @mma_tile_float_to_float(%cast_5, %cast_6, %true, %49, %c32, %c64, %cast_7, %c-1_i64, %c0_i64, %c-1_i64, %72, %68, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %96 = affine.apply #map6()[%82, %85]
          %97 = affine.apply #map6()[%84, %90]
          %98 = affine.apply #map7()[%85]
          %99 = affine.apply #map8()[%85]
          %100 = affine.apply #map7()[%90]
          %101 = affine.apply #map8()[%90]
          %subview_8 = memref.subview %95[%100, %98, %99, %101] [%97, %96, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
          %subview_9 = memref.subview %reinterpret_cast_2[0, 0] [%86, %91] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          %cast_10 = memref.cast %subview_8 : memref<?x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          %cast_11 = memref.cast %subview_9 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_10, %cast_11, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
          hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        }
      } {fixpipe_for_mmad_result_already_inserted = true}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %23]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
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
    %c10240_i64 = arith.constant 10240 : i64
    %c9728_i64 = arith.constant 9728 : i64
    %c5632_i64 = arith.constant 5632 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c24_i32 = arith.constant 24 : i32
    %0 = arith.muli %arg11, %arg12 : i32
    %1 = arith.muli %0, %arg13 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg13, %arg12 : i32
    %5 = arith.index_cast %arg10 : i32 to index
    %6 = hivm.hir.get_sub_block_idx -> i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = affine.apply #map21()[%7]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg14 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %9 = arith.divsi %arg14, %arg13 : i32
      %10 = arith.remsi %9, %arg12 : i32
      %11 = arith.divsi %arg14, %4 : i32
      %12 = arith.remsi %11, %arg11 : i32
      %13 = arith.divsi %10, %c8_i32 : i32
      %14 = arith.remsi %10, %c8_i32 : i32
      %15 = arith.muli %13, %arg10 : i32
      %16 = arith.muli %15, %c8_i32 : i32
      %17 = arith.addi %16, %14 : i32
      %18 = arith.muli %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.muli %17, %c128_i32 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %16 : i32 to index
      %23 = arith.index_cast %14 : i32 to index
      %24 = arith.muli %12, %c32_i32 : i32
      %25 = arith.maxsi %24, %c0_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = affine.apply #map22()[%26, %22, %23]
      %28 = affine.max #map2()[%5, %26]
      %29 = affine.min #map3()[%28]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [32], strides: [8] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %30 = arith.index_cast %17 : i32 to index
      %31 = affine.apply #map23()[%30, %26]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg9 to offset: [%31], sizes: [32], strides: [8] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %32 = hivm.hir.pointer_cast(%c0_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %subview = memref.subview %32[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
      %33 = arith.subi %c0_i32, %24 : i32
      %34 = arith.maxsi %33, %c0_i32 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = affine.min #map4()[%35, %28]
      %37 = affine.apply #map5()[%29, %36]
      %38 = arith.cmpi slt, %37, %c32 : index
      %39 = affine.min #map24()[%7, %29, %36]
      %40 = affine.min #map25()[%39, %29, %36]
      %subview_1 = memref.subview %reinterpret_cast[%8] [16] [1] : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %subview_1[0] [%40] [1] : memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %41 = affine.max #map26()[%36, %7]
      %42 = affine.min #map27()[%28, %7]
      %43 = affine.max #map28()[%42, %36, %7]
      %44 = affine.apply #map5()[%43, %41]
      %45 = affine.apply #map29()[%41, %7]
      %subview_3 = memref.subview %subview[%45] [%44] [1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %38 {
        %base_buffer_21, %offset_22, %sizes_23, %strides_24 = memref.extract_strided_metadata %subview : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_25 = memref.reinterpret_cast %base_buffer_21 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_26 = memref.cast %reinterpret_cast_25 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_2d(%cst, %cast_26) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %subview_2 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_4 = memref.cast %subview_3 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast, %cast_4, %c2_i32, %cst, %36) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      %46 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %subview_5 = memref.subview %46[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_0[%8] [16] [1] : memref<32xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %subview_6[0] [%40] [1] : memref<16xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %subview_5[%45] [%44] [1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %38 {
        %base_buffer_21, %offset_22, %sizes_23, %strides_24 = memref.extract_strided_metadata %subview_5 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_25 = memref.reinterpret_cast %base_buffer_21 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_26 = memref.cast %reinterpret_cast_25 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_2d(%cst, %cast_26) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_9 = memref.cast %subview_7 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_10 = memref.cast %subview_8 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_9, %cast_10, %c2_i32, %cst, %36) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %47 = hivm.hir.pointer_cast(%c512_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %subview_11 = memref.subview %47[0, 0] [16, 1] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_5 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %subview_11 : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [16, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %48 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_18 = memref.cast %reinterpret_cast_12 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %reinterpret_cast_17 : memref<16x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %48 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_2d_float(%cast_18, %cast_19, %cast_20) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %49 = arith.cmpi slt, %arg15, %c1_i32 : i32
        scf.if %49 {
          %51 = arith.muli %arg15, %c64_i32 : i32
          %52 = arith.maxsi %51, %c0_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = affine.apply #map11()[%53, %19, %26]
          %55 = affine.max #map13()[%53]
          %56 = affine.min #map14()[%55]
          %57 = affine.max #map12()[%5, %53, %26]
          %58 = affine.min #map4()[%35, %57]
          %59 = affine.min #map3()[%57]
          %60 = affine.apply #map5()[%59, %58]
          %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [32, 64], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %61 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
          %62 = arith.subi %c0_i32, %51 : i32
          %63 = arith.maxsi %62, %c0_i32 : i32
          %64 = arith.index_cast %63 : i32 to index
          %65 = affine.min #map15()[%64, %55]
          %66 = affine.apply #map5()[%56, %65]
          %67 = arith.cmpi slt, %60, %c32 : index
          %68 = arith.cmpi slt, %66, %c64 : index
          %69 = arith.ori %67, %68 : i1
          %70 = affine.min #map24()[%7, %59, %58]
          %71 = affine.min #map25()[%70, %59, %58]
          %subview_22 = memref.subview %reinterpret_cast_21[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_23 = memref.subview %subview_22[0, 0] [%71, %66] [1, 1] : memref<16x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %72 = affine.max #map26()[%58, %7]
          %73 = affine.min #map27()[%57, %7]
          %74 = affine.max #map28()[%73, %58, %7]
          %75 = affine.apply #map5()[%74, %72]
          %76 = affine.apply #map29()[%72, %7]
          %subview_24 = memref.subview %61[%76, %65] [%75, %66] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
          scf.if %69 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_35 = memref.collapse_shape %61 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_36 = memref.cast %collapse_shape_35 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_36) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
          %cast_25 = memref.cast %subview_23 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_26 = memref.cast %subview_24 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_25, %cast_26, %c2_i32, %cst, %65) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
          %77 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x64xf32, #hivm.address_space<ub>>
          %78 = hivm.hir.pointer_cast(%c5120_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_27 = memref.cast %61 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_28 = memref.cast %expand_shape : memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %77 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %78 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_27, %cast_28, %cast_29, %cast_30) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %79 = hivm.hir.get_block_idx -> i64
          %80 = arith.index_cast %79 : i64 to index
          %81 = affine.apply #map16(%80)
          %view = memref.view %arg2[%81][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
          %subview_31 = memref.subview %view[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
          %collapse_shape = memref.collapse_shape %77 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
          %collapse_shape_32 = memref.collapse_shape %subview_31 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_33 = memref.cast %collapse_shape : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_34 = memref.cast %collapse_shape_32 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_33, %cast_34, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
        }
        %50 = arith.cmpi slt, %arg15, %c2_i32 : i32
        scf.if %50 {
          %51 = arith.muli %arg15, %c64_i32 : i32
          %52 = arith.maxsi %51, %c0_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = affine.apply #map17()[%53, %21, %26]
          %55 = affine.max #map19()[%53]
          %56 = affine.min #map14()[%55]
          %57 = affine.max #map18()[%5, %53, %26]
          %58 = affine.min #map4()[%35, %57]
          %59 = affine.min #map3()[%57]
          %60 = affine.apply #map5()[%59, %58]
          %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [32, 64], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          %61 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
          %62 = arith.subi %c0_i32, %51 : i32
          %63 = arith.maxsi %62, %c0_i32 : i32
          %64 = arith.index_cast %63 : i32 to index
          %65 = affine.min #map15()[%64, %55]
          %66 = affine.apply #map5()[%56, %65]
          %67 = arith.cmpi slt, %60, %c32 : index
          %68 = arith.cmpi slt, %66, %c64 : index
          %69 = arith.ori %67, %68 : i1
          %70 = affine.min #map24()[%7, %59, %58]
          %71 = affine.min #map25()[%70, %59, %58]
          %subview_22 = memref.subview %reinterpret_cast_21[%8, 0] [16, 64] [1, 1] : memref<32x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_23 = memref.subview %subview_22[0, 0] [%71, %66] [1, 1] : memref<16x64xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
          %72 = affine.max #map26()[%58, %7]
          %73 = affine.min #map27()[%57, %7]
          %74 = affine.max #map28()[%73, %58, %7]
          %75 = affine.apply #map5()[%74, %72]
          %76 = affine.apply #map29()[%72, %7]
          %subview_24 = memref.subview %61[%76, %65] [%75, %66] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
          scf.if %69 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_40 = memref.collapse_shape %61 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_41 = memref.cast %collapse_shape_40 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_41) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          %cast_25 = memref.cast %subview_23 : memref<?x?xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_26 = memref.cast %subview_24 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast_25, %cast_26, %c2_i32, %cst, %65) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
          %77 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
          %78 = hivm.hir.pointer_cast(%c9728_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_27 = memref.cast %61 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_28 = memref.cast %expand_shape : memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %77 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %78 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_27, %cast_28, %cast_29, %cast_30) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %expand_shape_31 = memref.expand_shape %subview_11 [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[8]>, #hivm.address_space<ub>> into memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
          %79 = hivm.hir.pointer_cast(%c5632_i64) : memref<16x64xf32, #hivm.address_space<ub>>
          %80 = hivm.hir.pointer_cast(%c10240_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_32 = memref.cast %77 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %expand_shape_31 : memref<16x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_34 = memref.cast %79 : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %80 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_32, %cast_33, %cast_34, %cast_35) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %81 = hivm.hir.get_block_idx -> i64
          %82 = arith.index_cast %81 : i64 to index
          %83 = affine.apply #map20(%82)
          %view = memref.view %arg2[%83][] : memref<?xi8, #hivm.address_space<gm>> to memref<32x64xf32, #hivm.address_space<gm>>
          %subview_36 = memref.subview %view[%8, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
          %collapse_shape = memref.collapse_shape %79 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
          %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1]] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_38 = memref.cast %collapse_shape : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_39 = memref.cast %collapse_shape_37 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_38, %cast_39, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
        }
      } {fixpipe_for_mmad_result_already_inserted = true}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
