#map = affine_map<()[s0] -> (s0 * 32768 + 16384)>
#map1 = affine_map<()[s0] -> (s0 * 32768)>
#map2 = affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>
#map3 = affine_map<()[s0, s1, s2, s3] -> ((s0 + ((s2 - s3) floordiv 96) * s1) mod 2)>
#map4 = affine_map<()[s0, s1, s2, s3] -> (s0 + ((s2 - s3) floordiv 96) * s1)>
#map5 = affine_map<()[s0] -> (s0 + 4)>
#map6 = affine_map<(d0)[s0] -> (d0 * 24 + s0)>
#map7 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map8 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map9 = affine_map<()[s0] -> (16, s0)>
#map10 = affine_map<()[s0, s1] -> (16, s1, s0)>
#map11 = affine_map<()[s0, s1] -> (s0 - s1)>
#map12 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map13 = affine_map<()[s0] -> (s0 floordiv 16)>
#map14 = affine_map<()[s0] -> (s0 mod 16)>
#map15 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map16 = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
#map17 = affine_map<()[s0] -> (s0 * 8)>
#map18 = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 + s2)>
#map19 = affine_map<()[s0, s1] -> (s0 + s1 * 4)>
#map20 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>
#map21 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>
#map22 = affine_map<()[s0, s1] -> (s1 * 8, s0)>
#map23 = affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>
#map24 = affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>
#map25 = affine_map<()[s0, s1] -> (s0 - s1 * 8)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @recompute_w_u_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c32768 = arith.constant 32768 : index
    return %c32768 : index
  }
  func.func @recompute_w_u_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @recompute_w_u_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c96_i32 = arith.constant 96 : i32
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c9216_i64 = arith.constant 9216 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c14336_i64 = arith.constant 14336 : i64
    %0 = hivm.hir.pointer_cast(%c5120_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c14336_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c1024_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c10240_i64) : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c9216_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    %6 = arith.muli %arg11, %arg12 : i32
    %7 = arith.muli %6, %arg13 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map()[%11]
    %view = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
    %13 = hivm.hir.get_block_idx -> i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = affine.apply #map1()[%14]
    %view_0 = memref.view %arg2[%15][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7 ffts_base_addr = %arg0
    %16 = arith.index_cast %7 : i32 to index
    %17 = arith.muli %arg13, %arg12 : i32
    %18 = arith.index_cast %arg10 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    scf.for %arg14 = %9 to %7 step %c96_i32  : i32 {
      %19 = arith.index_cast %arg14 : i32 to index
      %20 = affine.min #map2(%19)[%16]
      scf.for %arg15 = %c0 to %20 step %c1 {
        %21 = arith.index_cast %arg14 : i32 to index
        %22 = arith.index_cast %9 : i32 to index
        %23 = affine.apply #map3()[%arg15, %20, %21, %22]
        %24 = arith.index_cast %23 : index to i64
        %25 = arith.cmpi eq, %24, %c1_i64 : i64
        %26 = arith.select %25, %1, %0 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
        %27 = arith.index_cast %arg14 : i32 to index
        %28 = arith.index_cast %9 : i32 to index
        %29 = affine.apply #map3()[%arg15, %20, %27, %28]
        %30 = arith.index_cast %29 : index to i64
        %31 = arith.cmpi eq, %30, %c1_i64 : i64
        %32 = arith.select %31, %3, %2 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>>
        %33 = arith.index_cast %arg14 : i32 to index
        %34 = arith.index_cast %9 : i32 to index
        %35 = affine.apply #map3()[%arg15, %20, %33, %34]
        %36 = arith.index_cast %35 : index to i64
        %37 = arith.cmpi eq, %36, %c1_i64 : i64
        %38 = arith.select %37, %5, %4 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
        %39 = arith.index_cast %arg14 : i32 to index
        %40 = arith.index_cast %9 : i32 to index
        %41 = affine.apply #map4()[%arg15, %20, %39, %40]
        %42 = arith.index_cast %41 : index to i64
        %43 = arith.index_cast %arg14 : i32 to index
        %44 = arith.index_cast %9 : i32 to index
        %45 = affine.apply #map3()[%arg15, %20, %43, %44]
        %46 = arith.index_cast %45 : index to i1
        %47 = arith.select %46, %c3_i64, %c4_i64 : i64
        %48 = arith.index_cast %45 : index to i1
        %49 = arith.select %48, %c1_i64, %c2_i64 : i64
        %50 = arith.index_cast %arg15 : index to i64
        %51 = affine.apply #map5()[%arg15]
        %52 = arith.index_cast %51 : index to i64
        %53 = affine.apply #map6(%arg15)[%19]
        %54 = arith.index_cast %53 : index to i32
        %55 = arith.divsi %54, %arg13 : i32
        %56 = arith.remsi %55, %arg12 : i32
        %57 = arith.divsi %54, %17 : i32
        %58 = arith.remsi %57, %arg11 : i32
        %59 = arith.divsi %56, %c4_i32 : i32
        %60 = arith.remsi %56, %c4_i32 : i32
        %61 = arith.muli %59, %arg10 : i32
        %62 = arith.muli %61, %c4_i32 : i32
        %63 = arith.addi %62, %60 : i32
        %64 = arith.muli %63, %c32_i32 : i32
        %65 = arith.index_cast %64 : i32 to index
        %66 = arith.muli %63, %c64_i32 : i32
        %67 = arith.index_cast %66 : i32 to index
        %68 = arith.muli %58, %c16_i32 : i32
        %69 = arith.maxsi %68, %c0_i32 : i32
        %70 = arith.index_cast %69 : i32 to index
        %71 = arith.muli %63, %c16_i32 : i32
        %72 = arith.index_cast %71 : i32 to index
        %73 = affine.apply #map7()[%72, %70]
        %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%73], sizes: [16, 16], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %74 = arith.subi %c0_i32, %68 : i32
        %75 = arith.maxsi %74, %c0_i32 : i32
        %76 = arith.index_cast %75 : i32 to index
        %77 = affine.max #map8()[%18, %70]
        %78 = affine.min #map9()[%77]
        %79 = affine.min #map10()[%76, %77]
        %80 = affine.apply #map11()[%78, %79]
        %81 = arith.cmpi slt, %80, %c16 : index
        %subview = memref.subview %reinterpret_cast[0, 0] [%80, 16] [1, 1] : memref<16x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %82 = affine.apply #map12()[%78, %79]
        %83 = affine.apply #map13()[%79]
        %84 = affine.apply #map14()[%79]
        %subview_1 = memref.subview %38[0, %83, %84, 0] [2, %82, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %81 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %38 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          %cast_25 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_25) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %49]
        %cast = memref.cast %subview : memref<?x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_2 = memref.cast %subview_1 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast, %cast_2) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        %85 = affine.apply #map15()[%65, %70]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%85], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %view[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
        %cast_5 = memref.cast %subview_4 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_6 = memref.cast %32 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_5, %cast_6) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %52 ffts_base_addr = %arg0
        %86 = hivm.hir.pointer_cast(%c0_i64) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        %cast_7 = memref.cast %38 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_8 = memref.cast %32 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_9 = memref.cast %86 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_7, %cast_8, %true, %80, %c16, %c64, %cast_9, %c0_i64, %c1_i64, %c-1_i64, %c-1_i64, %42, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %subview_10 = memref.subview %86[0, %83, %84, 0] [2, %82, 16, 16] [1, 1, 1, 1] : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<2x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
        %subview_11 = memref.subview %reinterpret_cast_3[0, 0] [%80, 32] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_12 = memref.cast %subview_10 : memref<2x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_13 = memref.cast %subview_11 : memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_12, %cast_13, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        %87 = affine.apply #map16()[%67, %70]
        %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%87], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_15 = memref.subview %view_0[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %47]
        %cast_16 = memref.cast %subview_15 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_17 = memref.cast %26 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_16, %cast_17) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %50 ffts_base_addr = %arg0
        %88 = hivm.hir.pointer_cast(%c4096_i64) : memref<4x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        %cast_18 = memref.cast %38 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_19 = memref.cast %26 : memref<8x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_20 = memref.cast %88 : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_18, %cast_19, %true, %80, %c16, %c64, %cast_20, %c-1_i64, %c0_i64, %49, %47, %42, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %subview_21 = memref.subview %88[0, %83, %84, 0] [4, %82, 16, 16] [1, 1, 1, 1] : memref<4x1x16x16xf32, #hivm.address_space<cc>> to memref<4x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
        %subview_22 = memref.subview %reinterpret_cast_14[0, 0] [%80, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_23 = memref.cast %subview_21 : memref<4x?x16x16xf32, strided<[256, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_24 = memref.cast %subview_22 : memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_23, %cast_24, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
    }
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
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
    %c5120_i64 = arith.constant 5120 : i64
    %c4864_i64 = arith.constant 4864 : i64
    %c2816_i64 = arith.constant 2816 : i64
    %c2560_i64 = arith.constant 2560 : i64
    %c512_i64 = arith.constant 512 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i64 = arith.constant 0 : i64
    %c96_i32 = arith.constant 96 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.muli %arg11, %arg12 : i32
    %1 = arith.muli %0, %arg13 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.index_cast %1 : i32 to index
    %5 = arith.muli %arg13, %arg12 : i32
    %6 = arith.index_cast %arg10 : i32 to index
    %7 = hivm.hir.get_sub_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map17()[%8]
    hivm.hir.set_ffts_base_addr %arg0
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map()[%11]
    %view = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
    %13 = hivm.hir.get_block_idx -> i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = affine.apply #map1()[%14]
    %view_0 = memref.view %arg2[%15][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x64xf32, #hivm.address_space<gm>>
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg14 = %3 to %1 step %c96_i32  : i32 {
      %16 = arith.index_cast %arg14 : i32 to index
      %17 = affine.min #map2(%16)[%4]
      scf.for %arg15 = %c0 to %17 step %c1 {
        %18 = arith.index_cast %arg15 : index to i64
        %19 = affine.apply #map5()[%arg15]
        %20 = affine.apply #map6(%arg15)[%16]
        %21 = arith.index_cast %19 : index to i64
        %22 = arith.index_cast %20 : index to i32
        %23 = arith.divsi %22, %arg13 : i32
        %24 = arith.remsi %23, %arg12 : i32
        %25 = arith.divsi %22, %5 : i32
        %26 = arith.remsi %25, %arg11 : i32
        %27 = arith.divsi %24, %c4_i32 : i32
        %28 = arith.remsi %24, %c4_i32 : i32
        %29 = arith.muli %27, %arg10 : i32
        %30 = arith.muli %29, %c4_i32 : i32
        %31 = arith.addi %30, %28 : i32
        %32 = arith.muli %31, %c32_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        %34 = arith.muli %31, %c64_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.index_cast %30 : i32 to index
        %37 = arith.index_cast %28 : i32 to index
        %38 = arith.muli %26, %c16_i32 : i32
        %39 = arith.maxsi %38, %c0_i32 : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = affine.apply #map18()[%40, %36, %37]
        %42 = affine.apply #map16()[%35, %40]
        %43 = affine.apply #map15()[%33, %40]
        %44 = affine.max #map8()[%6, %40]
        %45 = affine.min #map9()[%44]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%41], sizes: [16, 1], strides: [4, 4] : memref<?xf32, #hivm.address_space<gm>> to memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
        %46 = arith.index_cast %31 : i32 to index
        %47 = affine.apply #map19()[%46, %40]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [%47], sizes: [16], strides: [4] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
        %48 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x1xf32, #hivm.address_space<ub>>
        %subview = memref.subview %48[0, 0, 0] [8, 1, 1] [1, 1, 1] : memref<8x8x1xf32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %49 = arith.subi %c0_i32, %38 : i32
        %50 = arith.maxsi %49, %c0_i32 : i32
        %51 = arith.index_cast %50 : i32 to index
        %52 = affine.min #map10()[%51, %44]
        %53 = affine.apply #map11()[%45, %52]
        %54 = arith.cmpi slt, %53, %c16 : index
        %55 = affine.min #map20()[%8, %45, %52]
        %56 = affine.min #map21()[%55, %45, %52]
        %subview_2 = memref.subview %reinterpret_cast[%9, 0] [8, 1] [1, 1] : memref<16x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %subview_2[0, 0] [%56, 1] [1, 1] : memref<8x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> to memref<?x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>>
        %57 = affine.max #map22()[%52, %8]
        %58 = affine.min #map23()[%44, %8]
        %59 = affine.max #map24()[%58, %52, %8]
        %60 = affine.apply #map11()[%59, %57]
        %61 = affine.apply #map25()[%57, %8]
        %subview_4 = memref.subview %subview[%61, 0] [%60, 1] [1, 1] : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
        scf.if %54 {
          %collapse_shape_63 = memref.collapse_shape %subview [[0, 1]] : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
          %base_buffer_64, %offset_65, %sizes_66, %strides_67 = memref.extract_strided_metadata %collapse_shape_63 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
          %reinterpret_cast_68 = memref.reinterpret_cast %base_buffer_64 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %cast_69 = memref.cast %reinterpret_cast_68 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_2d(%cst, %cast_69) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %collapse_shape = memref.collapse_shape %subview_3 [[0, 1]] : memref<?x1xf32, strided<[4, 4], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_5 = memref.collapse_shape %subview_4 [[0, 1]] : memref<?x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %cast = memref.cast %collapse_shape : memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_6 = memref.cast %collapse_shape_5 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast, %cast_6, %c2_i32, %cst, %52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %62 = hivm.hir.pointer_cast(%c256_i64) : memref<8x8xf32, #hivm.address_space<ub>>
        %subview_7 = memref.subview %62[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
        %subview_8 = memref.subview %reinterpret_cast_1[%9] [8] [1] : memref<16xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %subview_8[0] [%56] [1] : memref<8xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %subview_7[%61] [%60] [1] : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
        scf.if %54 {
          %base_buffer_63, %offset_64, %sizes_65, %strides_66 = memref.extract_strided_metadata %subview_7 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
          %reinterpret_cast_67 = memref.reinterpret_cast %base_buffer_63 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_68 = memref.cast %reinterpret_cast_67 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_2d(%cst, %cast_68) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        %cast_11 = memref.cast %subview_9 : memref<?xf32, strided<[4], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_12 = memref.cast %subview_10 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_11, %cast_12, %c2_i32, %cst, %52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %63 = hivm.hir.pointer_cast(%c256_i64) : memref<8x8xf32, #hivm.address_space<ub>>
        %subview_13 = memref.subview %63[0, 0] [8, 1] [1, 1] : memref<8x8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_7 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %subview_13 : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [0], sizes: [8, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %64 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_20 = memref.cast %reinterpret_cast_14 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_21 = memref.cast %reinterpret_cast_19 : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_22 = memref.cast %64 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_2d_float(%cast_20, %cast_21, %cast_22) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %reinterpret_cast_23 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [16, 64], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %65 = hivm.hir.pointer_cast(%c512_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %subview_24 = memref.subview %reinterpret_cast_23[%9, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_25 = memref.subview %subview_24[0, 0] [%56, 32] [1, 1] : memref<8x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_26 = memref.subview %65[%61, 0] [%60, 32] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_27 = memref.collapse_shape %65 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
        %cast_28 = memref.cast %collapse_shape_27 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_28) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_29 = memref.cast %subview_25 : memref<?x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_30 = memref.cast %subview_26 : memref<?x32xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_29, %cast_30, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %66 = hivm.hir.pointer_cast(%c512_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %67 = hivm.hir.pointer_cast(%c2560_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_31 = memref.cast %65 : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_32 = memref.cast %subview : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_33 = memref.cast %66 : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_34 = memref.cast %67 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_31, %cast_32, %cast_33, %cast_34) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_35 = memref.subview %view[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_36 = memref.collapse_shape %subview_35 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %collapse_shape_36[%9, 0] [8, 64] [1, 1] {to_be_bubbled_slice} : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %21
        %collapse_shape_38 = memref.collapse_shape %66 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        %collapse_shape_39 = memref.collapse_shape %subview_37 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_40 = memref.cast %collapse_shape_38 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %collapse_shape_39 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_40, %cast_41, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
        %reinterpret_cast_42 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [16, 64], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %68 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %subview_43 = memref.subview %reinterpret_cast_42[%9, 0] [8, 64] [1, 1] : memref<16x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_44 = memref.subview %subview_43[0, 0] [%56, 64] [1, 1] : memref<8x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_45 = memref.subview %68[%61, 0] [%60, 64] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
        scf.if %54 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape_63 = memref.collapse_shape %68 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %cast_64 = memref.cast %collapse_shape_63 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_64) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_46 = memref.cast %subview_44 : memref<?x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_47 = memref.cast %subview_45 : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_46, %cast_47, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %69 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %70 = hivm.hir.pointer_cast(%c4864_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_48 = memref.cast %68 : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_49 = memref.cast %subview : memref<8x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_50 = memref.cast %69 : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_51 = memref.cast %70 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_48, %cast_49, %cast_50, %cast_51) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %expand_shape = memref.expand_shape %subview_13 [[0, 1]] output_shape [8, 1] : memref<8xf32, strided<[8]>, #hivm.address_space<ub>> into memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
        %71 = hivm.hir.pointer_cast(%c2816_i64) : memref<8x64xf32, #hivm.address_space<ub>>
        %72 = hivm.hir.pointer_cast(%c5120_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_52 = memref.cast %69 : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_53 = memref.cast %expand_shape : memref<8x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_54 = memref.cast %71 : memref<8x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_55 = memref.cast %72 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_52, %cast_53, %cast_54, %cast_55) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        %subview_56 = memref.subview %view_0[%arg15, 0, 0] [1, 16, 64] [1, 1, 1] : memref<4x16x64xf32, #hivm.address_space<gm>> to memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_57 = memref.collapse_shape %subview_56 [[0, 1], [2]] : memref<1x16x64xf32, strided<[1024, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_58 = memref.subview %collapse_shape_57[%9, 0] [8, 64] [1, 1] {to_be_bubbled_slice} : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %18
        %collapse_shape_59 = memref.collapse_shape %71 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        %collapse_shape_60 = memref.collapse_shape %subview_58 [[0, 1]] : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_61 = memref.cast %collapse_shape_59 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_62 = memref.cast %collapse_shape_60 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_61, %cast_62, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
    }
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
