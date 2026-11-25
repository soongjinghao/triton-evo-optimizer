#map = affine_map<(d0) -> (d0 * 65536)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
#map6 = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 + (s2 floordiv 4) * s1 + ((s4 - s5) floordiv 24) * (s1 * ((s3 + 3) floordiv 4))) mod 2)>
#map7 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 + (s2 floordiv 4) * s1 + ((s4 - s5) floordiv 24) * (s1 * ((s3 + 3) floordiv 4)))>
#map8 = affine_map<(d0)[s0] -> (d0 + s0)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1)>
#map10 = affine_map<()[s0] -> (64, s0)>
#map11 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map12 = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
#map13 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map14 = affine_map<()[s0, s1] -> (s0 * s1)>
#map15 = affine_map<()[s0] -> (s0 + 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c65536 = arith.constant 65536 : index
    return %c65536 : index
  }
  func.func @matmul_kernel_simplified_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @matmul_kernel_simplified_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c24_i32 = arith.constant 24 : i32
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
    hivm.hir.set_ffts_base_addr %arg0
    %4 = arith.muli %arg12, %arg13 : i32
    %5 = arith.muli %4, %arg14 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
    %8 = arith.muli %arg14, %arg13 : i32
    %9 = arith.addi %arg6, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %11 = arith.addi %arg7, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %12, %c8_i32 : i32
    %14 = arith.addi %arg8, %c63_i32 : i32
    %15 = arith.divsi %14, %c64_i32 : i32
    %16 = arith.muli %arg10, %c64_i32 : i32
    %17 = arith.index_cast %15 : i32 to index
    %18 = arith.index_cast %arg10 : i32 to index
    %19 = arith.index_cast %arg7 : i32 to index
    %20 = arith.index_cast %16 : i32 to index
    %21 = arith.index_cast %arg11 : i32 to index
    %22 = arith.index_cast %arg6 : i32 to index
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
      %33 = arith.muli %30, %c64_i32 : i32
      %34 = arith.muli %32, %c64_i32 : i32
      %35 = arith.index_cast %33 : i32 to index
      %36 = arith.index_cast %34 : i32 to index
      %37 = hivm.hir.get_block_idx -> i64
      %38 = arith.index_cast %37 : i64 to index
      %39 = affine.apply #map(%38)
      %view = memref.view %arg2[%39][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x64x64xf32, #hivm.address_space<gm>>
      %40 = affine.max #map1()[%36, %19]
      %41 = affine.min #map2()[%40, %36]
      %42 = affine.min #map3()[%41, %36]
      %43 = affine.max #map4()[%42]
      %44 = arith.cmpi slt, %43, %c64 : index
      %45 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      %46 = scf.for %arg16 = %c0_i32 to %15 step %c4_i32 iter_args(%arg17 = %c0) -> (index)  : i32 {
        %54 = arith.index_cast %arg16 : i32 to index
        %55 = affine.min #map5(%54)[%17]
        %56 = scf.for %arg18 = %c0 to %55 step %c1 iter_args(%arg19 = %arg17) -> (index) {
          %57 = arith.index_cast %arg16 : i32 to index
          %58 = arith.index_cast %15 : i32 to index
          %59 = arith.index_cast %arg15 : i32 to index
          %60 = arith.index_cast %7 : i32 to index
          %61 = affine.apply #map6()[%arg18, %55, %57, %58, %59, %60]
          %62 = arith.index_cast %61 : index to i64
          %63 = arith.cmpi eq, %62, %c1_i64 : i64
          %64 = arith.select %63, %1, %0 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
          %65 = arith.index_cast %arg16 : i32 to index
          %66 = arith.index_cast %15 : i32 to index
          %67 = arith.index_cast %arg15 : i32 to index
          %68 = arith.index_cast %7 : i32 to index
          %69 = affine.apply #map6()[%arg18, %55, %65, %66, %67, %68]
          %70 = arith.index_cast %69 : index to i64
          %71 = arith.cmpi eq, %70, %c1_i64 : i64
          %72 = arith.select %71, %3, %2 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
          %73 = arith.index_cast %arg16 : i32 to index
          %74 = arith.index_cast %15 : i32 to index
          %75 = arith.index_cast %arg15 : i32 to index
          %76 = arith.index_cast %7 : i32 to index
          %77 = affine.apply #map7()[%arg18, %55, %73, %74, %75, %76]
          %78 = arith.index_cast %77 : index to i64
          %79 = arith.index_cast %arg16 : i32 to index
          %80 = arith.index_cast %15 : i32 to index
          %81 = arith.index_cast %arg15 : i32 to index
          %82 = arith.index_cast %7 : i32 to index
          %83 = affine.apply #map6()[%arg18, %55, %79, %80, %81, %82]
          %84 = arith.index_cast %83 : index to i1
          %85 = arith.select %84, %c3_i64, %c4_i64 : i64
          %86 = arith.index_cast %83 : index to i1
          %87 = arith.select %86, %c1_i64, %c2_i64 : i64
          %88 = arith.index_cast %arg18 : index to i64
          %89 = affine.apply #map8(%arg18)[%54]
          %90 = arith.index_cast %89 : index to i32
          %91 = arith.muli %90, %c64_i32 : i32
          %92 = arith.subi %arg8, %91 : i32
          %93 = affine.apply #map9()[%arg19, %36]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%93], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %94 = arith.index_cast %92 : i32 to index
          %95 = affine.max #map4()[%94]
          %96 = affine.min #map10()[%95]
          %97 = affine.max #map4()[%96]
          %98 = arith.cmpi slt, %97, %c64 : index
          %subview_3 = memref.subview %view[%arg18, 0, 0] [1, 64, 64] [1, 1, 1] : memref<4x64x64xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %87]
          %cast_4 = memref.cast %subview_3 : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_5 = memref.cast %72 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_float(%cast_4, %cast_5) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %88 ffts_base_addr = %arg0
          %99 = arith.ori %98, %44 : i1
          %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%97, %43] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %100 = affine.apply #map11()[%97]
          %101 = affine.apply #map12()[%43]
          %subview_7 = memref.subview %64[0, 0, 0, 0] [%101, %100, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
          scf.if %99 {
            hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
            %collapse_shape = memref.collapse_shape %64 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
            hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
            %cast_13 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
            func.call @set_l1_2d_float(%cst, %cast_13) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
            hivm.hir.pipe_barrier[<PIPE_MTE2>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %85]
          %cast_8 = memref.cast %subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_9 = memref.cast %subview_7 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_float(%cast_8, %cast_9) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
          %102 = arith.cmpi eq, %90, %c0_i32 : i32
          %cast_10 = memref.cast %72 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_11 = memref.cast %64 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_12 = memref.cast %45 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          func.call @mma_tile_float_to_float(%cast_10, %cast_11, %102, %c64, %c64, %43, %cast_12, %c0_i64, %c1_i64, %87, %85, %78, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
          %103 = affine.apply #map9()[%arg19, %20]
          scf.yield %103 : index
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
        scf.yield %56 : index
      }
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %47 = affine.apply #map13()[%36, %35, %21]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%47], sizes: [64, 64], strides: [%21, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %48 = affine.max #map1()[%35, %22]
      %49 = affine.min #map2()[%48, %35]
      %50 = affine.min #map3()[%49, %35]
      %51 = affine.max #map4()[%50]
      %52 = affine.apply #map11()[%51]
      %53 = affine.apply #map11()[%43]
      %subview = memref.subview %45[0, 0, 0, 0] [%53, %52, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
      %subview_0 = memref.subview %reinterpret_cast[0, 0] [%51, %43] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %cast = memref.cast %subview : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
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
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @set_l1_2d_float(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @matmul_kernel_simplified_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i64 = arith.constant 0 : i64
    %c24_i32 = arith.constant 24 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.muli %arg12, %arg13 : i32
    %1 = arith.muli %0, %arg14 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg14, %arg13 : i32
    %5 = arith.addi %arg6, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.addi %arg7, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.addi %arg8, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32 : i32
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = arith.index_cast %11 : i32 to index
    %14 = arith.index_cast %arg6 : i32 to index
    %15 = hivm.hir.get_sub_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = arith.cmpi eq, %16, %c0 : index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %18 = arith.divsi %arg15, %4 : i32
      %19 = arith.remsi %18, %arg12 : i32
      %20 = arith.divsi %19, %9 : i32
      %21 = arith.muli %20, %c8_i32 : i32
      %22 = arith.subi %6, %21 : i32
      %23 = arith.minsi %22, %c8_i32 : i32
      %24 = arith.remsi %19, %23 : i32
      %25 = arith.addi %21, %24 : i32
      %26 = arith.muli %25, %c64_i32 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = affine.apply #map14()[%27, %12]
      %29 = hivm.hir.get_block_idx -> i64
      %30 = arith.index_cast %29 : i64 to index
      %31 = affine.apply #map(%30)
      %view = memref.view %arg2[%31][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x64x64xf32, #hivm.address_space<gm>>
      %32 = affine.max #map1()[%27, %14]
      %33 = affine.min #map2()[%32, %27]
      %34 = affine.min #map3()[%33, %27]
      %35 = affine.max #map4()[%34]
      %36 = arith.cmpi slt, %35, %c64 : index
      %37 = scf.for %arg16 = %c0_i32 to %11 step %c4_i32 iter_args(%arg17 = %28) -> (index)  : i32 {
        %38 = arith.index_cast %arg16 : i32 to index
        %39 = affine.min #map5(%38)[%13]
        %40 = scf.for %arg18 = %c0 to %39 step %c1 iter_args(%arg19 = %arg17) -> (index) {
          %41 = arith.index_cast %arg18 : index to i64
          %42 = affine.apply #map8(%arg18)[%38]
          %43 = arith.index_cast %42 : index to i32
          %44 = arith.muli %43, %c64_i32 : i32
          %45 = arith.subi %arg8, %44 : i32
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%arg19], sizes: [64, 64], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %46 = hivm.hir.pointer_cast(%c0_i64) : memref<64x64xf32, #hivm.address_space<ub>>
          %47 = arith.index_cast %45 : i32 to index
          %48 = affine.max #map4()[%47]
          %49 = affine.min #map10()[%48]
          %50 = affine.max #map4()[%49]
          %51 = arith.cmpi slt, %50, %c64 : index
          %52 = arith.ori %36, %51 : i1
          %subview = memref.subview %reinterpret_cast[0, 0] [%35, %50] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_0 = memref.subview %46[0, 0] [%35, %50] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
          scf.if %52 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_3 = memref.collapse_shape %46 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_4 = memref.cast %collapse_shape_3 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_4) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          %cast = memref.cast %subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_1 = memref.cast %subview_0 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast, %cast_1, %c2_i32, %cst, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %subview_2 = memref.subview %view[%arg18, 0, 0] [1, 64, 64] [1, 1, 1] : memref<4x64x64xf32, #hivm.address_space<gm>> to memref<1x64x64xf32, strided<[4096, 64, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_2 [[0, 1], [2]] : memref<1x64x64xf32, strided<[4096, 64, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %41
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          scf.if %17 {
            %collapse_shape_3 = memref.collapse_shape %46 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            %collapse_shape_4 = memref.collapse_shape %collapse_shape [[0, 1]] : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %cast_5 = memref.cast %collapse_shape_3 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_6 = memref.cast %collapse_shape_4 : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_5, %cast_6, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          } {limit_sub_block_id0}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
          %53 = affine.apply #map15()[%arg19]
          scf.yield %53 : index
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
        scf.yield %40 : index
      }
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
